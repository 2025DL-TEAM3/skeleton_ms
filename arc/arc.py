from transformers import GenerationConfig
import torch
from typing import List
import numpy as np

from .utils import system_prompt, user_message_template1, user_message_template2, user_message_template3
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer


class ARCSolver:
    """
    You should implement a `Solver` class for the project.
    """

    def __init__(self, token=None):
        """
        Args:
            token (str): a huggingface token for restricted models such as llama3
        """
        config_path = "artifacts/config/config.yml"
        model_id = "Qwen/Qwen3-4B"

        # Configure the BitsAndBytes settings for 4-bit quantization to reduce memory usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_use_double_quant=True,  # Use double quantization for improved precision
            bnb_4bit_quant_type="nf4",  # Specify the quantization type
            bnb_4bit_compute_dtype=torch.float16,  # Set the computation data type
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True, # Allow the model to use custom code from the repository
            quantization_config=bnb_config, # Apply the 4-bit quantization configuration
            attn_implementation='sdpa', # Use scaled-dot product attention for better performance
            torch_dtype=torch.float16, # Set the data type for the model
            use_cache=False, # Disable caching to save memory
            device_map='auto', # Automatically map the model to available devices (e.g., GPUs)
            token=token
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)
        ]
        self.sep = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def parse_grid(self, ids: List[int]):
        """
        Parse LLM generated sequence into ARC grid format

        Args:
            ids (List[int]): LLM generated token list

        Returns:
            grid (List[List[int]]): parsed 2D grid
        """
        grid = []
        row = []
        # 토큰 id → 숫자(0~9)로 역변환하는 맵
        inv_map = {k: i for i, k in enumerate(self.pixel_ids)}
        
        for idx in ids:
            if idx == self.sep:
                if len(row) > 0:
                    grid.append(row.copy())
                    row.clear()
            else:
                row.append(inv_map.get(idx, 0)) # 없는 값은 0으로 처리
        return grid

    def format_grid(self, grid: List[List[int]]):
        """
        Format 2D grid into LLM input tokens

        Args:
            grid (List[List[int]]): 2D grid

        Returns:
            ids (List[int]): Token list for LLM
        """
        ids = []

        for row in grid:
            for col in row:
                ids.append(self.pixel_ids[col])
            ids.append(self.sep)  # 한 행 끝마다 줄바꿈
        return ids

    def format_prompt(self, datapoint):
        """
        학습 예시와 테스트 입력을 LLM 입력 프롬프트로 포맷팅
        Args:
            datapoint (dict): contains training data, test input
        
        Returns:
            prompt (dict): dictionary that contains input ids and additional informations
        """

        training_data = datapoint['train']
        input_test_data = datapoint['test'][0]['input']

        # system 프롬프트
        sys = self.tokenizer.encode("<|begin_of_text|><|start_header_id|>system<|end_header_id|>" + "\n" + system_prompt, add_special_tokens=False)
        # user 프롬프트(문제 설명)
        user = self.tokenizer.encode("<|start_header_id|>user<|end_header_id|>" + "\n" + user_message_template1 + "\n", add_special_tokens=False)
        inp_desc = self.tokenizer.encode("input:\n", add_special_tokens=False)
        out_desc = self.tokenizer.encode("output:\n", add_special_tokens=False)
        # 학습 예시 추가
        for ex in training_data:
            inp = ex['input']
            out = ex['output']
            inp = self.format_grid(inp)
            out = self.format_grid(out)

            user += inp_desc
            user += inp
            user += out_desc
            user += out

        # 추가 설명 및 테스트 입력 추가
        user += self.tokenizer.encode("\n" + user_message_template2 + "\n", add_special_tokens=False)
        user += inp_desc
        user += self.format_grid(input_test_data)
        user += self.tokenizer.encode("\n" + user_message_template3, add_special_tokens=False)

        # assistant 역할 시작 토큰 추가
        messages = sys + user
        assis = self.tokenizer.encode("<|eot_id|><|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False)
        messages += assis

        return {
            "input_ids": messages,
            "input": input_test_data,
            "train": training_data
        }

    def train(self, train_dataset):
        """
        Train the model using LoRA fine-tuning
        
        Args:
            train_dataset: HuggingFace dataset containing training examples
        """
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import TrainingArguments, Trainer
        import torch

        # LoRA 설정
        lora_config = LoraConfig(
            r=16,                     # LoRA 랭크
            lora_alpha=32,            # LoRA 알파
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen3-4B의 attention 모듈
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # 모델을 k-bit 학습을 위해 준비
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA 적용
        self.model = get_peft_model(self.model, lora_config)
        
        # 학습 인자 설정
        training_args = TrainingArguments(
            output_dir="artifacts/checkpoint-final",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            warmup_ratio=0.1,
        )

        # 데이터셋 전처리
        def preprocess_function(examples):
            prompts = []
            for train_examples, test_example in zip(examples["train"], examples["test"]):
                datapoint = {
                    "train": train_examples,
                    "test": [{"input": test_example["input"]}]
                }
                prompt = self.format_prompt(datapoint)
                prompts.append(prompt["input_ids"])
            return {"input_ids": prompts}

        # 데이터셋 전처리
        processed_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )

        # Trainer 설정
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_dataset,
            data_collator=lambda data: {"input_ids": torch.stack([torch.tensor(f) for f in data["input_ids"]])}
        )

        # 학습 실행
        trainer.train()

        # 모델 저장
        trainer.save_model()

    def predict(self, examples, questions_input):
        """
        A single example of test data is given.
        You should predict 2D grid (List[List[int]] or np.ndarray)

        Args:
            examples (List[dict]): List of training examples,
                each list element is a dictionary that contains "input" and "output"
                for example,
                [
                    {
                        "input": [[1,2],[3,4]],
                        "output": [[4,5],[6,7]],
                    },
                    {
                        "input": [[0,1],[2,3]],
                        "output": [[3,4],[5,6]],
                    }
                ]
            questions_input (List[List[int]]): A 2d grid,
                which is a input for a given question
        Returns:
            output (List[List[int]]): A 2d grid,
                which is the output of given input question.
        """
        # 프롬프트 데이터 구성
        datapoint = {
            "train": examples,
            "test": [
                {
                    "input": questions_input
                }
            ]
        }

        # 프롬프트 생성 및 토큰화
        prompt = self.format_prompt(datapoint)
        input_ids = torch.tensor(prompt['input_ids'], dtype=torch.long).to(self.device).view(1, -1)

        # 생성 설정
        config = GenerationConfig(
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=150,
        )

        # 모델로부터 출력 생성
        output = self.model.generate(
            input_ids=input_ids,
            generation_config=config,
        ).squeeze().cpu()
        N_prompt = input_ids.numel()

        # 프롬프트 이후의 토큰만 추출
        output = output[N_prompt:].tolist()
        train_input = np.array(prompt['train'][0]['input'])
        train_output = np.array(prompt['train'][0]['output'])
        test_input = np.array(prompt['input'])

        # LLM-generated grid may have wrong shape
        # So adjust shape by input-output pairs
        if train_input.shape == train_output.shape:
            x, y = test_input.shape
        else:
            x = (train_output.shape[0] // train_input.shape[0]) * test_input.shape[0]
            y = (train_output.shape[1] // train_input.shape[1]) * test_input.shape[1]

        try:
            grid = np.array(self.parse_grid(output))
            grid = grid[:x, :y]  # shape 맞추기
        except Exception as e:
            # 파싱 실패 시 랜덤 그리드 반환
            grid = np.random.randint(0, 10, (x, y))

        return grid

    def prepare_evaluation(self):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        # self.model.load_adapter("artifacts/checkpoint-final")
        self.model.eval()


if __name__ == "__main__":
    solver = ARCSolver()
