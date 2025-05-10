from transformers import GenerationConfig
import torch
from typing import List
import numpy as np
import os
import json

from .utils import system_prompt, user_message_template1, user_message_template2, user_message_template3
from .dataset import ARCDataset
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftConfig, PeftModel
from torch.utils.data import DataLoader
from torch.optim import AdamW

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
        Format training data and test input into LLM input tokens

        Args:
            datapoint (dict): contains training data, test input
        
        Returns:
            prompt (dict): dictionary that contains input ids and additional informations
        """
        train_examples = datapoint['train']
        test_input = datapoint['test'][0]['input']

        # manual ChatML-like encoding
        tokens = []
        # system
        tokens += self.tokenizer.encode(
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}\n",
            add_special_tokens=False)
        # user description
        tokens += self.tokenizer.encode(
            f"<|start_header_id|>user<|end_header_id|>\n{user_message_template1}\n",
            add_special_tokens=False)
        # train examples
        for ex in train_examples:
            inp = self.format_grid(ex['input'])
            out = self.format_grid(ex['output'])
            tokens += self.tokenizer.encode("input:\n", add_special_tokens=False) + inp
            tokens += self.tokenizer.encode("output:\n", add_special_tokens=False) + out
        # test example
        tokens += self.tokenizer.encode(f"\n{user_message_template2}\n", add_special_tokens=False)
        tokens += self.tokenizer.encode("input:\n", add_special_tokens=False)
        tokens += self.format_grid(test_input)
        tokens += self.tokenizer.encode(f"\n{user_message_template3}", add_special_tokens=False)
        # assistant start
        tokens += self.tokenizer.encode(
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            add_special_tokens=False)
        return {"input_ids": tokens,
                "input": test_input,
                "train": train_examples}

    def seq2seq_loss(self, prompt_ids, target_ids):
        """
        Calculate loss for sequence-to-sequence learning.
        Uses teacher forcing to help the model predict the correct next token.

        Args:
            prompt_ids (torch.Tensor): Token IDs of the input prompt
            target_ids (torch.Tensor): Token IDs of the target sequence

        Returns:
            torch.Tensor: Computed loss value
        """
        # Get EOS token ID
        eos = self.tokenizer.eos_token_id
        
        # Append EOS token if missing
        # This helps the model recognize the end of the sequence
        if target_ids[0, -1] != eos:
            target_ids = torch.cat([target_ids, torch.tensor([[eos]], device=target_ids.device)], dim=1)
        
        # Concatenate input and target to create the full sequence
        inp = torch.cat([prompt_ids, target_ids], dim=1)
        
        # Create labels for loss calculation
        # Set prompt portion to -100 to exclude from loss calculation
        labels = inp.clone()
        labels[:, :prompt_ids.size(1)] = -100
        
        # Pass through model to calculate loss
        # Teacher forcing: model is trained to predict the correct next token
        outputs = self.model(input_ids=inp, labels=labels)
        return outputs.loss

    def train(self, train_dataset, batch_size, lr, num_epochs, steps_per_file, steps_accum):
        """
        Train the model using LoRA fine-tuning.
        
        Args:
            train_dataset: HuggingFace dataset containing training examples
            batch_size (int): Number of samples per training batch
            lr (float): Learning rate for the optimizer
            num_epochs (int): Number of training epochs
            steps_per_file (int): Number of training steps per file
            steps_accum (int): Number of steps to accumulate gradients before updating
        """
        # Configure LoRA parameters for efficient fine-tuning
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8,                     # LoRA rank - determines the size of the update matrices
            lora_alpha=32,           # LoRA scaling factor - controls the magnitude of updates
            lora_dropout=0.1,        # Dropout probability for LoRA layers
            target_modules=["q_proj","k_proj","v_proj","o_proj"], # Apply LoRA to attention modules only
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters() # Display the number of trainable parameters
        
        # Initialize dataset and data loader
        dataset = ARCDataset(train_dataset, self.tokenizer, self, steps_per_file=steps_per_file)
        loader = DataLoader(dataset, batch_size)
        
        # Initialize optimizer with specified learning rate
        optimizer = AdamW(self.model.parameters(), lr=lr)
        self.model.train()  # Set model to training mode
        
        # Training loop
        global_step = 0
        for epoch in range(num_epochs):
            total_loss = 0
            # steps = len(dataset) / batch_size
            for step, batch in enumerate(loader):
                global_step += 1
                
                # Move batch to device and compute loss
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                loss = self.seq2seq_loss(input_ids, target_ids)
                
                # Backpropagation
                loss.backward()
                
                # Gradient accumulation and optimization
                if global_step % steps_accum == 0:
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()  # Update model parameters
                    optimizer.zero_grad()  # Clear gradients
                
                # Track and display training progress
                total_loss += loss.item()
                if step % 10 == 0:
                    print(f"[Epoch {epoch+1}] step {step} loss {loss.item():.4f}")
            
            # Print average loss for the epoch
            print(f"Epoch {epoch+1} avg loss {total_loss/len(loader):.4f}")
        
        self.model.eval()  # Set model to evaluation mode after training

    def save_model(self, path="artifacts/qwen3-4b-lora"):
        """
        Save the model and its configuration
        """
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        # save config
        info = {
            'base': self.model.config._name_or_path,
            'type': self.model.config.model_type,
            'hidden_size': int(self.model.config.hidden_size),
            'vocab_size': int(self.model.config.vocab_size),
        }
        with open(os.path.join(path, "model_config.json"), 'w') as f:
            json.dump(info, f, indent=2)

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
        datapoint = {"train": examples, "test": [{"input": questions_input}]}

        # 프롬프트 생성 및 토큰화
        prompt = self.format_prompt(datapoint)
        ids = torch.tensor(prompt['input_ids'], device=self.device).unsqueeze(0) # (1, seq_len)
        attn_mask = torch.ones_like(ids)

        # 생성 설정
        config = GenerationConfig(do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                               bos_token_id=self.tokenizer.bos_token_id,
                               eos_token_id=self.tokenizer.eos_token_id,
                               pad_token_id=self.tokenizer.pad_token_id,
                               max_new_tokens=150)

        # 모델로부터 출력 생성
        out = self.model.generate(input_ids=ids, attention_mask=attn_mask, generation_config=config).squeeze().cpu()
        N_prompt = ids.size(1)

        # 프롬프트 이후의 토큰만 추출
        output = out[N_prompt:].tolist()
        train_input = np.array(prompt['train'][0]['input'])
        train_output = np.array(prompt['train'][0]['output'])
        test_input = np.array(prompt['input'])

        # LLM-generated grid may have wrong shape
        # So adjust shape by input-output pairs
        if train_input.shape == train_output.shape:
            x, y = test_input.shape
        else:
            x = (train_output.shape[0] * test_input.shape[0]) // train_input.shape[0]
            y = (train_output.shape[1] * test_input.shape[1]) // train_input.shape[1]

        try:
            grid = np.array(self.parse_grid(output))
        except Exception as e:
            # 파싱 실패 시 랜덤 그리드 반환
            grid = np.random.randint(0, 10, (x, y))

        return grid

    def prepare_evaluation(self, path="artifacts/qwen3-4b-lora"):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        try:
            peft_conf = PeftConfig.from_pretrained(path)
            self.model = PeftModel.from_pretrained(self.model, path, is_trainable=False)
            print("Loaded LoRA adapter")
        except Exception as e:
            print("No adapter found:", e)
        self.model.eval()

if __name__ == "__main__":
    solver = ARCSolver()
