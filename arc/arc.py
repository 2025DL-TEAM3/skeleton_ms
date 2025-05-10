import os
os.environ["HF_HOME"]            = "/2025pdp/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/2025pdp/.cache"
os.environ["HF_DATASETS_CACHE"]  = "/2025pdp/.cache/huggingface/datasets"
os.environ["HF_METRICS_CACHE"]   = "/2025pdp/.cache/huggingface/metrics"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "true"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 이미 다운로드된 경우만 사용

from transformers import GenerationConfig
import torch
from typing import List
import numpy as np
import json

from .utils import system_prompt, user_message_template1, user_message_template2, user_message_template3
from .dataset import ARCDataset
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 허깅페이스 캐시 디렉토리 설정
        cache_dir = "/2025pdp/.cache"
        os.makedirs(cache_dir, exist_ok=True)

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
            token=token,
            cache_dir=cache_dir  # 캐시 디렉토리 지정
        ).to(self.device)  # Move model to device

        ###### add this if you get OOM error
        # self.model.gradient_checkpointing_enable()
        # self.model.config.use_cache = False
        # if hasattr(self.model, "enable_input_require_grads"):
        #     self.model.enable_input_require_grads()
        ######

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            token=token,
            cache_dir=cache_dir,  # 토크나이저도 동일한 캐시 디렉토리 사용
            padding_side='left', # decoder-only architecture는 왼쪽 padding 사용
        )

        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)
        ]
        self.sep = self.tokenizer.encode("\n", add_special_tokens=False)[0]

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

    def dynamic_collate(self, batch):
        """
        Custom collate function to handle variable-length sequences
        """
        # 1) get input_ids and target_ids
        input_ids = [item['input_ids'] for item in batch]
        target_ids = [item['target_ids'] for item in batch]

        # 2) get max length
        pad_id = self.tokenizer.pad_token_id
        max_in_len = max(len(x) for x in input_ids)
        max_tgt_len = max(len(x) for x in target_ids)

        # 3) padding tensors
        batch_size = len(input_ids)
        padding_input_ids = torch.full((batch_size, max_in_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_in_len), dtype=torch.long)
        padding_target_ids = torch.full((batch_size, max_tgt_len), pad_id, dtype=torch.long)

        # 4) fill tensors
        for i, seq in enumerate(input_ids):
            seq_len = seq.size(0)
            start_idx = max_in_len - seq_len
            padding_input_ids[i, start_idx:] = seq
            attention_mask[i, start_idx:] = 1

        for i, seq in enumerate(target_ids):
            seq_len = seq.size(0)
            start_idx = max_tgt_len - seq_len
            padding_target_ids[i, start_idx:] = seq

        return {
            "input_ids": padding_input_ids,
            "target_ids": padding_target_ids,
            "attention_mask": attention_mask
        }

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
        # Concatenate input and target
        inp = torch.cat([prompt_ids, target_ids], dim=1)

        # Attention_mask: ignore padding tokens
        attn_mask = inp.ne(self.tokenizer.pad_token_id).long()

        # Create labels: -100 for prompt portion, padding portion
        labels = inp.clone()
        labels[:, :prompt_ids.size(1)] = -100
        labels[inp == self.tokenizer.pad_token_id] = -100

        # Pass through model to calculate loss: Teacher forcing
        outputs = self.model(input_ids=inp, attention_mask=attn_mask, labels=labels)
        return outputs.loss

    def train(self, train_dataset, batch_size, lr, num_epochs, steps_per_file, steps_accum, warmup_rate, 
               save_dir: str = 'artifacts/qwen3-4b-lora', resume_from: str = None, validation_dataset=None, 
               patience=5, val_steps=1000, val_seed=42, max_val_files=50, val_steps_per_file=2, val_batch_size=4):
        """
        Train the model using LoRA fine-tuning.
        
        Args:
            train_dataset: HuggingFace dataset containing training examples
            batch_size (int): Number of samples per training batch
            lr (float): Learning rate for the optimizer
            num_epochs (int): Number of training epochs
            steps_per_file (int): Number of training steps per file
            steps_accum (int): Number of steps to accumulate gradients before updating
            warmup_rate (float): Warmup rate for the learning rate
            save_dir (str): Directory to save the model and optimizer state
            resume_from (str): Path to resume training from checkpoint
            validation_dataset: Dataset for validation during training
            patience (int): Number of validation checks without improvement before early stopping
            val_steps (int): Number of steps between validation checks
            val_seed (int): Random seed for validation sampling
            max_val_files (int): Maximum number of files to use for validation
            val_steps_per_file (int): Steps per file for validation (less than training)
            val_batch_size (int): Batch size for validation
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 로깅 파일 설정
        log_file = os.path.join(save_dir, "training_log.txt")
        
        # 로깅 함수 정의
        def log_message(message, print_to_console=True):
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{message}\n")
            if print_to_console:
                print(message)
        
        # 학습 시작 정보 로깅
        timestamp = torch.cuda.current_device() if torch.cuda.is_available() else "CPU"
        log_message(f"===== Training started at {timestamp} =====")
        log_message(f"Batch size: {batch_size}, LR: {lr}, Epochs: {num_epochs}")
        log_message(f"Steps per file: {steps_per_file}, Grad. accum: {steps_accum}, Warmup rate: {warmup_rate}")
        
        # Configure LoRA parameters for efficient fine-tuning
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=16,                     # LoRA rank - determines the size of the update matrices
            lora_alpha=32,           # LoRA scaling factor - controls the magnitude of updates
            lora_dropout=0.1,        # Dropout probability for LoRA layers
            target_modules=["q_proj","k_proj","v_proj","o_proj"], # Apply LoRA to attention modules only
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters() # Display the number of trainable parameters
        
        # Initialize dataset and data loader
        dataset = ARCDataset(train_dataset, self.tokenizer, self, steps_per_file=steps_per_file, is_validation=False)
        loader = DataLoader(dataset, batch_size, shuffle=True, pin_memory=True, collate_fn=self.dynamic_collate)

        # Initialize validation dataset and data loader if provided
        val_loader = None
        if validation_dataset is not None:
            val_dataset = ARCDataset(validation_dataset, self.tokenizer, self, 
                                    steps_per_file=val_steps_per_file,  # 검증에는 더 적은 스텝 사용
                                    is_validation=True,  # 결정적 샘플링 활성화
                                    seed=val_seed,       # 고정된 시드 사용
                                    max_val_files=max_val_files)  # 검증에 사용할 최대 파일 수
            val_loader = DataLoader(val_dataset, val_batch_size, shuffle=False, pin_memory=True, collate_fn=self.dynamic_collate)
            log_message(f"Validation dataset loaded with {len(val_dataset)} steps, using seed {val_seed}, max files {max_val_files}, steps/file {val_steps_per_file}, batch size {val_batch_size}")
            log_message(f"Total validation batches: {len(val_loader)}")

        # Initialize optimizer with specified learning rate
        optimizer = AdamW(self.model.parameters(), lr=lr)

        # warmup steps
        total_steps = num_epochs * len(dataset) // batch_size
        warmup_steps = int(total_steps * warmup_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        start_epoch, global_step = 0, 0
        if resume_from:
            start_epoch, global_step = self.load_checkpoint(resume_from, optimizer, scheduler)
            log_message(f"Resuming from {resume_from}: epoch {start_epoch} and global step {global_step}")

        # Early stopping variables
        best_val_loss = float('inf')
        val_loss_history = []
        patience_counter = 0
        best_model_path = os.path.join(save_dir, "checkpoint-best")
        
        # CSV 형식의 메트릭 로그 파일 생성
        metrics_log_file = os.path.join(save_dir, "metrics_log.txt")
        with open(metrics_log_file, 'w', encoding='utf-8') as f:
            f.write("global_step,epoch,train_loss,val_loss,val_accuracy,learning_rate\n")

        # Set model to training mode
        self.model.train()
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            total_loss = 0
            # steps = len(dataset) / batch_size
            for step, batch in enumerate(loader):
                global_step += 1
                
                # Move batch to device and compute loss
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                loss = self.seq2seq_loss(input_ids, target_ids) / steps_accum
                
                # Backpropagation
                loss.backward()
                
                # Gradient accumulation and optimization
                if global_step % steps_accum == 0:
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()  # Update model parameters
                    scheduler.step()  # Update learning rate
                    optimizer.zero_grad()  # Clear gradients
                
                # Track and display training progress
                total_loss += loss.item()
                current_lr = scheduler.get_last_lr()[0]
                
                if step % 100 == 0:
                    log_message(f"[Epoch {epoch+1}] step {step} loss {loss.item():.4f} lr {current_lr:.6f}")

                # Validation check
                if val_loader is not None and global_step % val_steps == 0:
                    val_loss, val_accuracy = self.validate(val_loader)
                    log_message(f"[Validation] global_step {global_step} loss {val_loss:.4f} accuracy {val_accuracy:.4f}")
                    
                    # 메트릭 로깅
                    with open(metrics_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"{global_step},{epoch+1},{loss.item():.6f},{val_loss:.6f},{val_accuracy:.6f},{current_lr:.8f}\n")
                    
                    # Early stopping check
                    val_loss_history.append(val_loss)
                    
                    # If validation loss improved
                    if val_loss < best_val_loss:
                        log_message(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
                        best_val_loss = val_loss
                        # Save best model
                        self.save_model(best_model_path, optimizer, scheduler, epoch, global_step)
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        log_message(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
                    
                    # Early stopping
                    if patience_counter >= patience:
                        log_message(f"Early stopping triggered after {patience} validation checks without improvement")
                        # Load best model
                        self.load_checkpoint(best_model_path, optimizer, scheduler)
                        return
                    
                    # Set model back to training mode after validation
                    self.model.train()

                # Save checkpoint every 5000 steps
                if global_step % 5000 == 0:
                    self.save_model(os.path.join(save_dir, f"checkpoint-{global_step}"), optimizer, scheduler, epoch, global_step)
            
            # Print average loss for the epoch
            avg_epoch_loss = total_loss/len(loader)
            log_message(f"Epoch {epoch+1} avg loss {avg_epoch_loss:.4f}")
            
            # 에폭 완료 후 메트릭 로깅
            if val_loader is not None:
                val_loss, val_accuracy = self.validate(val_loader)
                log_message(f"[Epoch End Validation] epoch {epoch+1} loss {val_loss:.4f} accuracy {val_accuracy:.4f}")
                with open(metrics_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{global_step},{epoch+1},{avg_epoch_loss:.6f},{val_loss:.6f},{val_accuracy:.6f},{current_lr:.8f}\n")
        
        self.model.eval()  # Set model to evaluation mode after training
        log_message("===== Training completed =====")

        # save final model
        self.save_model(os.path.join(save_dir, "checkpoint-final"), optimizer, scheduler, epoch, global_step)

    def validate(self, val_loader):
        """
        Validate the model on the validation dataset
        
        Args:
            val_loader: DataLoader for validation dataset
            
        Returns:
            avg_loss (float): Average loss on validation dataset
            accuracy (float): Accuracy on validation dataset
        """
        print(f"=== Starting validation (total {len(val_loader)} batches) ===")
        self.model.eval()
        total_loss = 0
        total_samples = 0
        correct_predictions = 0
        config = GenerationConfig(do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
            bos_token_id= 151643,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=150)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # 진행 상황 출력 (20배치마다)
                if batch_idx % 20 == 0 or batch_idx == len(val_loader) - 1:
                    print(f"Validation progress: {batch_idx+1}/{len(val_loader)} batches")
                
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                # Compute validation loss
                loss = self.seq2seq_loss(input_ids, target_ids)
                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)

                # get attention mask (should ignore padding tokens when generating)
                attn_mask = batch['attention_mask'].to(self.device)
                
                # Generate outputs for accuracy calculation
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    generation_config=config,
                )
                
                # Calculate accuracy by comparing predictions with targets
                # This is a simple token-level accuracy metric
                for i in range(input_ids.size(0)):
                    pred_tokens = outputs[i, input_ids.size(1):].tolist()
                    target_tokens = target_ids[i].tolist()
                    
                    # Remove padding tokens from target
                    target_tokens = [t for t in target_tokens if t != self.tokenizer.pad_token_id]
                    
                    # Compare predicted grid with target grid
                    pred_grid = self.parse_grid(pred_tokens)
                    target_grid = self.parse_grid(target_tokens)
                    
                    # Check if grids are same shape and values
                    if len(pred_grid) == len(target_grid):
                        all_match = True
                        for row_idx in range(len(pred_grid)):
                            if row_idx < len(target_grid):
                                if pred_grid[row_idx] != target_grid[row_idx]:
                                    all_match = False
                                    break
                            else:
                                all_match = False
                                break
                        
                        if all_match:
                            correct_predictions += 1
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        print(f"=== Validation complete: loss {avg_loss:.4f}, accuracy {accuracy:.4f} ===")
        return avg_loss, accuracy

    def load_checkpoint(self, path, optimizer=None, scheduler=None):
        """
        Load the model and its configuration
        """
        # 1) model + PEFT adapter load
        self.model = PeftModel.from_pretrained(self.model, path, is_trainable=True).to(self.device)
        # 2) optimizer / scheduler state load
        opt_path = os.path.join(path, "optimizer.pth")
        if optimizer is not None and os.path.isfile(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
        sch_path = os.path.join(path, "scheduler.pth")
        if scheduler is not None and os.path.isfile(sch_path):
            scheduler.load_state_dict(torch.load(sch_path, map_location=self.device))
        # 3) epoch / global_step restore (optional)
        state_path = os.path.join(path, "training_state.json")
        start_epoch, start_step = 0, 0
        if os.path.isfile(state_path):
            st = json.load(open(state_path))
            start_epoch = st.get('epoch', 0)
            start_step  = st.get('global_step', 0)
        return start_epoch, start_step

    def save_model(self, path=None, optimizer=None, scheduler=None, epoch=None, global_step=None):
        """
        Save the model and its configuration
        """
        if path is None:
            path = "artifacts/qwen3-4b-lora/checkpoint-final"
        os.makedirs(path, exist_ok=True)
        # save model weight + PEFT adapter
        self.model.save_pretrained(path)
        # save optimizer + scheduler state
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pth"))
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(path, "scheduler.pth"))
        # training state
        state = {}
        if epoch is not None:
            state['epoch'] = epoch
        if global_step is not None:
            state['global_step'] = global_step
        if state:
            with open(os.path.join(path, "training_state.json"), "w") as f:
                json.dump(state, f, indent=2)
        # save config metadata
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
        attn_mask = ids.ne(self.tokenizer.pad_token_id).long()

        # 생성 설정
        config = GenerationConfig(do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                               bos_token_id= 151643,
                               eos_token_id=self.tokenizer.eos_token_id,
                               pad_token_id=self.tokenizer.pad_token_id,
                               max_new_tokens=150)

        # 모델로부터 출력 생성
        with torch.no_grad():
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

    def prepare_evaluation(self, path="artifacts/qwen3-4b-lora/checkpoint-final"):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        try:
            # 캐시 디렉토리 설정
            cache_dir = "/2025pdp/.cache"
            
            peft_conf = PeftConfig.from_pretrained(path, cache_dir=cache_dir)
            self.model = PeftModel.from_pretrained(
                self.model, 
                path, 
                is_trainable=False,
                cache_dir=cache_dir
            )
            print(f"Loaded LoRA adapter: {path}")
        except Exception as e:
            print(f"No adapter found: {e}")
        self.model.eval()

if __name__ == "__main__":
    solver = ARCSolver()
