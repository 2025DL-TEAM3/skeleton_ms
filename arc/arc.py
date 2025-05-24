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
import random

from .utils import system_prompt, user_message_template1, user_message_template2, user_message_template3
from .dataset import ARCDataset, FileBatchSampler
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from collections import Counter
from .custom_head import apply_custom_head

class ARCSolver:
    """
    You should implement a `Solver` class for the project.
    """

    def __init__(self, stage1_path=None, token=None):
        """
        Args:
            stage1_path (str): path to the stage1 model, stage1 = attention head optimization / stage2 = lm_head optimization
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

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            token=token,
            cache_dir=cache_dir,
        )
        self.tokenizer.bos_token_id = 151643 # Default for Qwen3

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True, # Allow the model to use custom code from the repository
            quantization_config=bnb_config, # Apply the 4-bit quantization configuration
            attn_implementation='sdpa', # Use scaled-dot product attention for better performance
            torch_dtype=torch.float16, # Set the data type for the model
            use_cache=False, # Disable caching to save memory
            device_map=self.device, # Automatically map the model to available devices (e.g., GPUs)
            token=token,
            cache_dir=cache_dir,  # 캐시 디렉토리 지정
        ).to(self.device)  # Move model to device

        ###### add this if you get OOM error
        # self.model.gradient_checkpointing_enable()
        # self.model.config.use_cache = False
        # if hasattr(self.model, "enable_input_require_grads"):
        #     self.model.enable_input_require_grads()
        ######

        # now untie the embed_tok and lm_head
        self.model.tie_word_embeddings = False

        # clone embedding to lm_head, shrink model embeddings
        # in_emb: [V, hidden_dim] -> [K, hidden_dim] , out_emb: [V, hidden_dim] -> [K, hidden_dim]
        self.model, self.tokenizer, self.token_mapping = apply_custom_head(self.model, self.tokenizer)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        print(f"✓ Model vocabulary optimized for ARC: {len(self.token_mapping)} tokens kept")
            
        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)
        ]
        self.sep = self.tokenizer.encode("\n", add_special_tokens=False)[0]

        # color permutation
        self.color_perms = [
            [0,2,3,4,5,6,7,8,9,1],
            [0,4,2,3,5,1,6,8,9,7],
            [0,2,3,1,5,6,4,8,9,7],
            [0,9,1,2,3,4,5,6,7,8],
        ]
        self.color_perms_inv = [np.argsort(perm) for perm in self.color_perms]

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
                if idx == self.tokenizer.eos_token_id:
                    break
                row.append(inv_map.get(idx, 0)) # 없는 값은 0으로 처리
        if row:
            grid.append(row.copy())
        return grid

    # def format_grid(self, grid: List[List[int]]):
    #     """
    #     Format 2D grid into LLM input tokens

    #     Args:
    #         grid (List[List[int]]): 2D grid

    #     Returns:
    #         ids (List[int]): Token list for LLM
    #     """
    #     ids = []

    #     for row in grid:
    #         for col in row:
    #             ids.append(self.pixel_ids[col])
    #         ids.append(self.sep)  # 한 행 끝마다 줄바꿈
    #     return ids

    # def format_prompt(self, datapoint):
    #     """
    #     Format training data and test input into LLM input tokens

    #     Args:
    #         datapoint (dict): contains training data, test input
        
    #     Returns:
    #         prompt (dict): dictionary that contains input ids and additional informations
    #     """

    #     # 1) system prompt
    #     token_ids = []
    #     system_block = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    #     token_ids += self.tokenizer.encode(system_block, add_special_tokens=False)

    #     # 2) user prompt 1
    #     user_block_1 = f"<|im_start|>user\n{user_message_template1.format(n=len(datapoint['train']), plural=('s' if len(datapoint['train'])!=1 else ''))}\n"
    #     token_ids += self.tokenizer.encode(user_block_1, add_special_tokens=False)

    #     # 3) examples
    #     for i, ex in enumerate(datapoint['train'], start=1):
    #         token_ids += self.tokenizer.encode(f"Example {i} Input:\n", add_special_tokens=False)
    #         token_ids += self.format_grid(ex['input'])
    #         token_ids += self.tokenizer.encode(f"Example {i} Output:\n", add_special_tokens=False)
    #         token_ids += self.format_grid(ex['output'])

    #     # 4) user prompt 2
    #     user_block_2 = f"\n{user_message_template2}\n"
    #     token_ids += self.tokenizer.encode(user_block_2, add_special_tokens=False)

    #     # 5) test input
    #     token_ids += self.tokenizer.encode("Test Input:\n", add_special_tokens=False)
    #     token_ids += self.format_grid(datapoint['test'][0]['input'])

    #     # 6) user prompt 3
    #     user_block_3 = f"\n{user_message_template3}<|im_end|>\n"
    #     token_ids += self.tokenizer.encode(user_block_3, add_special_tokens=False)

    #     # 7) assistant response
    #     token_ids += self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    #     # 8) 최종 프롬프트 생성
    #     all_ids = token_ids
    #     tokens = torch.tensor(all_ids, dtype=torch.long, device=self.device)

    #     return {
    #         "input_ids": tokens,
    #         "input": datapoint['test'][0]['input'],
    #         "train": datapoint['train']
    #     }

    def grid_to_str(self, grid: List[List[int]]):
        # 줄마다 012… 형식, 마지막 줄 포함 모든 줄 끝에 \n
        return "".join(
            f"{''.join(str(c) for c in row)}\n"
            for row in grid
        )

    def format_prompt(self, datapoint):
        # Build example block
        n = len(datapoint['train'])
        plural = 's' if n != 1 else ''
        examples_block = ''
        for i, ex in enumerate(datapoint['train'], start=1):
            examples_block += f"Example {i} Input:\n"
            examples_block += self.grid_to_str(ex['input'])
            examples_block += f"Example {i} Output:\n"
            examples_block += self.grid_to_str(ex['output'])
        template1 = user_message_template1.format(n=n, plural=plural) + "\n" + examples_block + "Observe how each input becomes its output."

        # Build test input block
        test_input = f"Test Input:\n{self.grid_to_str(datapoint['test'][0]['input'])}"
        template2 = user_message_template2 + "\n" + test_input

        # Assemble messages for chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": template1 + "\n" + template2 + "\n" + user_message_template3}
        ]

        # 3) Apply chat template without tokenizing
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False
        )

        # 4) Manually tokenize the resulting prompt text
        inputs = self.tokenizer(text, return_tensors="pt")
        # Extract the first sequence in the batch
        input_ids = inputs["input_ids"][0]

        return {
            'input_ids': input_ids,
            'input': datapoint['test'][0]['input'],
            'train': datapoint['train'],
        }

    def dynamic_collate(self, batch):
        """
        Custom collate function to handle variable-length sequences
        """
        input_ids = [item['input_ids'] for item in batch]
        target_ids = [item['target_ids'] for item in batch]

        padded_input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side="left"
        )
        padded_target_ids = pad_sequence(
            target_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side="left"
        )
        
        return {
            "input_ids": padded_input_ids,
            "target_ids": padded_target_ids,
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
               patience=10, val_steps=1000, fixed_seed=42, max_val_files=50, val_steps_per_file=2, val_batch_size=4):
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
            fixed_seed (int): Random seed for validation sampling
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
        peft_attn = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=16,                     # LoRA rank - determines the size of the update matrices
            lora_alpha=32,           # LoRA scaling factor - controls the magnitude of updates
            lora_dropout=0.1,        # Dropout probability for LoRA layers
            target_modules=["q_proj","k_proj","v_proj","o_proj", "lm_head"],
        )
        # wrap the base model as a Peft model
        self.model = get_peft_model(self.model, peft_attn, adapter_name="attn")
        self.model.print_trainable_parameters() # Display the number of trainable parameters
        
        # Initialize dataset and data loader
        dataset = ARCDataset(train_dataset, self.tokenizer, self, steps_per_file=steps_per_file, is_validation=False)
        
        # Initialize optimizer with specified learning rate
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

        # 1) 한 에폭당 optimizer step 수
        steps_per_epoch = (len(dataset) // batch_size) // steps_accum

        # 2) 전체 optimizer step 수
        total_optimizer_steps = num_epochs * steps_per_epoch

        # 3) 워밍업 스텝 수
        warmup_steps = int(total_optimizer_steps * warmup_rate)
        log_message(f"Warmup steps: {warmup_steps}, Total optimizer steps: {total_optimizer_steps}")

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )

        start_epoch, global_step = 0, 0
        # Early stopping variables
        best_val_accuracy = 0
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = os.path.join(save_dir, "checkpoint-best")

        if resume_from:
            start_epoch, global_step, best_val_accuracy, best_val_loss = self.load_checkpoint(resume_from, optimizer, scheduler)
            log_message(f"Resuming from {resume_from}: epoch {start_epoch} and global step {global_step}")
        
        # best checkpoint가 있으면 값 불러오기
        if os.path.exists(os.path.join(save_dir, "checkpoint-best")):
            log_message(f"Found best checkpoint in {save_dir}, loading best validation metrics")
            _, _, best_checkpoint_accuracy, best_checkpoint_loss = self.load_checkpoint(
                os.path.join(save_dir, "checkpoint-best"), None, None
            )
            # 저장된 값이 현재 값보다 좋으면 업데이트
            if best_checkpoint_accuracy > best_val_accuracy:
                best_val_accuracy = best_checkpoint_accuracy
                log_message(f"Loaded best validation accuracy: {best_val_accuracy:.4f}")
            if best_checkpoint_loss < best_val_loss:
                best_val_loss = best_checkpoint_loss
                log_message(f"Loaded best validation loss: {best_val_loss:.4f}")
        
        # 한 에폭당 배치 수 계산
        per_file = (steps_per_file + batch_size - 1) // batch_size
        total_batches_per_epoch = len(dataset.examples) * per_file
        
        # resume 시 skip할 배치 수 계산 (에폭 수 반영)
        skip_batches = 0
        if resume_from and global_step > 0:
            skip_batches = global_step % total_batches_per_epoch
            log_message(f"Resuming from batch {skip_batches} (epoch {start_epoch}, step {global_step})")
        
        # batch sampler 생성
        batch_sampler = FileBatchSampler(
            dataset, 
            batch_size,
            seed=fixed_seed + start_epoch if resume_from else None,  # resume 시에만 seed 설정
            skip_batches=skip_batches
        )
        loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=self.dynamic_collate)

        # Initialize validation dataset and data loader if provided
        val_loader = None
        if validation_dataset is not None:
            val_dataset = ARCDataset(validation_dataset, self.tokenizer, self, 
                                    steps_per_file=val_steps_per_file,  # 검증에는 더 적은 스텝 사용
                                    is_validation=True,  # 결정적 샘플링 활성화
                                    seed=fixed_seed,       # 고정된 시드 사용
                                    max_val_files=max_val_files)  # 검증에 사용할 최대 파일 수
            val_loader = DataLoader(val_dataset, val_batch_size, shuffle=False, collate_fn=self.dynamic_collate)
            log_message(f"Validation dataset loaded with {len(val_dataset)} steps, using seed {fixed_seed}, max files {max_val_files}, steps/file {val_steps_per_file}, batch size {val_batch_size}")
            log_message(f"Total validation batches: {len(val_loader)}")

        # CSV 형식의 메트릭 로그 파일 생성
        metrics_log_file = os.path.join(save_dir, "metrics_log.txt")
        with open(metrics_log_file, 'w', encoding='utf-8') as f:
            f.write("global_step,epoch,train_loss,val_loss,val_accuracy,learning_rate\n")

        # Set model to training mode
        self.model.train()
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            # 새로운 에폭 시작 시 skip_batches 리셋
            if epoch > start_epoch:
                batch_sampler.reset_skip_batches()
                # 새로운 에폭의 시드 설정
                random.seed(fixed_seed + epoch)
                batch_sampler.seed = fixed_seed + epoch
                log_message(f"Starting epoch {epoch} with seed {fixed_seed + epoch}")
            
            total_loss = 0
            # steps = len(dataset) / batch_size
            for step, batch in enumerate(loader):
                global_step += 1
                
                # Move batch to device and compute loss
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                loss = self.seq2seq_loss(input_ids, target_ids) / steps_accum
                # loss = self.seq2seq_loss_with_regularization(input_ids, target_ids, epoch=epoch) / steps_accum
                
                # Backpropagation
                loss.backward()
                
                # Gradient accumulation and optimization
                if global_step % steps_accum == 0:
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()  # Update model parameters
                    scheduler.step()  # Update learning rate
                    optimizer.zero_grad(set_to_none=True)  # Clear gradients
                
                # Track and display training progress
                total_loss += loss.item()
                current_lr = scheduler.get_last_lr()[0]
                
                if step % 100 == 0:
                    log_message(f"[Epoch {epoch+1}] step {step} loss {loss.item():.4f} lr {current_lr:.6f}")

                # Validation check - 첫 번째 에폭에서는 검증 건너뛰기
                if val_loader is not None and global_step % val_steps == 0 and epoch >= 0:
                    val_loss, val_accuracy = self.validate(val_loader)
                    # torch.cuda.empty_cache() # 메모리 정리
                    log_message(f"[Validation] global_step {global_step} loss {val_loss:.4f} accuracy {val_accuracy:.4f}")
                    
                    # 메트릭 로깅
                    with open(metrics_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"{global_step},{epoch+1},{loss.item():.6f},{val_loss:.6f},{val_accuracy:.6f},{current_lr:.8f}\n")
                    
                    # 손실과 정확도를 모두 고려한 early stopping
                    improved = False
                    improvement_message = []
                    
                    # 정확도 향상 확인
                    if val_accuracy > best_val_accuracy:
                        improvement_message.append(f"Validation accuracy improved from {best_val_accuracy:.4f} to {val_accuracy:.4f}")
                        best_val_accuracy = val_accuracy
                        improved = True
                    
                    # 손실 향상 확인 (정확도가 동일할 때도 손실이 감소했으면 개선으로 간주)
                    if val_loss < best_val_loss:
                        improvement_message.append(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
                        best_val_loss = val_loss
                        improved = True
                    
                    if improved:
                        log_message(", ".join(improvement_message))
                        # Save best model
                        self.save_model(best_model_path, optimizer, scheduler, epoch, global_step, val_accuracy, val_loss)
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        log_message(f"Validation metrics did not improve. Patience: {patience_counter}/{patience}")
                    
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
                    self.save_model(os.path.join(save_dir, f"checkpoint-{global_step}"), optimizer, scheduler, epoch, global_step, best_val_accuracy, best_val_loss)

                # 배치 단위 메모리 정리
                del input_ids, target_ids, loss
            # Print average loss for the epoch
            avg_epoch_loss = total_loss/len(loader)
            log_message(f"Epoch {epoch+1} avg loss {avg_epoch_loss:.4f}")

            # 에폭 완료 후 캐시 정리
            # torch.cuda.empty_cache()
            
            # 에폭 완료 후 메트릭 로깅 - train loss만 기록
            with open(metrics_log_file, 'a', encoding='utf-8') as f:
                f.write(f"{global_step},{epoch+1},{avg_epoch_loss:.6f},,,{current_lr:.8f}\n")
        
        self.model.eval()  # Set model to evaluation mode after training
        log_message("===== Training completed =====")

        # save final model
        self.save_model(os.path.join(save_dir, "checkpoint-final"), optimizer, scheduler, epoch, global_step, best_val_accuracy, best_val_loss)

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
        correct_predictions = 0.0
        
        # 검증 시에는 Greedy decoding으로 일관된 출력 생성
        val_config = GenerationConfig(
            do_sample=False,  # 샘플링 없이 확정적 생성
            use_cache=False,  # 캐시 사용 비활성화
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=150
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # 진행 상황 출력
                if batch_idx == (len(val_loader) // 2) or batch_idx == len(val_loader) - 1:
                    print(f"Validation progress: {batch_idx+1}/{len(val_loader)} batches")
                
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                # Compute validation loss
                loss = self.seq2seq_loss(input_ids, target_ids)
                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)

                # get attention mask (should ignore padding tokens when generating)
                attn_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
                
                # Generate outputs for accuracy calculation
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    generation_config=val_config,
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
                    
                    try:
                        # numpy 배열로 변환하여 shape 비교 및 내용 확인을 간소화
                        pred_np = np.array(pred_grid)
                        target_np = np.array(target_grid)
                        
                        # 점수 계산 방식:
                        # 1. shape 일치 시 기본적으로 0.5점
                        # 2. 내용까지 완벽히 일치하면 1.0점
                        # 3. shape 불일치 시 0점
                        
                        # 두 배열의 shape 비교
                        if pred_np.shape == target_np.shape:
                            # Shape이 일치하면 기본 0.5점
                            score = 0.5
                            
                            # 내용까지 모두 일치하면 1.0점으로 업그레이드
                            if np.array_equal(pred_np, target_np):
                                score = 1.0
                                
                            correct_predictions += score
                        else:
                            # Shape 불일치
                            correct_predictions += 0.0
                    except (ValueError, TypeError) as e:
                        # 배열 변환 실패 시 0점 처리
                        correct_predictions += 0.0

                # 배치 단위 메모리 정리
                del input_ids, target_ids, outputs, attn_mask, loss
        
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
        val_accuracy, val_loss = 0, float('inf')
        if os.path.isfile(state_path):
            st = json.load(open(state_path))
            start_epoch = st.get('epoch', 0)
            start_step  = st.get('global_step', 0)
            val_accuracy = st.get('val_accuracy', 0)
            val_loss = st.get('val_loss', float('inf'))
        return start_epoch, start_step, val_accuracy, val_loss

    def save_model(self, path=None, optimizer=None, scheduler=None, epoch=None, global_step=None, val_accuracy=None, val_loss=None):
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
        if val_accuracy is not None:
            state['val_accuracy'] = val_accuracy
        if val_loss is not None:
            state['val_loss'] = val_loss
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

    # def predict(self, examples, questions_input):
    #     """
    #     A single example of test data is given.
    #     You should predict 2D grid (List[List[int]] or np.ndarray)

    #     Args:
    #         examples (List[dict]): List of training examples,
    #             each list element is a dictionary that contains "input" and "output"
    #             for example,
    #             [
    #                 {
    #                     "input": [[1,2],[3,4]],
    #                     "output": [[4,5],[6,7]],
    #         questions_input (List[List[int]]): A 2d grid,
    #             which is a input for a given question
    #     Returns:
    #         output (List[List[int]]): A 2d grid,
    #             which is the output of given input question.
    #     """
    #     # 프롬프트 데이터 구성
    #     datapoint = {"train": examples, "test": [{"input": questions_input}]}

    #     # 프롬프트 생성 및 토큰화
    #     prompt = self.format_prompt(datapoint)
    #     ids = prompt['input_ids'].unsqueeze(0).to(self.device) # (1, seq_len)
    #     attn_mask = ids.ne(self.tokenizer.pad_token_id).long()

    #     # 생성 설정 - 실제 예측에서는 샘플링 사용
    #     config = GenerationConfig(
    #         do_sample=True, 
    #         temperature=0.7, 
    #         top_p=0.8,
    #         top_k=20,
    #         eos_token_id=self.tokenizer.eos_token_id,
    #         pad_token_id=self.tokenizer.pad_token_id,
    #         max_new_tokens=150
    #     )

    #     # 모델로부터 출력 생성
    #     with torch.no_grad():
    #         out = self.model.generate(input_ids=ids, attention_mask=attn_mask, generation_config=config).squeeze().cpu()
    #     N_prompt = ids.size(1)

    #     # 프롬프트 이후의 토큰만 추출
    #     output = out[N_prompt:].tolist()
    #     train_input = np.array(prompt['train'][0]['input'])
    #     train_output = np.array(prompt['train'][0]['output'])
    #     test_input = np.array(prompt['input'])

    #     # LLM-generated grid may have wrong shape
    #     # So adjust shape by input-output pairs
    #     if train_input.shape == train_output.shape:
    #         x, y = test_input.shape
    #     else:
    #         x = (train_output.shape[0] * test_input.shape[0]) // train_input.shape[0]
    #         y = (train_output.shape[1] * test_input.shape[1]) // train_input.shape[1]

    #     try:
    #         grid = np.array(self.parse_grid(output))
    #     except Exception as e:
    #         # 파싱 실패 시 랜덤 그리드 반환
    #         grid = np.random.randint(0, 10, (x, y))

    #     return grid

    def apply_augmentation(self, examples, questions_input, i):
        if i < 2: # original, 90 degree rotated
            augmented_examples = []
            for example in examples:
                augmented_examples.append({
                    "input": np.rot90(example["input"], k=i).tolist(),
                    "output": np.rot90(example["output"], k=i).tolist()
                })
            augmented_questions_input = np.rot90(questions_input, k=i).tolist()
        elif i == 2: # flip horizontally
            augmented_examples = []
            for example in examples:
                augmented_examples.append({
                    "input": np.flip(example["input"], axis=1).tolist(),
                    "output": np.flip(example["output"], axis=1).tolist()
                })
            augmented_questions_input = np.flip(questions_input, axis=1).tolist()
        elif i == 3: # transpose
            augmented_examples = []
            for example in examples:
                augmented_examples.append({
                    "input": np.transpose(np.array(example["input"])).tolist(),
                    "output": np.transpose(np.array(example["output"])).tolist()
                })
            augmented_questions_input = np.transpose(np.array(questions_input)).tolist()
        elif i == 4: # invert colors
            augmented_examples = []
            for example in examples:
                augmented_examples.append({
                    "input": (9 - np.array(example["input"])).tolist(),
                    "output": (9 - np.array(example["output"])).tolist()
                })
            augmented_questions_input = (9 - np.array(questions_input)).tolist()            
        elif i < 9: # color permutation
            perm = np.array(self.color_perms[i-5]) 
            augmented_examples = []
            for example in examples:
                augmented_examples.append({
                    "input": perm[np.array(example["input"])].tolist(),
                    "output": perm[np.array(example["output"])].tolist()
                })
            augmented_questions_input = perm[np.array(questions_input)].tolist()
        else:
            raise ValueError(f"Invalid augmentation index: {i}")

        return augmented_examples, augmented_questions_input

    def unapply_augmentation(self, logits_grid: torch.Tensor, i):
        if i < 2: # original, 90 degree rotated
            k = (4 - i) % 4
            geom_restored = torch.rot90(logits_grid, k=k, dims=(0, 1))
        elif i == 2: # flip horizontally
            geom_restored = torch.flip(logits_grid, dims=(1,))
        elif i == 3: # transpose
            geom_restored = logits_grid.transpose(0, 1)
        elif i == 4: # invert colors
            V = logits_grid.size(-1)
            perm = torch.arange(V, device=logits_grid.device)
            for c in range(10):
                orig_id = self.pixel_ids[c]
                aug_id = self.pixel_ids[9 - c]
                perm[orig_id] = aug_id
            geom_restored = logits_grid.index_select(-1, perm)
        elif i < 9: # color permutation
            inv = self.color_perms_inv[i-5]             # numpy array len-10
            V   = logits_grid.size(-1)
            perm = torch.arange(V, device=logits_grid.device)
            for c in range(10):
                orig = self.pixel_ids[c]
                invc = self.pixel_ids[inv[c]]
                perm[orig] = invc
            geom_restored = logits_grid.index_select(-1, perm) 
        else:
            raise ValueError(f"Invalid augmentation index: {i}")
        return geom_restored

    def build_augmented_prompts(self, examples, questions_input, augment_num):
        prompts_ids = []

        for i in range(augment_num):
            augmented_examples, augmented_questions_input = self.apply_augmentation(examples, questions_input, i)
            prompt = self.format_prompt({"train": augmented_examples, "test": [{"input": augmented_questions_input}]})
            prompts_ids.append(prompt['input_ids'])
        
        tensors = [ids.detach().clone() for ids in prompts_ids]
        padded = pad_sequence(tensors, batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side="left")
        attn_mask = padded.ne(self.tokenizer.pad_token_id).long()

        return padded.to(self.device), attn_mask.to(self.device)

    def predict(self, examples, questions_input):
        # 프롬프트 데이터 구성
        augment_num = 8
        padded_ids, attn_mask = self.build_augmented_prompts(examples, questions_input, augment_num) # (B, seq_len)
        batch_size = padded_ids.size(0) # B
        N_prompt = padded_ids.size(1) # seq_len

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=padded_ids,
                attention_mask=attn_mask,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=150,
                do_sample=True, 
                temperature=0.7, 
                top_p=0.8,
                top_k=20,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # (B, gen_len, vocab_size)
        scores = torch.stack(outputs.scores, dim=1)

        batch_logits_grids = []
        for i in range(batch_size):
            seq = outputs.sequences[i] # (seq_len + gen_len)
            gen_ids = seq[N_prompt:].tolist()

            grid_rows = []
            row_logits = []
            for step_idx, tok in enumerate(gen_ids):
                if tok == self.sep:
                    if row_logits:
                        grid_rows.append(torch.stack(row_logits, dim=0))
                        row_logits = []
                elif tok == self.tokenizer.eos_token_id or tok == self.tokenizer.pad_token_id:
                    break
                else:
                    # get logits and apply softmax
                    prob = torch.softmax(scores[i, step_idx, :], dim=-1)
                    row_logits.append(prob)
            if row_logits:
                grid_rows.append(torch.stack(row_logits, dim=0))

            try:
                grid_tensor = torch.stack(grid_rows, dim=0) # (H, W, V)
                unapplied_grid_tensor = self.unapply_augmentation(grid_tensor, i) # (H', W', V)
                batch_logits_grids.append(unapplied_grid_tensor)
            except Exception as e:
                continue
        
        shape_3d = [g for g in batch_logits_grids if g.dim() == 3]
        shape_counts = Counter(tuple(g.shape) for g in shape_3d)
        most_frequent_shape = torch.Size(max(shape_counts, key=shape_counts.get))
        valid_shape_3d = [g for g in shape_3d if tuple(g.shape) == tuple(most_frequent_shape)]

        # (N, H, W, V)
        stacked = torch.stack(valid_shape_3d, dim=0)

        # (H, W, V)
        sum_logits = torch.sum(stacked, dim=0)
        # weights = stacked.max(dim=-1).values
        # sum_logits = torch.sum(stacked * weights.unsqueeze(-1), dim=0)

        # (H, W)
        mask = torch.full_like(sum_logits, -1e9) # filter out other than 0-9
        mask[:, :, self.pixel_ids] = 0
        sum_logits = sum_logits + mask
        top_token_ids = torch.argmax(sum_logits, dim=-1)

        # (H, W)
        inv_map = {k: i for i, k in enumerate(self.pixel_ids)}
        values = [
            [inv_map[tok] for tok in row]
            for row in top_token_ids.tolist()
        ]
        most_likely_values = torch.tensor(values, dtype=torch.int64)

        return most_likely_values.cpu().numpy()
                
    def prepare_evaluation(self, path="artifacts/qwen3-4b-lora/checkpoint-final", use_custom_head=True):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        try:
            # 캐시 디렉토리 설정
            cache_dir = "/2025pdp/.cache"

            # custom head 사용 시 tokenizer 설정
            if use_custom_head:
                # now untie the embed_tok and lm_head
                self.model.tie_word_embeddings = False

                # clone embedding to lm_head, shrink model embeddings
                # in_emb: [V, hidden_dim] -> [K, hidden_dim] , out_emb: [V, hidden_dim] -> [K, hidden_dim]
                self.model, self.tokenizer, self.token_mapping = apply_custom_head(self.model, self.tokenizer)
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
                self.model.config.eos_token_id = self.tokenizer.eos_token_id
                print(f"✓ Model vocabulary optimized for ARC: {len(self.token_mapping)} tokens kept")

                self.pixel_ids = [
                    self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)
                ]
                self.sep = self.tokenizer.encode("\n", add_special_tokens=False)[0]
            
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
