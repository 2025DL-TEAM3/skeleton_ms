import os
import json
import glob
import random
import torch
from torch.utils.data import Dataset

class ARCDataset(Dataset):
    """
    Custom dataset class for the ARC dataset.
    """
    def __init__(self, dataset, tokenizer, solver, steps_per_file=50, is_validation=False, seed=42, max_val_files=50):
        """
        Initialize the ARCDataset.

        Args:
            dataset: HuggingFace dataset containing training examples
            tokenizer: Tokenizer for encoding and decoding text
            solver: ARCSolver instance for formatting prompts
            steps_per_file: Maximum number of steps per file
            is_validation: If True, sampling will be deterministic for validation
            seed: Random seed for validation sampling
            max_val_files: Maximum number of files to use for validation
        """
        self.tokenizer = tokenizer
        self.solver = solver
        self.steps_per_file = steps_per_file
        self.is_validation = is_validation
        self.seed = seed
        self.max_val_files = max_val_files

        # load JSON files
        if hasattr(dataset, 'data_files') and dataset.data_files:
            files = dataset.data_files
        else:
            raise ValueError("Dataset must have a 'data_files' attribute")

        print(f"Loading {len(files)} files")

        # load examples
        self.examples = []
        for p in files:
            with open(p) as f:
                file_examples = json.load(f)
                if isinstance(file_examples, list) and len(file_examples) > 0:
                    self.examples.append({
                        'file_path': p,
                        'examples': file_examples
                    })
                else:
                    print(f"Skipping {p} because it is not a list or empty")
        if not self.examples:
            raise ValueError("No examples loaded")

        # 검증용 데이터셋인 경우 미리 샘플을 준비
        self.validation_samples = None
        if is_validation:
            self._prepare_validation_samples()

        self.total_steps = len(self.examples) * self.steps_per_file
        print(f"Loaded total {len(self.examples)} examples")
        print(f"Total steps per epoch: {self.total_steps}")

    def _prepare_validation_samples(self):
        """
        검증 데이터셋인 경우 미리 샘플을 결정적으로 준비합니다.
        """
        self.validation_samples = []
        
        # 파일이 max_val_files보다 많으면 랜덤하게 선택
        file_indices = list(range(len(self.examples)))
        if len(file_indices) > self.max_val_files:
            file_indices = random.sample(file_indices, self.max_val_files)
            print(f"Randomly selected {self.max_val_files} files for validation from {len(self.examples)} total files")
        
        for file_idx in file_indices:
            file_data = self.examples[file_idx]
            examples = file_data['examples']
            
            for _ in range(self.steps_per_file):
                if len(examples) < 4:
                    # 예제가 부족한 경우 다른 파일에서 가져옴
                    chosen_file_idx = random.choice(file_indices)
                    chosen_examples = self.examples[chosen_file_idx]['examples']
                else:
                    chosen_examples = examples
                
                # 4개 예제 선택
                selected_examples = random.sample(chosen_examples, 4)
                
                self.validation_samples.append({
                    'file_idx': file_idx,
                    'selected_examples': selected_examples
                })
        
        print(f"Prepared {len(self.validation_samples)} fixed validation samples from {len(file_indices)} files")

    def __len__(self):
        # Number of iterations per epoch
        if self.is_validation:
            return len(self.validation_samples)
        else:
            return self.total_steps

    def __getitem__(self, idx):
        """
        Return one training sample per call:

        1. Select one JSON file
        2. Sample 4 examples: 3 for train, 1 for test
        3. Format prompt and target tensors
        """
        if self.is_validation and self.validation_samples:
            # 검증용 데이터셋인 경우 미리 준비된 샘플 사용
            sample = self.validation_samples[idx % len(self.validation_samples)]
            selected_examples = sample['selected_examples']
        else:
            # 1) choose file, not random = suitable for ARC-AGI dataset (idx would be shuffled due to shuffle=True in DataLoader)
            file_idx = (idx // self.steps_per_file) % len(self.examples)
            file_data = self.examples[file_idx]
            examples = file_data['examples']

            # choose another file if there are less than 4 examples
            # this would not happen (100-1000 examples per file)
            if len(examples) < 4:
                chosen_file = random.choice(self.examples)
                examples = chosen_file['examples']
            
            # 2) sample 4 examples
            selected_examples = random.sample(examples, 4)
        
        # 3) format prompt and target tensors
        train_exs = selected_examples[:3]
        test_ex = selected_examples[3]

        # Build datapoint
        datapoint = {"train": train_exs, "test": [{"input": test_ex["input"]}]}
        prompt = self.solver.format_prompt(datapoint)

        if isinstance(prompt['input_ids'], torch.Tensor):
            inp_ids = prompt['input_ids'].clone().detach()
        else:
            inp_ids = torch.tensor(prompt['input_ids'], dtype=torch.long)

        # target_ids tensor (answer)
        tgt_tokens = self.solver.format_grid(test_ex["output"])
        if tgt_tokens[-1] != self.tokenizer.eos_token_id: # <|im_end|>
            tgt_tokens.append(self.tokenizer.eos_token_id)
        tgt_ids = torch.tensor(tgt_tokens, dtype=torch.long)

        return {"input_ids": inp_ids, "target_ids": tgt_ids}

class FileBatchSampler(torch.utils.data.Sampler):
    """
    매 배치가 동일한 JSON 파일에서 온 samples만 포함하도록 하는 BatchSampler.
    """
    def __init__(self, dataset, batch_size, seed=None, skip_batches=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.steps_per_file = dataset.steps_per_file
        self.num_files = len(dataset.examples)
        self.seed = seed
        self.skip_batches = skip_batches

    def reset_skip_batches(self):
        """새로운 에폭 시작 시 skip_batches를 리셋"""
        self.skip_batches = 0

    def __iter__(self):
        # 시드가 있으면 초기화
        if self.seed is not None:
            random.seed(self.seed)

        # 파일 단위로 모든 배치 인덱스 리스트 생성
        batch_list = []
        file_indices = list(range(self.num_files))
        random.shuffle(file_indices)
        for f in file_indices:
            start = f * self.steps_per_file
            end   = start + self.steps_per_file
            idxs  = list(range(start, end))
            random.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                batch_list.append(idxs[i : i + self.batch_size])

        # 이미 본 배치 건너뛰기
        for batch in batch_list[self.skip_batches:]:
            yield batch

    def __len__(self):
        per_file = (self.steps_per_file + self.batch_size - 1) // self.batch_size
        return self.num_files * per_file - self.skip_batches
