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
    def __init__(self, dataset, tokenizer, solver, steps_per_file=50):
        """
        Initialize the ARCDataset.

        Args:
            dataset: HuggingFace dataset containing training examples
            tokenizer: Tokenizer for encoding and decoding text
            solver: ARCSolver instance for formatting prompts
            steps_per_file: Maximum number of steps per file
        """
        self.tokenizer = tokenizer
        self.solver = solver
        self.steps_per_file = steps_per_file

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

        self.total_steps = len(self.examples) * self.steps_per_file
        print(f"Loaded total {len(self.examples)} examples")
        print(f"Total steps per epoch: {self.total_steps}")

    def __len__(self):
        # Number of iterations per epoch
        return self.total_steps

    def __getitem__(self, idx):
        """
        Return one training sample per call:

        1. Select one JSON file
        2. Sample 4 examples: 3 for train, 1 for test
        3. Format prompt and target tensors
        """
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
        tgt_ids = torch.tensor(tgt_tokens, dtype=torch.long)

        return {"input_ids": inp_ids, "target_ids": tgt_ids}
