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

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

# system prompt
system_prompt = (
    "You are an expert at solving puzzles from the Abstraction and Reasoning Corpus (ARC). "
    "From three input/output examples, infer the transformation rule "
    "and apply it to a new test grid."
)

# user prompt 1: examples
user_message_template1 = (
    "Here are {n} example input and output pair{plural} from which you should learn the underlying rule to later predict the output for the given test input:\n"
)

# user prompt 2: test input
user_message_template2 = (
    "\nNow, solve the following puzzle based on its input grid by applying the rules you have learned from the training data:\n"
)

# user prompt 3: output format
user_message_template3 = (
    "What is the output grid? Please provide only the grid where each row is a sequence of digits, where each row ends on a new line, and no extra text or spaces:\n"
)

model_id = "Qwen/Qwen3-0.6B"

def format_grid(grid):
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
            ids.append(tokenizer.encode(str(col), add_special_tokens=False)[0])
        ids.append(tokenizer.encode("\n", add_special_tokens=False)[0])
    return ids

def grid_to_str(grid):
    return "\n".join([' '.join([str(col) for col in row]) for row in grid])

# Configure the BitsAndBytes settings for 4-bit quantization to reduce memory usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_use_double_quant=True,  # Use double quantization for improved precision
    bnb_4bit_quant_type="nf4",  # Specify the quantization type
    bnb_4bit_compute_dtype=torch.float16,  # Set the computation data type
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True, # Allow the model to use custom code from the repository
    quantization_config=bnb_config, # Apply the 4-bit quantization configuration
    attn_implementation='sdpa', # Use scaled-dot product attention for better performance
    torch_dtype=torch.float16, # Set the data type for the model
    use_cache=False, # Disable caching to save memory
    device_map="auto", # Automatically map the model to available devices (e.g., GPUs)
    cache_dir="/2025pdp/.cache"  # 캐시 디렉토리 지정
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    cache_dir="/2025pdp/.cache",  # 토크나이저도 동일한 캐시 디렉토리 사용
)

datapoint = {
    'train': [
        {
            'input': [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ],
            'output': [
                [9, 9, 9],
                [9, 9, 9],
                [9, 9, 9]
            ]
        },
        {
            'input': [
                [8, 8, 8],
                [8, 8, 8],
                [8, 8, 8]
            ],
            'output': [
                [5, 5, 5],
                [5, 5, 5],
                [5, 5, 5]
            ]
        }
    ],
    'test': [
        {
            'input': [
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3]
            ]
        },
        {
            'output': [
                [4, 4, 4],
                [4, 4, 4],
                [4, 4, 4]
            ]
        }
    ]
}

# 1) 시스템 + 유저 헤더
prefix_ids = tokenizer.apply_chat_template(
    [
        {"role":"system", "content": system_prompt},
        {"role":"user",   "content": user_message_template1.format(
                            n=len(datapoint['train']),
                            plural=('s' if len(datapoint['train'])!=1 else '')
                        )}
    ],
    tokenize=True,
    add_special_tokens=False,
    add_generation_prompt=False
)

# 각 prefix_ids 내 id를 (id: "문자 token"), ... 형식으로 출력
print("Prefix IDs:")
for i, id in enumerate(prefix_ids):
    print(f"({id}: {tokenizer.decode(id)})", end=" ")


# 2) 그리드 토큰 직접 삽입
grid_ids = []
grid_ids += tokenizer.encode(
    "<|im_start|>user\n", add_special_tokens=False
)
for i, ex in enumerate(datapoint['train'], start=1):
    grid_ids += tokenizer.encode(f"Example {i} Input:\n", add_special_tokens=False)
    grid_ids += tokenizer.encode(grid_to_str(ex['input']), add_special_tokens=False)
    grid_ids += tokenizer.encode(f"Example {i} Output:\n", add_special_tokens=False)
    grid_ids += tokenizer.encode(grid_to_str(ex['output']), add_special_tokens=False)
grid_ids += tokenizer.encode(user_message_template2, add_special_tokens=False)
grid_ids += tokenizer.encode("Test Input:\n", add_special_tokens=False)
grid_ids += tokenizer.encode(grid_to_str(datapoint['test'][0]['input']), add_special_tokens=False)
grid_ids += tokenizer.encode(
    "<|im_end|>\n", add_special_tokens=False
)
print("\nGrid IDs:")
for i, id in enumerate(grid_ids):
    print(f"({id}: {tokenizer.decode(id)})", end=" ")

# 3) 어시스턴트 생성 프롬프트
suffix_ids = tokenizer.apply_chat_template(
    [
        {"role":"user", "content": user_message_template3}
    ],
    tokenize=True,
    add_special_tokens=False,
    add_generation_prompt=True
)
print("\nSuffix IDs:")
for i, id in enumerate(suffix_ids):
    print(f"({id}: {tokenizer.decode(id)})", end=" ")
