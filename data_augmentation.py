import argparse
import json
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random


def apply_augmentation(example, i):
    if i < 4:
        return {
            "input": np.rot90(example["input"], k=i).tolist(),
            "output": np.rot90(example["output"], k=i).tolist()
        }
    elif i < 6:
        return {
            "input": np.flip(example["input"], axis=(i - 4) % 2).tolist(),
            "output": np.flip(example["output"], axis=(i - 4) % 2).tolist()
        }
    elif i == 6:
        return {
            "input": (9 - np.array(example["input"])).tolist(),
            "output": (9 - np.array(example["output"])).tolist()
        }
    elif i == 7:
        return {
            "input": np.transpose(np.array(example["input"])).tolist(),
            "output": np.transpose(np.array(example["output"])).tolist()
        }
    else: # color permutation
        shuffled = list(range(1, 10))
        random.shuffle(shuffled)
        mapping = [0] + shuffled

        return {
            "input": [[mapping[x] for x in row] for row in example["input"]],
            "output": [[mapping[x] for x in row] for row in example["output"]]
        }

def get_augmentation_name(i):
    if i < 4:
        return f"rot90_{i}"
    elif i < 6:
        return f"flip_{i-4}"
    elif i == 6:
        return "invert"
    elif i == 7:
        return "transpose"
    else:
        return "color_perm"

def main():
    parser = argparse.ArgumentParser(description='Apply data augmentation to JSON files')
    parser.add_argument('--target_dir', type=str, required=True, help='Directory containing input JSON files')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save augmented JSON files')
    args = parser.parse_args()

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Get list of JSON files
    json_files = list(Path(args.target_dir).glob('*.json'))
    print(f"총 {len(json_files)}개의 JSON 파일을 찾았습니다.")

    # Process each JSON file in the target directory
    for json_file in tqdm(json_files, desc="파일 처리 중"):
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Apply each augmentation technique
        for i in tqdm(range(9), desc=f"{json_file.stem} 변환 중", leave=False):
            augmented_data = []
            for example in data:
                augmented_example = apply_augmentation(example, i)
                augmented_data.append(augmented_example)

            # Save augmented data
            aug_name = get_augmentation_name(i)
            output_filename = f"{json_file.stem}_{aug_name}.json"
            output_path = os.path.join(args.save_dir, output_filename)
            
            with open(output_path, 'w') as f:
                json.dump(augmented_data, f, indent=2)

    print(f"\n모든 변환이 완료되었습니다. 결과는 {args.save_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()