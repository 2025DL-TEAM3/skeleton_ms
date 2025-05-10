import argparse
import glob
import json
import random
import os
from pathlib import Path
import numpy as np

def render_grid(grid):
    """
    그리드를 시각적으로 출력합니다.
    """
    color_map = {
        0: '\033[97m⬜\033[0m',  # 하얀색
        1: '\033[91m🟥\033[0m',  # 빨간색
        2: '\033[92m🟩\033[0m',  # 초록색
        3: '\033[94m🟦\033[0m',  # 파란색
        4: '\033[93m🟨\033[0m',  # 노란색
        5: '\033[95m🟪\033[0m',  # 보라색
        6: '\033[96m🟦\033[0m',  # 청록색
        7: '\033[90m⬛\033[0m',  # 검은색
        8: '\033[37m⬜\033[0m',  # 회색
        9: '\033[33m🟧\033[0m',  # 주황색
    }
    
    for row in grid:
        print(''.join([color_map.get(cell, f'{cell:2}') for cell in row]))

def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='ARC 데이터셋 샘플 시각화')
    parser.add_argument('--dataset', type=str, required=True, help='데이터셋 디렉토리 또는 JSON 파일 경로')
    parser.add_argument('--samples', type=int, default=10, help='시각화할 샘플 수')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    args = parser.parse_args()
    
    # 랜덤 시드 설정
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 입력 경로가 디렉토리인지 파일인지 확인
    dataset_path = Path(args.dataset)
    all_examples = []
    
    if dataset_path.is_dir():
        # 디렉토리 내의 모든 JSON 파일을 찾음
        json_files = glob.glob(os.path.join(args.dataset, "*.json"))
        if not json_files:
            print(f"경고: {args.dataset}에서 JSON 파일을 찾을 수 없습니다.")
            return
        
        print(f"총 {len(json_files)}개의 JSON 파일을 찾았습니다.")
        
        # 각 파일에서 예제를 로드하고 합침
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    examples = json.load(f)
                
                if isinstance(examples, list):
                    all_examples.extend(examples)
                    print(f"{Path(json_file).name}에서 {len(examples)}개의 예제를 로드했습니다.")
                else:
                    print(f"경고: {Path(json_file).name}의 형식이 올바르지 않습니다. 리스트가 아닙니다.")
            except Exception as e:
                print(f"오류: {Path(json_file).name} 처리 중 예외 발생: {str(e)}")
    
    elif dataset_path.is_file() and dataset_path.suffix == '.json':
        # 단일 JSON 파일을 처리
        try:
            with open(dataset_path, 'r') as f:
                examples = json.load(f)
            
            if isinstance(examples, list):
                all_examples = examples
                print(f"{dataset_path.name}에서 {len(examples)}개의 예제를 로드했습니다.")
            else:
                print(f"경고: {dataset_path.name}의 형식이 올바르지 않습니다. 리스트가 아닙니다.")
                return
        except Exception as e:
            print(f"오류: {dataset_path.name} 처리 중 예외 발생: {str(e)}")
            return
    else:
        print(f"오류: {args.dataset}는 유효한 디렉토리 또는 JSON 파일이 아닙니다.")
        return
    
    # 전체 예제 수 확인
    total_examples = len(all_examples)
    if total_examples == 0:
        print("오류: 로드된 예제가 없습니다.")
        return
    
    print(f"\n총 {total_examples}개의 예제를 로드했습니다.")
    
    # 샘플 수 조정
    num_samples = min(args.samples, total_examples)
    
    # 무작위로 샘플 선택
    selected_examples = random.sample(all_examples, num_samples)
    
    # 선택된 샘플 시각화
    print(f"\n선택된 {num_samples}개의 샘플을 시각화합니다:\n")
    
    for i, example in enumerate(selected_examples):
        print(f"샘플 {i+1}/{num_samples}:")
        print("입력(Input):")
        render_grid(example["input"])
        print("\n출력(Output):")
        render_grid(example["output"])
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main() 