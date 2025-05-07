import argparse
import os
from transformers import set_seed
from datasets import load_dataset
from arc import ARCSolver
import json
import glob
import random

def load_arc_dataset(dataset_path, train_ratio=0.8):
    """
    ARC 데이터셋을 로드하고 전처리하는 함수
    
    Args:
        dataset_path (str): 데이터셋이 있는 디렉토리 경로
        train_ratio (float): 학습 데이터 비율 (0.0 ~ 1.0)
        
    Returns:
        dataset: HuggingFace dataset 형식의 데이터셋
    """
    # JSON 파일 목록 가져오기
    data_files = glob.glob(f"{dataset_path}/*.json")
    if not data_files:
        raise ValueError(f"데이터셋 경로에서 JSON 파일을 찾을 수 없습니다: {dataset_path}")
    
    print(f"총 {len(data_files)}개의 JSON 파일을 찾았습니다.")
    
    # 각 파일별로 데이터 처리
    processed_data = []
    total_examples = 0
    total_train = 0
    total_test = 0
    
    for file_path in data_files:
        try:
            with open(file_path, 'r') as f:
                file_examples = json.load(f)
                if not isinstance(file_examples, list) or len(file_examples) < 2:
                    continue
                
                # 각 파일별로 학습/테스트 분할
                random.shuffle(file_examples)
                split_idx = max(1, int(len(file_examples) * train_ratio))  # 최소 1개는 학습 데이터로
                train_examples = file_examples[:split_idx]
                test_examples = file_examples[split_idx:]
                
                # 각 테스트 예제에 대해 학습-테스트 쌍 생성
                for test_example in test_examples:
                    processed_data.append({
                        "train": train_examples,
                        "test": [{"input": test_example["input"], "output": test_example["output"]}]
                    })
                
                total_examples += len(file_examples)
                total_train += len(train_examples)
                total_test += len(test_examples)
                
                print(f"파일 {os.path.basename(file_path)}: {len(file_examples)}개 예제")
                print(f"  - 학습: {len(train_examples)}개")
                print(f"  - 테스트: {len(test_examples)}개")
                
        except Exception as e:
            print(f"파일 로드 중 오류 발생: {file_path} - {e}")
    
    if not processed_data:
        raise ValueError("유효한 예제가 포함된 JSON 파일이 없습니다.")
    
    print("\n전체 데이터셋 통계:")
    print(f"총 예제 수: {total_examples}개")
    print(f"총 학습 예제 수: {total_train}개")
    print(f"총 테스트 예제 수: {total_test}개")
    print(f"생성된 학습-테스트 쌍: {len(processed_data)}개")
    
    # HuggingFace dataset으로 변환
    from datasets import Dataset
    dataset = Dataset.from_list(processed_data)
    return dataset

def main():
    parser = argparse.ArgumentParser(description='Qwen3-4B 모델을 ARC 데이터셋으로 파인튜닝')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace 토큰')
    parser.add_argument('--dataset', type=str, default='/home/student/workspace/dataset', 
                        help='데이터셋 경로')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--train-ratio', type=float, default=0.8, 
                        help='학습 데이터 비율 (0.0 ~ 1.0)')
    args = parser.parse_args()
    
    # 랜덤 시드 설정
    set_seed(args.seed)
    
    # 데이터셋 로드
    print("데이터셋 로드 중...")
    dataset = load_arc_dataset(args.dataset, args.train_ratio)
    print(f"로드된 데이터셋 크기: {len(dataset)}")
    
    # ARCSolver 인스턴스 생성
    print("모델 초기화 중...")
    solver = ARCSolver(token=args.token)
    
    # 모델 학습
    print("학습 시작...")
    solver.train(dataset)
    
    print("학습 완료! 결과는 artifacts/checkpoint-final에 저장되었습니다.")

if __name__ == "__main__":
    main()
