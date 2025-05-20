import json
import os
import numpy as np
from typing import List, Dict, Any
import glob

def is_valid_grid(grid: List[List[int]]) -> bool:
    """
    2차원 배열이 맞고, 모든 값이 0-9 사이의 정수인지 검증
    크기가 10x10을 초과하지 않는지도 검증
    """
    try:
        # null 체크
        if grid is None:
            return False
            
        # numpy 배열로 변환
        grid_np = np.array(grid)
        
        # 2차원 배열인지 확인
        if len(grid_np.shape) != 2:
            return False
            
        # 크기가 10x10을 초과하는지 확인
        if grid_np.shape[0] > 10 or grid_np.shape[1] > 10:
            return False
            
        # 모든 값이 0-9 사이의 정수인지 확인
        if not np.all((grid_np >= 0) & (grid_np <= 9)):
            return False
            
        return True
    except:
        return False

def validate_and_fix_json(file_path: str) -> bool:
    """
    JSON 파일을 검증하고 필요한 경우 수정
    Returns:
        bool: 파일이 수정되었는지 여부
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        modified = False
        valid_examples = []
        
        for example in data:
            # input과 output 키가 있는지 먼저 확인
            if 'input' not in example or 'output' not in example:
                modified = True
                print(f"Missing input/output key in {file_path}")
                continue
                
            input_grid = example['input']
            output_grid = example['output']
            
            # input과 output이 모두 유효한 그리드인 경우만 유지
            if is_valid_grid(input_grid) and is_valid_grid(output_grid):
                valid_examples.append(example)
            else:
                modified = True
                print(f"Invalid grid found in {file_path}:")
                if not is_valid_grid(input_grid):
                    print(f"  Invalid input grid: {input_grid}")
                if not is_valid_grid(output_grid):
                    print(f"  Invalid output grid: {output_grid}")
        
        # 유효한 예제가 20개 미만인 경우 파일 삭제
        if len(valid_examples) < 20:
            print(f"Deleting {file_path} due to insufficient valid examples ({len(valid_examples)})")
            os.remove(file_path)
            return True
            
        # 유효한 예제가 20개 이상인 경우 파일 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(valid_examples, f, indent=2)
        return modified
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        # 파일 전체가 에러인 경우 파일 삭제
        print(f"Deleting {file_path} due to processing error")
        os.remove(file_path)
        return True

def main():
    # dataset_* 패턴의 모든 폴더 찾기
    dataset_dirs = glob.glob("big_dataset")
    
    total_files = 0
    modified_files = 0
    
    print("검사할 데이터셋 폴더들:")
    for dir_path in dataset_dirs:
        print(f"- {dir_path}")
    
    # 각 데이터셋 폴더 내의 모든 json 파일 찾기
    for dataset_dir in dataset_dirs:
        json_files = glob.glob(os.path.join(dataset_dir, "**/*.json"), recursive=True)
        total_files += len(json_files)
        
        print(f"\n{dataset_dir} 폴더에서 {len(json_files)}개의 JSON 파일을 찾았습니다.")
        
        for file_path in json_files:
            print(f"\n처리 중: {file_path}...")
            if validate_and_fix_json(file_path):
                modified_files += 1
                print(f"수정됨: {file_path}")
    
    print(f"\n검증 완료!")
    print(f"총 처리된 파일 수: {total_files}")
    print(f"수정된 파일 수: {modified_files}")

if __name__ == "__main__":
    main() 