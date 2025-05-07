import numpy as np
import random
from typing import List, Dict, Tuple
import copy
import argparse
import glob
import json
import os
from pathlib import Path

class GridAugmentor:
    def __init__(self, seed: int = 42):
        """
        그리드 데이터 증강을 위한 클래스
        
        Args:
            seed (int): 랜덤 시드
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
    def rotate_90(self, grid: List[List[int]]) -> List[List[int]]:
        """90도 회전"""
        return list(map(list, zip(*grid[::-1])))
    
    def rotate_180(self, grid: List[List[int]]) -> List[List[int]]:
        """180도 회전"""
        return [row[::-1] for row in grid[::-1]]
    
    def rotate_270(self, grid: List[List[int]]) -> List[List[int]]:
        """270도 회전"""
        return list(map(list, zip(*grid)))[::-1]
    
    def flip_horizontal(self, grid: List[List[int]]) -> List[List[int]]:
        """좌우 반전"""
        return [row[::-1] for row in grid]
    
    def flip_vertical(self, grid: List[List[int]]) -> List[List[int]]:
        """상하 반전"""
        return grid[::-1]
    
    def transpose(self, grid: List[List[int]]) -> List[List[int]]:
        """전치"""
        return list(map(list, zip(*grid)))
    
    def color_permutation(self, grid: List[List[int]], preserve_zero: bool = True) -> Tuple[List[List[int]], Dict[int, int]]:
        """
        색상 번호 무작위 치환
        
        Args:
            grid: 입력 그리드
            preserve_zero: 0을 보존할지 여부
            
        Returns:
            변환된 그리드와 색상 매핑 딕셔너리
        """
        grid_np = np.array(grid)
        unique_colors = np.unique(grid_np)
        if preserve_zero and 0 in unique_colors:
            unique_colors = unique_colors[unique_colors != 0]
        
        # 색상 매핑 생성
        color_mapping = {0: 0} if preserve_zero else {}
        shuffled_colors = list(unique_colors)
        random.shuffle(shuffled_colors)
        for old_color, new_color in zip(unique_colors, shuffled_colors):
            color_mapping[old_color] = new_color
        
        # 색상 변환
        new_grid = [[color_mapping[cell] for cell in row] for row in grid]
        return new_grid, color_mapping
    
    def inverse_color(self, grid: List[List[int]], preserve_zero: bool = True) -> List[List[int]]:
        """
        역색 변환 (9 - old_color)
        
        Args:
            grid: 입력 그리드
            preserve_zero: 0을 보존할지 여부
        """
        new_grid = []
        for row in grid:
            new_row = []
            for cell in row:
                if cell == 0 and preserve_zero:
                    new_row.append(0)
                else:
                    new_row.append(9 - cell)
            new_grid.append(new_row)
        return new_grid
    
    def scale_up(self, grid: List[List[int]], factor: int = 2) -> List[List[int]]:
        """
        그리드 크기 확대
        
        Args:
            grid: 입력 그리드
            factor: 확대 배수
        """
        if factor <= 1:
            return grid
            
        new_grid = []
        for row in grid:
            # 각 행을 factor번 반복
            for _ in range(factor):
                new_row = []
                for cell in row:
                    # 각 셀을 factor번 반복
                    new_row.extend([cell] * factor)
                new_grid.append(new_row)
        return new_grid
    
    def augment_example(self, example: Dict, augmentations: List[str]) -> List[Dict]:
        """
        단일 예제에 대한 데이터 증강
        
        Args:
            example: {"input": grid, "output": grid} 형태의 예제
            augmentations: 적용할 증강 방법 리스트
            
        Returns:
            증강된 예제들의 리스트
        """
        augmented_examples = [example]  # 원본 예제 포함
        
        input_grid = example["input"]
        output_grid = example["output"]
        
        for aug in augmentations:
            if aug == "rotate_90":
                new_input = self.rotate_90(input_grid)
                new_output = self.rotate_90(output_grid)
                augmented_examples.append({"input": new_input, "output": new_output})
                
            elif aug == "rotate_180":
                new_input = self.rotate_180(input_grid)
                new_output = self.rotate_180(output_grid)
                augmented_examples.append({"input": new_input, "output": new_output})
                
            elif aug == "rotate_270":
                new_input = self.rotate_270(input_grid)
                new_output = self.rotate_270(output_grid)
                augmented_examples.append({"input": new_input, "output": new_output})
                
            elif aug == "flip_h":
                new_input = self.flip_horizontal(input_grid)
                new_output = self.flip_horizontal(output_grid)
                augmented_examples.append({"input": new_input, "output": new_output})
                
            elif aug == "flip_v":
                new_input = self.flip_vertical(input_grid)
                new_output = self.flip_vertical(output_grid)
                augmented_examples.append({"input": new_input, "output": new_output})
                
            elif aug == "transpose":
                new_input = self.transpose(input_grid)
                new_output = self.transpose(output_grid)
                augmented_examples.append({"input": new_input, "output": new_output})
                
            elif aug == "color_perm":
                new_input, color_mapping = self.color_permutation(input_grid)
                # output에도 동일한 색상 매핑 적용
                new_output = [[color_mapping[cell] for cell in row] for row in output_grid]
                augmented_examples.append({"input": new_input, "output": new_output})
                
            elif aug == "inverse_color":
                new_input = self.inverse_color(input_grid)
                new_output = self.inverse_color(output_grid)
                augmented_examples.append({"input": new_input, "output": new_output})
                
            elif aug == "scale_up":
                new_input = self.scale_up(input_grid)
                new_output = self.scale_up(output_grid)
                augmented_examples.append({"input": new_input, "output": new_output})
        
        return augmented_examples
    
    def augment_dataset(self, examples: List[Dict], augmentations: List[str]) -> List[Dict]:
        """
        전체 데이터셋에 대한 데이터 증강
        
        Args:
            examples: 예제들의 리스트
            augmentations: 적용할 증강 방법 리스트
            
        Returns:
            증강된 예제들의 리스트
        """
        augmented_dataset = []
        
        for example in examples:
            augmented_examples = self.augment_example(example, augmentations)
            augmented_dataset.extend(augmented_examples)
        
        return augmented_dataset

def example():
    # 테스트용 예제
    test_example = {
        "input": [
            [8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 0],
            [8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 6],
            [8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8]
        ],
        "output": [
            [8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8],
            [6, 6, 6, 6, 6, 6, 6, 6],
            [8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8]
        ]
    }
    
    # 증강기 초기화
    augmentor = GridAugmentor()
    
    # 적용할 증강 방법들 (회전 및 반전은 문제의 의미가 달라질 수 있으므로 주의)
    augmentations = [
        "rotate_90",
        "rotate_180",
        "rotate_270",
        "flip_h",
        "flip_v",
        "transpose",
        "color_perm",
        "inverse_color",
        "scale_up"
    ]
    
    # 증강된 예제 생성
    augmented_examples = augmentor.augment_example(test_example, augmentations)
    
    # 결과 출력
    print(f"원본 예제 1개에 대해 {len(augmented_examples)}개의 증강된 예제가 생성되었습니다.")
    
    # 증강된 예제 시각화 (utils.py의 render_grid 함수 사용)
    from utils import render_grid
    for i, example in enumerate(augmented_examples):
        print(f"\n증강된 예제 {i+1}:")
        print("Input:")
        render_grid(example["input"])
        print("Output:")
        render_grid(example["output"])

def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='ARC 데이터셋 증강')
    parser.add_argument('--dataset', type=str, required=True, help='데이터셋 디렉토리 경로')
    parser.add_argument('--output', type=str, default=None, help='출력 디렉토리 경로 (기본값: dataset 디렉토리)')
    parser.add_argument('--augmentations', type=str, nargs='+', 
                       default=['rotate_90', 'rotate_180', 'rotate_270', 'flip_h', 'flip_v', 
                               'transpose', 'color_perm', 'inverse_color', 'scale_up'],
                       help='적용할 증강 방법들')
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    output_dir = args.output if args.output else args.dataset
    os.makedirs(output_dir, exist_ok=True)
    
    # 증강기 초기화
    augmentor = GridAugmentor()
    
    # JSON 파일 목록 가져오기
    json_files = glob.glob(os.path.join(args.dataset, "*.json"))
    if not json_files:
        print(f"경고: {args.dataset}에서 JSON 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(json_files)}개의 JSON 파일을 찾았습니다.")
    
    # 각 파일에 대해 증강 수행
    for json_file in json_files:
        filename = Path(json_file).stem
        print(f"\n{filename}.json 처리 중...")
        
        try:
            # JSON 파일 읽기
            with open(json_file, 'r') as f:
                examples = json.load(f)
            
            if not isinstance(examples, list):
                print(f"경고: {filename}.json의 형식이 올바르지 않습니다. 리스트가 아닙니다.")
                continue
                
            print(f"  - {len(examples)}개의 예제 발견")
            
            # 각 증강 방법에 대해 별도의 파일 생성
            for aug in args.augmentations:
                print(f"  - {aug} 증강 적용 중...")
                
                # 증강된 데이터셋 생성
                augmented_examples = []
                for example in examples:
                    # 단일 예제에 대해 현재 증강 방법만 적용
                    augmented = augmentor.augment_example(example, [aug])
                    # 원본 예제는 제외하고 증강된 예제만 추가
                    augmented_examples.extend(augmented[1:])
                
                # 증강된 데이터셋 저장
                output_file = os.path.join(output_dir, f"{filename}_{aug}.json")
                with open(output_file, 'w') as f:
                    json.dump(augmented_examples, f, indent=2)
                
                print(f"    - {len(augmented_examples)}개의 증강된 예제를 {output_file}에 저장했습니다.")
        
        except Exception as e:
            print(f"오류: {filename}.json 처리 중 예외 발생: {str(e)}")
            continue
    
    print("\n모든 파일의 증강이 완료되었습니다.")

if __name__ == "__main__":
    main()
