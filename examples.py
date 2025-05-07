from utils import render_grid  # 그리드 시각화 함수 임포트
from rich import print         # rich 라이브러리의 print 함수 (컬러 출력)
import os                      # 파일/디렉토리 작업용 표준 라이브러리
import pandas as pd            # 데이터프레임 처리를 위한 pandas
from typing import List        # 타입 힌트용
import json                    # JSON 파일 입출력용
import numpy as np             # 수치 연산 및 랜덤 샘플링용

# 데이터셋을 불러와서 랜덤 샘플링하는 함수
def load_data(base_dir):
    filenames = os.listdir(base_dir)  # base_dir 내 파일명 리스트
    data_files = [os.path.join(base_dir, p) for p in filenames if ".json" in p]  # json 파일만 추출

    dataset = []
    for fn in data_files:
        with open(fn) as fp:
            data = json.load(fp)  # 각 json 파일을 파싱
        dataset.append(data)      # 리스트에 추가

    filenames = [fn.split(".")[0] for fn in filenames]  # 파일명에서 확장자 제거
    data = []
    MAX_LEN = 10
    rng = np.random.default_rng(42)  # 랜덤 시드 고정

    N = len(dataset)  # 데이터셋 개수

    # MAX_LEN만큼 랜덤 샘플링하여 데이터 생성
    while len(data) < MAX_LEN:
        task_idx = rng.integers(0, N)  # 랜덤하게 task 선택
        task = dataset[task_idx]
        file_name = filenames[task_idx]

        n_task = len(task)  # 해당 task 내 샘플 개수
        grids_idx =  rng.choice(n_task, size=4, replace=True)  # 4개 랜덤 선택
        train_grids = [task[i] for i in grids_idx[:3]]         # 앞 3개는 train
        test_grids = [task[i] for i in grids_idx[3:]]          # 마지막 1개는 test

        test_inputs = [{'input': grid['input']} for grid in test_grids]  # test input만 추출
        test_outputs = [grid['output'] for grid in test_grids]           # test output만 추출
        test_outputs_transformed = [{'output': grid} for grid in test_outputs]  # output 포맷 맞춤
        combined_tests = []
        for test_input, test_output in zip(test_inputs, test_outputs_transformed):
            combined_tests.append({'input': test_input['input'], 'output': test_output['output']})  # input/output 묶기

        data.append({
            'task': file_name,
            'train': train_grids,
            'test_input': test_inputs,
            'test_output': test_outputs,
            'test': combined_tests,
        })

    df = pd.DataFrame(data)  # pandas DataFrame으로 변환
    return df

N_data = 4
data_path = "/workspace/dataset"  # 데이터셋 경로
df = load_data(data_path)         # 데이터셋 로드

from datasets import Dataset
dataset = Dataset.from_pandas(df).shuffle(42).select(range(N_data))  # huggingface datasets로 변환, 셔플 후 N_data개 선택

print("-----Dataset Statistics-----")
print(dataset)  # 데이터셋 통계 출력
print("-----Train Question Example-----")
print(dataset[0]['train'])  # 첫 번째 샘플의 train 데이터 출력
print("-----Test Question Example-----")
print(dataset[0]['test'])   # 첫 번째 샘플의 test 데이터 출력

print("Train Input")
render_grid(dataset[0]['train'][0]['input'])  # 첫 번째 train input을 컬러풀하게 출력
print("Train Output")
render_grid(dataset[0]['train'][0]['output']) # 첫 번째 train output을 컬러풀하게 출력

print("Test Input")
render_grid(dataset[0]['test'][0]['input'])   # 첫 번째 test input을 컬러풀하게 출력
print("Test Output")
render_grid(dataset[0]['test'][0]['output'])  # 첫 번째 test output을 컬러풀하게 출력
