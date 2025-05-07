import numpy as np
from tqdm.auto import tqdm
import os

from transformers import set_seed
from datasets import load_dataset
import pandas as pd
import json

# 예측 결과와 정답이 일치하는지 확인하는 함수
def check_match(pred, truth):
    pred = np.array(pred, dtype=np.uint8)
    truth = np.array(truth, dtype=np.uint8)

    # 2차원 배열이 아니거나, shape이 다르면 오답 처리
    if len(pred.shape) != 2 or pred.shape != truth.shape:
        return 0
    else:
        # 완전히 일치하면 1, 아니면 0
        return int(np.all(pred == truth))

# 데이터셋을 불러와서 랜덤 샘플링하는 함수
def load_data(base_dir):
    filenames = os.listdir(base_dir)
    data_files = [os.path.join(base_dir, p) for p in filenames if ".json" in p]

    dataset = []
    for fn in data_files:
        with open(fn) as fp:
            data = json.load(fp)  # 각 json 파일을 파싱
        dataset.append(data)      # 리스트에 추가

    filenames = [fn.split(".")[0] for fn in filenames]  # 파일명에서 확장자 제거
    data = []
    MAX_LEN = 1000  # 최대 샘플 개수
    rng = np.random.default_rng(42)  # 랜덤 시드 고정

    N = len(dataset)

    # MAX_LEN만큼 랜덤 샘플링하여 데이터 생성
    while len(data) < MAX_LEN:
        task_idx = rng.integers(0, N)
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

# 평가 전체를 실행하는 메인 함수
def main():
    token = os.environ.get("HF_TOKEN", None)  # 환경변수에서 HF_TOKEN(허깅페이스 토큰) 읽기
    from arc import ARCSolver  # arc 모듈에서 ARCSolver 클래스 임포트

    solver = ARCSolver(token=token)  # 평가용 solver 객체 생성
    solver.prepare_evaluation()      # 평가 준비 (모델 로딩 등)

    set_seed(1234567890)  # 랜덤 시드 고정

    data_path = "/home/student/workspace/dataset"  # 데이터셋 경로
    N_data = 10  # 평가에 사용할 샘플 개수

    scores = []
    df = load_data(data_path)  # 데이터셋 로드

    from datasets import Dataset
    eval_dataset = Dataset.from_pandas(df).shuffle(42).select(range(N_data))  # huggingface datasets로 변환, 셔플 후 N_data개 선택

    # tqdm으로 진행상황을 표시하며 평가 반복
    for eval_data in tqdm(eval_dataset):
        preds = solver.predict(
            eval_data["train"],               # train 데이터
            eval_data["test"][0]["input"],    # test input (첫 번째)
        )
        s = check_match(preds, eval_data["test"][0]["output"])  # 예측과 정답 비교
        scores.append(s)
    
    score = np.array(scores).mean() * 100  # 평균 점수(%) 계산
    print(f"Evaluation scores: {score:.2f}", flush=True)
    print("Evaluation Success")

# 메인 함수 실행
if __name__ == "__main__":
    main()
