import numpy as np
from tqdm.auto import tqdm
import os
import argparse
import multiprocessing

from transformers import set_seed
from datasets import load_dataset
import pandas as pd
import json
from torch.utils.data import DataLoader

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

# 단일 파일을 로드하는 함수 (멀티프로세싱용)
def load_single_file(file_path):
    try:
        with open(file_path) as fp:
            return json.load(fp)
    except Exception as e:
        print(f"파일 로드 오류: {file_path} - {e}")
        return None

# 데이터셋을 불러와서 랜덤 샘플링하는 함수
def load_data(base_dir, num_workers=1):
    filenames = os.listdir(base_dir)
    data_files = [os.path.join(base_dir, p) for p in filenames if ".json" in p]

    # 멀티프로세싱으로 파일 로드
    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(load_single_file, data_files), total=len(data_files), desc="파일 로딩"))
        dataset = [data for data in results if data is not None]
    else:
        dataset = []
        for fn in tqdm(data_files, desc="파일 로딩"):
            try:
                with open(fn) as fp:
                    data = json.load(fp)  # 각 json 파일을 파싱
                dataset.append(data)      # 리스트에 추가
            except Exception as e:
                print(f"파일 로드 오류: {fn} - {e}")

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
    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(description='ARC 모델 평가')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace 토큰')
    parser.add_argument('--dataset', type=str, default='dataset', help='데이터셋 경로')
    parser.add_argument('--model_path', type=str, default=None, help='모델 경로')
    parser.add_argument('--num_examples', type=int, default=100, help='평가에 사용할 샘플 개수')
    parser.add_argument('--num_workers', type=int, default=4, help='데이터 로딩에 사용할 워커 수')
    args = parser.parse_args()

    # 토큰 가져오기
    token = args.token or os.environ.get("HF_TOKEN", None)
    
    from arc import ARCSolver  # arc 모듈에서 ARCSolver 클래스 임포트

    solver = ARCSolver(token=token)  # 평가용 solver 객체 생성
    solver.prepare_evaluation(args.model_path)  # 평가 준비 (모델 로딩 등)

    set_seed(1234567890)  # 랜덤 시드 고정

    print(f"데이터셋 로드 중... (경로: {args.dataset}, 워커 수: {args.num_workers})")
    df = load_data(args.dataset, num_workers=args.num_workers)  # 데이터셋 로드

    from datasets import Dataset
    eval_dataset = Dataset.from_pandas(df).shuffle(42).select(range(min(args.num_examples, len(df))))

    # 평가 실행
    scores = []
    print(f"{len(eval_dataset)}개의 샘플에 대해 평가를 시작합니다...")
    for eval_data in tqdm(eval_dataset, desc="평가 진행"):
        try:
            # 데이터 형식 확인 및 처리
            test_input = eval_data["test"][0]["input"]
            
            # numpy 배열로 변환하여 shape 문제 해결
            if not isinstance(test_input, np.ndarray):
                test_input = np.array(test_input)
                
            preds = solver.predict(
                eval_data["train"],     # train 데이터
                test_input,            # test input (첫 번째)
            )
            s = check_match(preds, eval_data["test"][0]["output"])  # 예측과 정답 비교
            scores.append(s)
            
        except Exception as e:
            print(f"평가 중 오류 발생: {e}")
            continue
    
    if scores:
        score = np.array(scores).mean() * 100  # 평균 점수(%) 계산
        print(f"Evaluation scores: {score:.2f}%", flush=True)
        print(f"성공한 평가 수: {len(scores)}/{len(eval_dataset)}")
    else:
        print("오류로 인해 평가 결과가 없습니다.")
    
    print("Evaluation Success")

# 메인 함수 실행
if __name__ == "__main__":
    main()
