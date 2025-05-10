import argparse
import glob
from datasets import load_dataset
from arc.arc import ARCSolver

def main():
    parser = argparse.ArgumentParser(description='Train ARCSolver with ARC dataset')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace token')
    parser.add_argument('--dataset', type=str, default='/home/student/workspace/dataset', help='Dataset name or path')
    args = parser.parse_args()
    
    # 데이터셋 로드
    print("Loading dataset...")
    data_files = glob.glob(f"{args.dataset}/*.json")
    dataset = load_dataset('json', data_files=data_files)
    # data_files 속성 추가
    dataset['train'].data_files = data_files
    
    # ARCSolver 인스턴스 생성
    print("Initializing model...")
    solver = ARCSolver(token=args.token)

    # configuration
    batch_size = 1
    lr = 1e-4
    num_epochs = 5
    steps_per_file = 50
    steps_accum = 4
    warmup_rate = 0.1
    
    # 모델 학습
    print("Starting training...")
    solver.train(dataset['train'], batch_size, lr, num_epochs, steps_per_file, steps_accum, warmup_rate, resume_from=None)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
