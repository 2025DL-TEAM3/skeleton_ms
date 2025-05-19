import argparse
import glob
import random
import numpy as np
import torch
from datasets import load_dataset
from arc.arc import ARCSolver

def main():
    parser = argparse.ArgumentParser(description='Train ARCSolver with ARC dataset')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace token')
    parser.add_argument('--dataset', type=str, default='/home/student/workspace/dataset', help='Dataset name or path')
    parser.add_argument('--save_dir', type=str, default='artifacts/qwen3-4b-lora', help='Save directory')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume training from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # 모든 랜덤 시드 고정
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 데이터셋 로드 및 분할
    print("Loading and splitting dataset...")
    data_files = sorted(glob.glob(f"{args.dataset}/*.json"))  # 정렬된 순서로 파일 목록 가져오기
    random.shuffle(data_files)  # 고정된 시드로 섞기
    split_idx = int(len(data_files) * 0.95)
    train_files = data_files[:split_idx]
    val_files = data_files[split_idx:]
    print(f"Train files: {len(train_files)}, Validation files: {len(val_files)}")
    
    dataset = {}
    dataset['train'] = load_dataset('json', data_files=train_files)['train']
    dataset['train'].data_files = train_files
    dataset['validation'] = load_dataset('json', data_files=val_files)['train'] if val_files else None
    if dataset['validation']:
        dataset['validation'].data_files = val_files
    
    # ARCSolver 인스턴스 생성
    print("Initializing model...")
    solver = ARCSolver(token=args.token)

    # configuration
    batch_size = 1
    lr = 5e-5
    num_epochs = 4
    steps_per_file = 50 # should be multiple of batch_size
    steps_accum = 4
    warmup_rate = 0.1
    fixed_seed = args.seed

    # validation configuration
    patience = 10
    val_steps = 20000
    val_steps_per_file = 1
    max_val_files = 128
    val_batch_size = 4
    
    # 모델 학습
    print("Starting training...")
    solver.train(
        train_dataset=dataset['train'], 
        batch_size=batch_size, 
        lr=lr, 
        num_epochs=num_epochs, 
        steps_per_file=steps_per_file, 
        steps_accum=steps_accum, 
        warmup_rate=warmup_rate,
        save_dir=args.save_dir,
        resume_from=args.resume_from,
        validation_dataset=dataset['validation'],
        patience=patience,
        val_steps=val_steps,
        fixed_seed=fixed_seed,
        max_val_files=max_val_files,
        val_steps_per_file=val_steps_per_file,
        val_batch_size=val_batch_size
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()
