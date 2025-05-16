import argparse
import glob
import json
import random
import os
from pathlib import Path
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import inch
from PIL import Image, ImageDraw, ImageFont
import io

def create_grid_image(grid, cell_size=20):
    """
    그리드를 이미지로 변환합니다.
    """
    color_map = {
        0: (255, 255, 255),  # 하얀색
        1: (255, 0, 0),      # 빨간색
        2: (0, 255, 0),      # 초록색
        3: (0, 0, 255),      # 파란색
        4: (255, 255, 0),    # 노란색
        5: (128, 0, 128),    # 보라색
        6: (0, 255, 255),    # 청록색
        7: (0, 0, 0),        # 검은색
        8: (128, 128, 128),  # 회색
        9: (255, 165, 0),    # 주황색
    }
    
    height = len(grid)
    width = len(grid[0])
    
    img = Image.new('RGB', (width * cell_size, height * cell_size), 'white')
    draw = ImageDraw.Draw(img)
    
    for y, row in enumerate(grid):
        for x, cell in enumerate(grid[y]):
            color = color_map.get(cell, (255, 255, 255))
            draw.rectangle(
                [x * cell_size, y * cell_size, 
                 (x + 1) * cell_size, (y + 1) * cell_size],
                fill=color,
                outline='black'
            )
    
    return img

def create_visualization_row(examples, cell_size=20):
    """
    4개의 예제를 한 줄로 시각화합니다.
    """
    images = []
    for example in examples:
        input_img = create_grid_image(example["input"], cell_size)
        output_img = create_grid_image(example["output"], cell_size)
        
        # 입력과 출력 이미지를 가로로 연결
        combined_width = input_img.width + output_img.width
        combined_height = max(input_img.height, output_img.height)
        combined_img = Image.new('RGB', (combined_width, combined_height), 'white')
        combined_img.paste(input_img, (0, 0))
        combined_img.paste(output_img, (input_img.width, 0))
        images.append(combined_img)
    
    # 4개의 이미지를 가로로 연결
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    row_img = Image.new('RGB', (total_width, max_height), 'white')
    
    x_offset = 0
    for img in images:
        row_img.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return row_img

def main():
    parser = argparse.ArgumentParser(description='ARC 데이터셋 샘플 시각화')
    parser.add_argument('--dataset', type=str, required=True, help='데이터셋 디렉토리 경로')
    parser.add_argument('--output', type=str, default='visualization.pdf', help='출력 PDF 파일 경로')
    args = parser.parse_args()
    
    # JSON 파일 목록 가져오기
    json_files = glob.glob(os.path.join(args.dataset, "*.json"))
    if not json_files:
        print(f"경고: {args.dataset}에서 JSON 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(json_files)}개의 JSON 파일을 찾았습니다.")
    
    # PDF 생성
    c = canvas.Canvas(args.output, pagesize=landscape(A4))
    page_width, page_height = landscape(A4)
    
    y_position = page_height - 50  # 시작 위치
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                examples = json.load(f)
            
            if not isinstance(examples, list):
                print(f"경고: {Path(json_file).name}의 형식이 올바르지 않습니다.")
                continue
            
            # 4개의 랜덤 샘플 선택
            if len(examples) >= 4:
                selected_examples = random.sample(examples, 4)
            else:
                selected_examples = examples
                print(f"경고: {Path(json_file).name}에 4개 미만의 예제가 있습니다.")
            
            # 시각화 행 생성
            row_img = create_visualization_row(selected_examples)
            
            # 이미지를 PDF에 추가
            img_data = io.BytesIO()
            row_img.save(img_data, format='PNG')
            img_data.seek(0)
            
            # 이미지 크기 조정
            img_width = min(page_width - 100, row_img.width)
            img_height = img_width * row_img.height / row_img.width
            
            c.drawImage(img_data, 50, y_position - img_height, width=img_width, height=img_height)
            c.drawString(50, y_position + 10, f"File: {Path(json_file).name}")
            
            y_position -= img_height + 50
            
            # 새 페이지 필요시
            if y_position < 50:
                c.showPage()
                y_position = page_height - 50
            
        except Exception as e:
            print(f"오류: {Path(json_file).name} 처리 중 예외 발생: {str(e)}")
    
    c.save()
    print(f"시각화가 {args.output}에 저장되었습니다.")

if __name__ == "__main__":
    main() 