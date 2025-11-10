import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image # Pillow 라이브러리

def move_file_pair(txt_file, img_file, target_dir):
    """.txt와 .img 파일을 안전하게 target_dir로 이동시킵니다."""
    try:
        shutil.move(str(txt_file), str(target_dir / txt_file.name))
    except Exception as e:
        print(f"    -> .txt 이동 실패: {txt_file.name} ({e})")
        
    if img_file and img_file.exists():
        try:
            shutil.move(str(img_file), str(target_dir / img_file.name))
        except Exception as e:
            print(f"    -> 이미지 이동 실패: {img_file.name} ({e})")
    elif not img_file:
        print(f"    -> [경고] {txt_file.name}의 짝이 되는 이미지 파일을 찾지 못했습니다.")

def validate_and_quarantine(source_folder):
    """
    YOLO 라벨을 검증하여 "5 키트, 각 키트당 11 센서" 구조가 아닌 파일을
    _invalid_structure_files 폴더로 격리(이동)합니다.
    
    가정: class 0 = 키트 (검사지), class 1 = 센서 (검사 패드)
    """
    
    source_dir = Path(source_folder)
    quarantine_dir = source_dir / "_invalid_structure_files"
    quarantine_dir.mkdir(exist_ok=True)
    
    print(f"'{source_dir}' 폴더 스캔 시작...")
    print(f"유효하지 않은 파일은 '{quarantine_dir.name}' 폴더로 이동합니다.\n")
    
    files_checked = 0
    files_moved = 0
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.PNG', '.JPEG']
    
    for txt_file in source_dir.glob("*.txt"):
        files_checked += 1
        base_name = txt_file.stem
        image_file = None
        
        # 1. 짝이 되는 이미지 파일 찾기
        for ext in image_extensions:
            potential_img_file = source_dir / (base_name + ext)
            if potential_img_file.exists():
                image_file = potential_img_file
                break
        
        if not image_file:
            print(f"[경고] {txt_file.name}: 짝이 되는 이미지 파일을 찾지 못함. (텍스트 파일만 이동)")
            # 텍스트 파일만 이동시킬 수도 있지만, 여기서는 일단 건너뜀
            # move_file_pair(txt_file, None, quarantine_dir) 
            continue

        reason = ""
        is_valid = True
        
        try:
            # 2. 이미지 크기 읽기
            with Image.open(image_file) as img:
                img_w, img_h = img.size
            
            # 3. 라벨 파일 읽고 절대 좌표로 변환
            with open(txt_file, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            if not lines:
                is_valid = False
                reason = "파일이 비어 있음"
            
            labels = []
            if is_valid:
                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        is_valid = False
                        reason = f"잘못된 라벨 형식 (항목 5개 필요): {line}"
                        break
                    
                    cls_id, x, y, w, h = map(float, parts)
                    x1 = (x - w / 2) * img_w
                    y1 = (y - h / 2) * img_h
                    x2 = (x + w / 2) * img_w
                    y2 = (y + h / 2) * img_h
                    labels.append([x1, y1, x2, y2, int(cls_id)])
            
            if not is_valid: # 라벨 파싱 중 오류 발생
                print(f"[오류] {txt_file.name}: {reason}")
                move_file_pair(txt_file, image_file, quarantine_dir)
                files_moved += 1
                continue

            # 4. NumPy 배열로 변환 및 클래스별 분리
            labels_np = np.array(labels)
            kits = labels_np[labels_np[:, 4] == 0]     # class 0 (키트)
            sensors = labels_np[labels_np[:, 4] == 1]  # class 1 (센서)

            # 5. 1차 검증 (개수)
            if len(kits) != 5:
                is_valid = False
                reason = f"키트(class 0) 개수 오류 (5개 필요, {len(kits)}개 발견)"
            elif len(sensors) != 55:
                is_valid = False
                reason = f"센서(class 1) 개수 오류 (55개 필요, {len(sensors)}개 발견)"
            
            # 6. 2차 검증 (포함 관계)
            if is_valid:
                for i, kit in enumerate(kits):
                    k_x1, k_y1, k_x2, k_y2, _ = kit
                    
                    # 센서의 *중심점*이 아닌 *박스 전체*가 포함되는지 확인
                    # Dataset 코드와 동일한 로직 (>) 사용
                    mask = (
                        (sensors[:, 0] > k_x1) & (sensors[:, 0] < k_x2) & # sensor.x1 > kit.x1 ...
                        (sensors[:, 2] < k_x2) & (sensors[:, 2] > k_x1) & # sensor.x2 < kit.x2 ...
                        (sensors[:, 1] > k_y1) & (sensors[:, 1] < k_y2) & # sensor.y1 > kit.y1 ...
                        (sensors[:, 3] < k_y2) & (sensors[:, 3] > k_y1)  # sensor.y2 < kit.y2 ...
                    )
                    
                    # Dataset 로직이 센서의 (x1, y1, x2, y2) 좌표를 기준으로 한 것이라면
                    # 위 mask가 정확합니다. 만약 중심점 기준이었다면 아래를 사용합니다.
                    # (제공된 Dataset 코드를 보면 x1, y1, x2, y2 기준으로 필터링하므로 위 mask가 맞습니다)
                    
                    # *참고: Dataset 코드 로직 (센서 박스의 (x1, y1) 좌표만 체크)*
                    # mask_from_dataset_code = (
                    #     (sensors[:, 0] > k_x1) & (sensors[:, 0] < k_x2) & 
                    #     (sensors[:, 1] > k_y1) & (sensors[:, 1] < k_y2)
                    # )
                    # contained_sensors_count = np.sum(mask_from_dataset_code)
                    
                    contained_sensors_count = np.sum(mask)

                    if contained_sensors_count != 11:
                        is_valid = False
                        reason = f"키트 #{i+1}이 {contained_sensors_count}개의 센서만 포함 (11개 필요)"
                        break # 이 키트가 실패했으므로 더 검사할 필요 없음

        except Exception as e:
            is_valid = False
            reason = f"파일 처리 중 심각한 오류 발생: {e}"

        # 7. 최종 결정 및 이동
        if not is_valid:
            print(f"[파일 이동] {txt_file.name}: {reason}")
            move_file_pair(txt_file, image_file, quarantine_dir)
            files_moved += 1
            
    # 8. 최종 요약
    print("\n--- 작업 완료 ---")
    print(f"총 {files_checked}개의 .txt 파일을 확인했습니다.")
    print(f"총 {files_moved}개의 (이미지+레이블) 쌍을 '{quarantine_dir.name}' 폴더로 이동했습니다.")

# --- 스크립트 실행 ---

# (중요) 검사할 폴더 경로를 여기에 입력하세요.
# Windows 경로는 r"..." 또는 "C:\\Users\\..." 형태로 입력해야 합니다.
TARGET_DIRECTORY = r"C:\Users\seoja\Desktop\new_dipstick\flat_images라벨검사용3"

validate_and_quarantine(TARGET_DIRECTORY)