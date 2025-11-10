import shutil
from pathlib import Path

def quarantine_mismatched_labels(source_folder, required_lines=60):
    """
    지정된 폴더를 스캔하여 .txt 레이블 파일의 줄 수가 60이 아닌 경우,
    해당 .txt 파일과 짝이 되는 이미지 파일을 '_mismatched_files' 폴더로 이동합니다.
    """
    
    # 1. 경로 설정
    source_dir = Path(source_folder)
    # 60줄이 아닌 파일들을 모아둘 "격리" 폴더 생성
    quarantine_dir = source_dir / "_mismatched_files"
    
    # 격리 폴더가 없으면 생성
    quarantine_dir.mkdir(exist_ok=True)
    
    print(f"'{source_dir}' 폴더를 스캔합니다...")
    print(f"줄 수가 {required_lines}이(가) 아닌 파일은 '{quarantine_dir.name}' 폴더로 이동합니다.\n")
    
    # 2. 카운터 초기화
    txt_files_checked = 0
    moved_pairs = 0
    
    # 3. 원본 폴더 내의 모든 .txt 파일 순회
    for txt_file in source_dir.glob("*.txt"):
        txt_files_checked += 1
        line_count = 0
        
        # 4. .txt 파일 열어서 줄 수 세기
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # 비어있지 않은 줄만 카운트 (YOLO 표준)
                line_count = len([line for line in lines if line.strip()])
                
        except Exception as e:
            print(f"파일 읽기 오류: {txt_file.name} - {e}. 문제가 있는 파일로 간주하여 이동합니다.")
            line_count = -1 # 오류가 난 경우도 60이 아닌 것으로 처리

        # 5. 줄 수가 60이 아닌지 확인
        if line_count != required_lines:
            print(f"  [불일치] '{txt_file.name}' 파일에 {line_count}개의 줄이 있습니다. (필요: {required_lines}개)")
            
            image_found_and_moved = False
            
            # 6. 짝이 되는 이미지 파일 찾기 (공통 확장자 .jpg, .png, .jpeg, .bmp)
            base_name = txt_file.stem
            
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.PNG']:
                image_file = source_dir / (base_name + ext)
                
                if image_file.exists():
                    try:
                        # 이미지 파일 이동
                        shutil.move(str(image_file), str(quarantine_dir / image_file.name))
                        image_found_and_moved = True
                        print(f"    -> {image_file.name} 이동 완료.")
                    except Exception as e:
                        print(f"    -> {image_file.name} 이동 실패: {e}")
                    break # 짝을 찾았으므로 중단
            
            # 7. .txt 레이블 파일 이동
            try:
                shutil.move(str(txt_file), str(quarantine_dir / txt_file.name))
                print(f"    -> {txt_file.name} 이동 완료.")
                if image_found_and_moved:
                    moved_pairs += 1
                else:
                    print(f"    -> [경고] {txt_file.name}의 짝이 되는 이미지 파일을 찾지 못했습니다.")
            except Exception as e:
                print(f"    -> {txt_file.name} 이동 실패: {e}")

    # 8. 최종 요약
    print("\n--- 작업 완료 ---")
    print(f"총 {txt_files_checked}개의 .txt 레이블 파일을 확인했습니다.")
    print(f"총 {moved_pairs}개의 (이미지+레이블) 쌍을 '{quarantine_dir.name}' 폴더로 이동했습니다.")

# --- 스크립트 실행 ---

# 1. 대상 폴더 경로를 지정합니다.
# (중요) 백슬래시(\)는 두 개씩(W\) 쓰거나, 경로 앞에 r을 붙여주세요.
TARGET_DIRECTORY = r"C:\Users\seoja\Desktop\new_dipstick\flat_images"

# 2. 함수 실행 (필요한 라인 수는 60)
quarantine_mismatched_labels(TARGET_DIRECTORY, required_lines=60)