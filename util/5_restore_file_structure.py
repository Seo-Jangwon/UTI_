import os
import shutil
from pathlib import Path
from tqdm import tqdm # 진행률 표시를 위한 라이브러리

def restore_file_structure(source_dir_str, separator="__PATHSEP__"):
    """
    Flatten된 파일명을 파싱하여 원래의 폴더 구조로 복원합니다.
    
    로직:
    1. 파일명에서 마지막 부분 (예: 'Protein_10_...')을 파싱 -> 최상위 폴더 (예: 'Protein_10')
    2. 파일명에서 끝에서 두 번째 부분 (예: 'S21+')을 파싱 -> 하위 폴더 (예: 'S21+')
    3. 파일명에서 첫 번째 부분 (예: 'w Urobilin')을 확인 -> 하위 폴더명 수정 (예: 'S21+_w')
    """
    
    source_dir = Path(source_dir_str)
    # 복원된 파일이 저장될 새 폴더 (flat_images 폴더와 같은 위치에 생성)
    target_root = source_dir.parent / "Restored_Structure"
    
    if not source_dir.exists():
        print(f"오류: 소스 폴더를 찾을 수 없습니다: {source_dir}")
        return

    target_root.mkdir(exist_ok=True)
    print(f"소스 폴더: {source_dir}")
    print(f"타겟 폴더: {target_root}\n")

    # .jpg 파일을 기준으로 루프를 돌고, 짝이 되는 .txt도 함께 처리
    jpg_files = list(source_dir.glob("*.jpg"))
    if not jpg_files:
        print("경고: 처리할 .jpg 파일을 찾을 수 없습니다.")
        return

    files_moved = 0
    errors = 0

    for jpg_path in tqdm(jpg_files, desc="파일 재구성 중"):
        base_name = jpg_path.stem # 확장자를 제외한 파일명
        txt_path = jpg_path.with_suffix(".txt")

        if not txt_path.exists():
            print(f"  [경고] {base_name}.jpg의 짝이 되는 .txt 파일이 없어 건너뜁니다.")
            errors += 1
            continue
            
        try:
            parts = base_name.split(separator)
            
            # parts 구조: [접두사, 중간경로1, ..., 기기명, 실제파일명]
            if len(parts) < 3: # 최소 [접두사, 기기명, 실제파일명]
                print(f"  [오류] {base_name}: 파일명 구조가 예상과 다릅니다. (구분자 부족)")
                errors += 1
                continue

            # 규칙 1 & 3 적용 (최상위 폴더 + 실제 파일명)
            filename_part = parts[-1]
            filename_splits = filename_part.split('_')
            
            if len(filename_splits) < 2:
                print(f"  [오류] {filename_part}: 파일명에서 '_'를 기준으로 폴더명을 추출할 수 없습니다.")
                errors += 1
                continue
                
            # 예: 'Protein_10'
            top_level_folder = f"{filename_splits[0]}_{filename_splits[1]}"
            
            # 규칙 2 적용 (기기명 폴더)
            device_folder_raw = parts[-2] # 예: 'S21+'
            prefix = parts[0] # 예: 'w Urobilin'
            
            device_folder_final = device_folder_raw # 기본값
            
            if prefix.startswith("w Urobilin"):
                # "w Urobilin..." -> S21+ becomes S21+_w
                device_folder_final = device_folder_raw + "_w"
            elif prefix.startswith("wo Urobilin"):
                # "wo Urobilin..." -> S21+ remains S21+
                device_folder_final = device_folder_raw
            # else: 그 외의 접두사는 기기명 원본을 그대로 사용

            # 4. 최종 경로 생성 및 파일 이동
            target_dir = target_root / top_level_folder / device_folder_final
            target_dir.mkdir(parents=True, exist_ok=True) # 폴더 생성 (하위 폴더까지)
            
            # 최종 파일명은 'Protein_10_4000_100.jpg'가 됩니다.
            target_jpg_path = target_dir / (filename_part + ".jpg")
            target_txt_path = target_dir / (filename_part + ".txt")

            # 파일 이동
            shutil.move(str(jpg_path), str(target_jpg_path))
            shutil.move(str(txt_path), str(target_txt_path))
            
            files_moved += 1

        except Exception as e:
            print(f"  [오류] {jpg_path.name} 처리 중 예외 발생: {e}")
            errors += 1

    print("\n--- 작업 완료 ---")
    print(f"총 {files_moved}개의 (이미지+레이블) 쌍을 '{target_root.name}' 폴더로 이동했습니다.")
    if errors > 0:
        print(f"총 {errors}개의 파일 처리 중 오류가 발생했습니다.")

# --- 스크립트 실행 ---

# 1. 'flatten'된 파일들이 모여있는 소스 폴더 경로
# (중요) Windows 경로는 r"..." 또는 "C:\\Users\\..." 형태로 입력해야 합니다.
FLAT_SOURCE_DIRECTORY = r"C:\Users\seoja\Desktop\new_dipstick\flat_images"

# 2. 함수 실행
restore_file_structure(FLAT_SOURCE_DIRECTORY)