import os
import glob
import shutil
from pathlib import Path

# --- 설정 ---

# 1. 라벨링이 완료된 파일들이 있는 단일 폴더
FLAT_SOURCE_DIR = r"./flat_images"

# 2. 원본 구조로 복구할 최상위 폴더
RESTORE_ROOT_DIR = r"./restored_dataset" 

# 3. Flatten 스크립트에서 사용했던 동일한 구분자
DELIMITER = "__PATHSEP__"

# --- 실행 ---

def restore_directory():
    print(f"=== 2. 폴더 구조 'Restore' 시작 ===")
    print(f"원본: {FLAT_SOURCE_DIR}")
    print(f"대상: {RESTORE_ROOT_DIR}")

    # 1. 복구할 파일들 검색 (모든 파일 대상)
    flat_files = glob.glob(os.path.join(FLAT_SOURCE_DIR, f"*{DELIMITER}*.*"))

    if not flat_files:
        print(f"❌ 오류: {FLAT_SOURCE_DIR}에서 {DELIMITER}가 포함된 파일을 찾을 수 없습니다.")
        return

    print(f"\n총 {len(flat_files)}개의 파일(이미지+라벨)을 복구합니다.")
    
    # 2. 파일 이동 및 경로 복원
    moved_count = 0
    for flat_path in flat_files:
        try:
            filename = os.path.basename(flat_path)
            
            # 3. 파일명에서 원본 경로 복원
            #    예: "w Urobilin__PATHSEP__A32__PATHSEP__img.txt"
            #    -> "w Urobilin/A32/img.txt"
            original_relative_path = filename.replace(DELIMITER, os.sep)
            
            # 4. 최종 목적지 경로 생성
            dest_path = os.path.join(RESTORE_ROOT_DIR, original_relative_path)
            
            # 5. 목적지 폴더 생성
            dest_dir = os.path.dirname(dest_path)
            os.makedirs(dest_dir, exist_ok=True)
            
            # 6. 파일 이동 (복구가 아닌 이동)
            shutil.move(flat_path, dest_path)
            moved_count += 1
            
        except Exception as e:
            print(f"  ❌ 오류: {filename} 복구 실패 - {e}")

    print(f"\n✅ 완료: {moved_count}개의 파일을 {RESTORE_ROOT_DIR}로 복구했습니다.")
    print("이제 'flat_images' 폴더는 비어있거나, 처리되지 않은 파일만 남아있어야 합니다.")

if __name__ == "__main__":
    restore_directory()