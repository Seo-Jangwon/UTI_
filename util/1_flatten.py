import os
import glob
import shutil
from pathlib import Path

# --- 설정 ---

# 1. 원본 데이터가 있는 최상위 폴더
#    (사용자님의 경로에 맞게 수정하세요)
SOURCE_ROOT_DIR = r"C:\Users\seoja\Desktop\new_dipstick\원본_학습데이터들\reconstructed_training_data_copy"

# 2. 모든 이미지를 복사해 넣을 단일 폴더
FLAT_OUTPUT_DIR = r"./flat_images_2222"

# 3. 경로를 파일명으로 인코딩할 때 사용할 구분자
#    (파일명에 절대 없을 것 같은 특별한 문자열)
DELIMITER = "__PATHSEP__"

# 4. 검색할 이미지 확장자
IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG')

# --- 실행 ---

def flatten_directory():
    print(f"=== 1. 폴더 구조 'Flatten' 시작 ===")
    print(f"원본: {SOURCE_ROOT_DIR}")
    print(f"대상: {FLAT_OUTPUT_DIR}")

    # 1. 대상 폴더 생성
    os.makedirs(FLAT_OUTPUT_DIR, exist_ok=True)

    # 2. 모든 하위 폴더에서 이미지 파일 검색
    all_image_paths = []
    for ext in IMAGE_EXTENSIONS:
        search_pattern = os.path.join(SOURCE_ROOT_DIR, '**', ext)
        all_image_paths.extend(glob.glob(search_pattern, recursive=True))

    if not all_image_paths:
        print("❌ 오류: 원본 폴더에서 이미지를 찾을 수 없습니다.")
        return

    print(f"\n총 {len(all_image_paths)}개의 이미지 파일을 찾았습니다.")

    # 3. 파일 복사 및 이름 변경
    copied_count = 0
    for src_path in all_image_paths:
        try:
            # 4. 원본 루트 대비 상대 경로 계산
            #    예: "w Urobilin\Bilirubin\A32\img.jpg"
            relative_path = os.path.relpath(src_path, SOURCE_ROOT_DIR)

            # 5. 새 파일명 생성 (경로 구분자를 DELIMITER로 변경)
            #    예: "w Urobilin__PATHSEP__Bilirubin__PATHSEP__A32__PATHSEP__img.jpg"
            new_name = relative_path.replace(os.sep, DELIMITER)
            
            # 6. 최종 목적지 경로
            dest_path = os.path.join(FLAT_OUTPUT_DIR, new_name)

            # 7. 파일 복사 (shutil.copy2는 메타데이터도 보존)
            if not os.path.exists(dest_path):
                shutil.copy2(src_path, dest_path)
                copied_count += 1
            
        except Exception as e:
            print(f"  ❌ 오류: {src_path} 복사 실패 - {e}")

    print(f"\n✅ 완료: {copied_count}개의 이미지를 {FLAT_OUTPUT_DIR}로 복사했습니다.")
    print("이제 이 'flat_images' 폴더에서 라벨링 작업을 수행하세요.")

if __name__ == "__main__":
    flatten_directory()