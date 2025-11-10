import os
import glob
import shutil
from pathlib import Path

# --- 설정 ---

# 1. 작업할 소스 폴더 (라벨링된 파일과 안된 파일이 섞여있는 곳)
SOURCE_DIR = r"C:\Users\seoja\Desktop\new_dipstick\욜로학습용폴더\inference_images"

# 2. 이미지 파일 목적지 폴더 (YOLO가 읽을 곳)
IMG_DEST_DIR = r"C:\Users\seoja\Desktop\new_dipstick\욜로학습용폴더\temp_images"

# 3. 라벨 파일 목적지 폴더 (YOLO가 읽을 곳)
LABEL_DEST_DIR = r"C:\Users\seoja\Desktop\new_dipstick\욜로학습용폴더\temp_labels"

# 4. 이미지로 간주할 확장자
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

# --- 실행 ---

def sort_labeled_files():
    print(f"=== 라벨링된 파일 분류 작업 시작 ===")
    print(f"소스 폴더: {SOURCE_DIR}")

    # 1. 목적지 폴더 생성
    os.makedirs(IMG_DEST_DIR, exist_ok=True)
    os.makedirs(LABEL_DEST_DIR, exist_ok=True)

    # 2. 소스 폴더 내의 모든 .txt 파일의 '파일명' (확장자 제외)을 set으로 만듭니다.
    #    예: { "file1__PATHSEP__A", "file2__PATHSEP__B", ... }
    label_files = glob.glob(os.path.join(SOURCE_DIR, "*.txt"))
    label_basenames = {Path(f).stem for f in label_files}
    
    if not label_basenames:
        print("⚠️  경고: 소스 폴더에 .txt 라벨 파일이 없습니다. 작업을 중단합니다.")
        return

    print(f"총 {len(label_basenames)}개의 .txt 라벨 파일을 찾았습니다.")

    # 3. 소스 폴더 내의 모든 이미지 파일을 찾습니다.
    image_files_to_check = []
    for ext in IMAGE_EXTENSIONS:
        image_files_to_check.extend(
            glob.glob(os.path.join(SOURCE_DIR, f"*{ext}"))
        )
        # 대소문자 구분 (예: .JPG)
        image_files_to_check.extend(
            glob.glob(os.path.join(SOURCE_DIR, f"*{ext.upper()}"))
        )

    print(f"매칭 여부를 확인할 총 {len(image_files_to_check)}개의 이미지 파일을 찾았습니다.")

    # 4. 이미지 파일을 하나씩 확인하며 .txt 짝이 있는지 검사
    moved_pairs_count = 0
    for img_path in image_files_to_check:
        img_basename = Path(img_path).stem # 예: "file1__PATHSEP__A"

        # 5. 이미지의 파일명(img_basename)이 아까 만든 .txt 파일명 set에 있는지 확인
        if img_basename in label_basenames:
            # --- 짝을 찾음! ---
            
            # 6. 원본 .txt 파일의 전체 경로를 다시 만듭니다.
            txt_path = os.path.join(SOURCE_DIR, img_basename + ".txt")

            # 7. 목적지 경로를 설정합니다.
            img_dest_path = os.path.join(IMG_DEST_DIR, os.path.basename(img_path))
            txt_dest_path = os.path.join(LABEL_DEST_DIR, os.path.basename(txt_path))

            # 8. 파일 "이동" (shutil.move)
            try:
                shutil.move(img_path, img_dest_path)
                shutil.move(txt_path, txt_dest_path)
                print(f"  -> [매칭 성공] {img_basename} (이미지 + 라벨 이동 완료)")
                moved_pairs_count += 1
            except OSError as e:
                print(f"  -> [오류] {img_basename} 파일 이동 실패: {e}")
            
    print("-" * 30)
    print(f"✅ 작업 완료: 총 {moved_pairs_count}쌍의 파일을 이동했습니다.")
    print(f"  - 이미지 -> {IMG_DEST_DIR}")
    print(f"  - 라벨   -> {LABEL_DEST_DIR}")
    print(f"\n이제 '{SOURCE_DIR}' 폴더에는 라벨링 되지 않은 파일들만 남아있습니다.")

if __name__ == "__main__":
    sort_labeled_files()