import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

def get_labels_from_txt(label_path, img_w, img_h):
    """YOLO .txt 라벨을 읽어 [x1, y1, x2, y2, cls_id] 리스트로 반환"""
    if not os.path.exists(label_path):
        return []
        
    with open(label_path, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    
    labels = []
    for line in lines:
        if not line:
            continue
        try:
            cls_id_float, x, y, w_norm, h_norm = map(float, line.split())
            cls_id = int(cls_id_float)
            
            x1 = (x - w_norm / 2) * img_w
            y1 = (y - h_norm / 2) * img_h
            x2 = (x + w_norm / 2) * img_w
            y2 = (y + h_norm / 2) * img_h
            
            labels.append([x1, y1, x2, y2, cls_id])
        except Exception as e:
            print(f"  [경고] 라벨 파싱 오류 {label_path}: {e}")
            continue
            
    return np.array(labels)

def number_sensor_pads(input_dir, output_dir):
    """
    [V4] 딥스틱(0) 내부의 11개 패드(1)를 찾고,
    cv2.fitLine을 사용해 "방향 벡터"를 찾아 정렬합니다.
    (회전에 100% 강인함)
    """
    print(f"이미지 번호 매기기 시작 (V4 - fitLine)...")
    print(f"입력 경로: {input_dir}")
    print(f"출력 경로: {output_dir} (모든 파일을 이 폴더에 저장)")
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_image_paths = []
    for root, _, files in os.walk(input_dir):
        for img_file in files:
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                all_image_paths.append(os.path.join(root, img_file))
                
    print(f"총 {len(all_image_paths)}개 이미지 파일 탐색 완료. 번호 매기기 시작...")
    
    progress_bar = tqdm(all_image_paths, desc="전체 진행률")
    for img_path in progress_bar:
        label_path = os.path.splitext(img_path)[0] + ".txt"
        
        if not os.path.exists(label_path):
            continue
            
        try:
            image_pil = Image.open(img_path).convert("RGB")
            image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            output_image = image.copy()
            img_h, img_w = image.shape[:2]
            
            labels = get_labels_from_txt(label_path, img_w, img_h)
            
            if len(labels) == 0:
                continue

            # 1. 클래스 ID로 딥스틱(0)과 센서(1) 분리
            dipstick_boxes = labels[labels[:, 4] == 0] # cls_id == 0
            sensor_boxes = labels[labels[:, 4] == 1]   # cls_id == 1
            
            if len(dipstick_boxes) == 0 or len(sensor_boxes) < 11:
                continue
                
            dipstick_boxes = dipstick_boxes[np.lexsort((dipstick_boxes[:,1], dipstick_boxes[:,0]))]

            found_pads_on_image = False
            
            # 2. 각 딥스틱(0번 라벨)별로 루프
            for kit in dipstick_boxes:
                x1, y1, x2, y2, _ = kit
                
                # 이 딥스틱 내부에 속한 센서(1번 라벨)들만 필터링
                mask = ((sensor_boxes[:, 0] > x1) & (sensor_boxes[:, 0] < x2) &
                        (sensor_boxes[:, 1] > y1) & (sensor_boxes[:, 1] < y2))
                group = sensor_boxes[mask]
                
                # 3. 정확히 11개인 그룹을 찾으면 번호 매기기
                if len(group) == 11:
                    found_pads_on_image = True
                    
                    # =======================================================
                    # [!!! V4 핵심 로직: "선 긋기" (Line Fitting) !!!]
                    # =======================================================
                    
                    # 1. 11개 패드의 중심점(centers)을 계산
                    centers = np.array([
                        ((p[0] + p[2]) / 2, (p[1] + p[3]) / 2) for p in group
                    ], dtype=np.float32)
                    
                    # 2. 11개 중심점을 가장 잘 통과하는 "선 (방향 벡터)"을 계산
                    #    vx, vy: 방향 벡터 (예: [0.99, 0.01] -> 거의 가로)
                    #    x0, y0: 선이 통과하는 원점
                    [vx, vy, x0, y0] = cv2.fitLine(centers, cv2.DIST_L2, 0, 0.01, 0.01)
                    direction_vector = np.array([vx[0], vy[0]])
                    line_origin = np.array([x0[0], y0[0]])

                    # 3. 11개 중심점을 이 "선"에 투영(dot product)시켜 1차원 거리값 계산
                    projected_distances = []
                    for center in centers:
                        distance = np.dot(center - line_origin, direction_vector)
                        projected_distances.append(distance)
                    
                    # 4. 이 "거리값"을 기준으로 패드들을 정렬
                    sort_indices = np.argsort(projected_distances)
                    sorted_group = group[sort_indices]
                    # =======================================================
                    
                    for index, sensor in enumerate(sorted_group):
                        px1, py1, px2, py2, _ = map(int, sensor)
                        
                        center_x = (px1 + px2) // 2
                        center_y = (py1 + py2) // 2
                        
                        text = str(index)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = (px2 - px1) / 50.0 
                        thickness = max(1, int(font_scale * 2))
                        
                        cv2.putText(output_image, text, (center_x - 5, center_y + 5),
                                    font, font_scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
                        cv2.putText(output_image, text, (center_x - 5, center_y + 5),
                                    font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

            # 4. 'flatten'하여 단일 폴더에 저장
            if found_pads_on_image:
                relative_path = os.path.relpath(img_path, input_dir)
                new_filename = relative_path.replace(os.sep, '_')
                output_path = os.path.join(output_dir, new_filename)
                
                output_pil = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
                output_pil.save(output_path)
        
        except Exception as e:
            print(f"  [오류] 이미지 처리 실패 {img_path}: {e}")

    print("\n모든 작업 완료. (V4 - Line Fitting)")

# =============================================================================
# [메인 실행 블록]
# =============================================================================
if __name__ == "__main__":
    
    print("스크립트 실행 전, 터미널에 'pip install opencv-python-headless numpy tqdm pillow'를 입력하세요.")
    print("-" * 50)
    
    # 1. (필수) 경로 설정
    INPUT_IMAGE_DIRECTORY = r"C:\Users\seoja\Desktop\new_dipstick\다중_클래스_분류_학습데이터들\dataset_mcc" 
    OUTPUT_FLATTEN_DIRECTORY = "./output_numbered_images_FLAT_V4_LineFit" # 새 폴더
    
    # 2. 스크립트 실행
    number_sensor_pads(INPUT_IMAGE_DIRECTORY, OUTPUT_FLATTEN_DIRECTORY)