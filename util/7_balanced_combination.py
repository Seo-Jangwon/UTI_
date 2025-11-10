"""
개선된 딥스틱 데이터 증강 스크립트
- 조건 파싱 로직 수정
- 바이오마커 매핑 개선
- 조합 생성 로직 단순화
"""

import os
import glob
import cv2
import numpy as np
from PIL import Image
import itertools
from collections import defaultdict
import shutil
import random
from pathlib import Path
import re

# 바이오마커별 패드 위치 매핑 (개선된 버전)
PAD_MAPPING = {
    0: "혈액",      # Hemo, Nonhemo 모두 혈액으로 분류
    1: "빌리루빈",  # Bilirubin
    2: "유로빌리노겐", # Urobilinogen 
    3: "케톤",      # Ketone
    4: "단백질",    # Protein
    5: "아질산염",  # Nitrite
    6: "포도당",    # Glucose
    7: "pH",       # pH
    8: "백혈구",    # Leukocyte (가정)
    9: "비중",      # Specific Gravity (가정)
    10: "아스코르브산", # Ascorbic Acid (가정)
}

# 바이오마커 매핑 (개선된 버전)
BIOMARKER_MAPPING = {
    "Bilirubin": "빌리루빈",
    "Glucose": "포도당", 
    "Hemo": "혈액",
    "Nonhemo": "혈액",  # Hemo와 같은 패드 위치 사용
    "Nitrite": "아질산염",
    "pH": "pH",
    "Protein": "단백질",
}

class ImprovedDataAugmentation:
    def __init__(self, root_dir, output_dir="./balanced_augmented_data_v3"):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.global_combinations = []
        self.stats = {
            'total_combinations': 0,
            'successful_augmentations': 0,
            'failed_augmentations': 0,
            'phone_models': set(),
            'generated_classes': set(),
            'class_usage_counts': defaultdict(int),
            'biomarker_usage_counts': defaultdict(int),
            'condition_counts': defaultdict(int)
        }
        
        os.makedirs(output_dir, exist_ok=True)
    
    def parse_condition_from_filename(self, filename: str) -> str:
        """개선된 조건 파싱 함수"""
        fn = Path(filename).stem
        
        # Shadow 조건 체크
        if 'Shadow' in fn:
            return 'Shadow'
        
        # 정규식을 사용한 더 정확한 파싱
        # 패턴: 클래스명_농도_색온도_밝기
        pattern = r'.*_(\d+)_(\d+)$'
        match = re.search(pattern, fn)
        
        if match:
            temp = match.group(1)      # 색온도
            brightness = match.group(2) # 밝기
            return f"{temp}_{brightness}"
        
        print(f"  ⚠️  [경고] 조건 파싱 실패: {filename}")
        return 'Unknown'
    
    def get_biomarker_from_class_name(self, class_name):
        """개선된 바이오마커 추출 함수"""
        for key, biomarker_name in BIOMARKER_MAPPING.items():
            if class_name.startswith(key):
                return biomarker_name
        
        print(f"  ⚠️  [경고] 바이오마커 매핑 실패: {class_name}")
        return None
    
    def get_pad_number_for_biomarker(self, biomarker_name):
        """바이오마커명으로 패드 번호 반환"""
        for pad_num, marker in PAD_MAPPING.items():
            if marker == biomarker_name:
                return pad_num
        print(f"  ⚠️  [경고] 패드 번호 매핑 실패: {biomarker_name}")
        return None
    
    def collect_data_by_condition_key(self):
        """(폰, 조건)을 key로 모든 파일맵 생성"""
        print("\n📊 (폰, 조건) key로 데이터 수집 중...")
        data_map = defaultdict(dict)
        root_path = Path(self.root_dir)
        
        all_classes = [d.name for d in root_path.iterdir() if d.is_dir()]
        
        for class_name in all_classes:
            class_path = root_path / class_name
            if not class_path.is_dir():
                continue
            
            # 폰 기종 폴더 순회
            for phone_folder in class_path.iterdir():
                phone_model = phone_folder.name
                if not phone_folder.is_dir():
                    continue
                    
                self.stats['phone_models'].add(phone_model)
                
                # 이미지 파일 순회
                image_files = list(phone_folder.glob("*.jpg"))
                
                for img_file in image_files:
                    condition_key_str = self.parse_condition_from_filename(img_file.name)
                    
                    if condition_key_str == 'Unknown':
                        continue
                        
                    self.stats['condition_counts'][condition_key_str] += 1
                        
                    txt_file = img_file.with_suffix('.txt')
                    
                    if txt_file.exists():
                        map_key = (phone_model, condition_key_str)
                        data_map[map_key][class_name] = (str(img_file), str(txt_file))
                    else:
                        print(f"  ⚠️  [경고] 라벨 파일 없음: {img_file.name}")

        print(f"📊 수집 완료:")
        print(f"   - {len(data_map)}개의 고유 (폰, 조건) key")
        print(f"   - {len(self.stats['phone_models'])}개 폰 기종: {self.stats['phone_models']}")
        print(f"   - {len(self.stats['condition_counts'])}개 조건: {list(self.stats['condition_counts'].keys())}")
        
        return data_map
    
    def generate_balanced_combinations(self, all_positive_classes, target_usage_per_class=50):
        """
        필수 조합을 우선적으로 생성하는 균형 조합 생성
        - 우선순위 1: [포도당+혈액+pH+단백질] 4개 조합
        - 우선순위 2: [혈액+pH+아질산염] 3개 조합
        - 나머지: 다양한 크기의 조합 생성
        """
        print(f"\n=== 균형 조합 생성 (목표: 각 클래스 {target_usage_per_class}회) ===")
        
        all_classes = sorted(list(all_positive_classes))
        print(f"전체 양성 클래스 수: {len(all_classes)}")
        
        # 바이오마커별 클래스 그룹화
        biomarker_classes = defaultdict(list)
        for class_name in all_classes:
            biomarker = self.get_biomarker_from_class_name(class_name)
            if biomarker:
                biomarker_classes[biomarker].append(class_name)
        
        print(f"바이오마커별 클래스 분포:")
        for biomarker, classes in biomarker_classes.items():
            print(f"  {biomarker}: {len(classes)}개 클래스")
        
        # 필수 조합 바이오마커 정의 (하드코딩 유지)
        mandatory_biomarkers_B = ['포도당', '혈액', 'pH', '단백질'] 
        mandatory_biomarkers_A = ['혈액', 'pH', '아질산염']
        
        print(f"📌 필수 조합:")
        print(f"   우선순위 A: {mandatory_biomarkers_A}")
        print(f"   우선순위 B: {mandatory_biomarkers_B}")
        
        # 조합 생성 설정
        class_usage = {cls: 0 for cls in all_classes}
        selected_combinations = set()
        
        max_iterations = target_usage_per_class * len(all_classes) * 30
        iteration = 0
        
        print("조합 생성 시작...")
        
        while iteration < max_iterations:
            iteration += 1
            made_mandatory = False

            # --- [필수] 우선순위 조합 생성 ---

            # [우선순위 1] 4개 조합 (포도당+혈액+pH+단백질)
            combo_classes = []
            used_biomarkers = set()
            for bm in mandatory_biomarkers_B:
                # 이 바이오마커에 대해 아직 목표 미달인 클래스 찾기
                found_class = None
                available_classes = biomarker_classes[bm][:]  # 복사본 생성
                random.shuffle(available_classes)  # 매번 섞음
                for cls in available_classes:
                    if class_usage[cls] < target_usage_per_class:
                        found_class = cls
                        break
                if found_class:
                    combo_classes.append(found_class)
                    used_biomarkers.add(bm)
            
            # 4개 바이오마커 모두에서 클래스를 찾았으면 조합 생성
            if len(combo_classes) == len(mandatory_biomarkers_B):
                combo = tuple(sorted(combo_classes))
                if combo not in selected_combinations:
                    selected_combinations.add(combo)
                    for cls in combo:
                        class_usage[cls] += 1
                    made_mandatory = True
                    continue  # 다음 메인 루프 반복

            # [우선순위 2] 3개 조합 (혈액+pH+아질산염)
            combo_classes = []
            used_biomarkers = set()
            for bm in mandatory_biomarkers_A:
                found_class = None
                available_classes = biomarker_classes[bm][:]  # 복사본 생성
                random.shuffle(available_classes)
                for cls in available_classes:
                    if class_usage[cls] < target_usage_per_class:
                        found_class = cls
                        break
                if found_class:
                    combo_classes.append(found_class)
                    used_biomarkers.add(bm)

            if len(combo_classes) == len(mandatory_biomarkers_A):
                combo = tuple(sorted(combo_classes))
                if combo not in selected_combinations:
                    selected_combinations.add(combo)
                    for cls in combo:
                        class_usage[cls] += 1
                    made_mandatory = True
                    continue

            # --- [기존] 우선순위 조합에 실패했을 때만 실행 ---
            if not made_mandatory:
                # 목표 미달 클래스 목록
                deficit_classes = [cls for cls, count in class_usage.items() 
                                  if count < target_usage_per_class]
            
                if not deficit_classes:
                    print(f"  (반복 {iteration}) 모든 클래스가 목표 도달! 루프 종료.")
                    break
                
                # 5, 4, 3개 조합 시도 (큰 것부터)
                for combo_size in [5, 4, 3]: 
                    if len(deficit_classes) < combo_size:
                        continue
                    
                    # 무작위 조합 생성
                    combo = []
                    used_biomarkers = set()
                    
                    random.shuffle(deficit_classes)
                    for cls in deficit_classes:
                        biomarker = self.get_biomarker_from_class_name(cls)
                        # 중복 바이오마커 방지
                        if biomarker and biomarker not in used_biomarkers:
                            combo.append(cls)
                            used_biomarkers.add(biomarker)
                            if len(combo) == combo_size:
                                break
                    
                    # 조합 성공 시
                    if len(combo) == combo_size:
                        combo_tuple = tuple(sorted(combo))
                        if combo_tuple not in selected_combinations:
                            selected_combinations.add(combo_tuple)
                            for cls in combo:
                                class_usage[cls] += 1
                        break  # combo_size 루프 탈출
            
            if iteration % 2000 == 0:
                remaining = sum(1 for count in class_usage.values() 
                              if count < target_usage_per_class)
                print(f"  반복 {iteration}/{max_iterations}: 미달성 클래스 {remaining}개...")
        
        print(f"\n최대 반복 도달. 총 {iteration}회 반복.")
        
        self.global_combinations = list(selected_combinations)
        
        # 통계 출력
        print(f"\n📈 조합 생성 결과:")
        print(f"   - 총 조합 수: {len(self.global_combinations)}")
        
        perfect_count = sum(1 for count in class_usage.values() 
                          if count >= target_usage_per_class)
        print(f"   - 목표 달성 클래스: {perfect_count}/{len(all_classes)} ({perfect_count/len(all_classes)*100:.1f}%)")
        
        # 조합 크기별 분포
        combo_size_dist = defaultdict(int)
        for combo in self.global_combinations:
            combo_size_dist[len(combo)] += 1
        
        print(f"   - 조합 크기별 분포:")
        for size, count in sorted(combo_size_dist.items()):
            print(f"     {size}개 조합: {count}개 ({count/len(self.global_combinations)*100:.1f}%)")
        
        # 필수 조합 통계
        mandatory_A_count = 0
        mandatory_B_count = 0
        for combo in self.global_combinations:
            combo_biomarkers = [self.get_biomarker_from_class_name(cls) for cls in combo]
            combo_biomarkers = [bm for bm in combo_biomarkers if bm]
            
            if set(mandatory_biomarkers_A).issubset(set(combo_biomarkers)):
                mandatory_A_count += 1
            if set(mandatory_biomarkers_B).issubset(set(combo_biomarkers)):
                mandatory_B_count += 1
        
        print(f"   - 필수 조합 A 포함: {mandatory_A_count}개")
        print(f"   - 필수 조합 B 포함: {mandatory_B_count}개")
        
        return self.global_combinations
    
    def imread_unicode(self, img_path):
        """유니코드 경로 이미지 읽기"""
        try:
            pil_image = Image.open(img_path)
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return opencv_image
        except Exception as e:
            print(f"이미지 읽기 오류 {img_path}: {e}")
            return None
    
    def imwrite_unicode(self, img_path, image):
        """유니코드 경로 이미지 저장"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            pil_image.save(img_path)
            return True
        except Exception as e:
            print(f"이미지 저장 오류 {img_path}: {e}")
            return False
    
    def parse_yolo_labels(self, label_path):
        """YOLO 라벨 파일 파싱"""
        labels = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 5:
                            cls_id, x, y, w, h = map(float, parts)
                            labels.append({
                                'class_id': int(cls_id),
                                'x': x, 'y': y, 'w': w, 'h': h
                            })
        except Exception as e:
            print(f"라벨 파싱 오류 {label_path}: {e}")
        return labels
    
    def find_pad_coordinates_with_numbering(self, labels, img_width, img_height):
        """패드들의 좌표를 찾고 번호를 할당"""
        dipsticks = []
        test_pads = []
        
        for label in labels:
            x, y, w, h = label['x'], label['y'], label['w'], label['h']
            x1 = int((x - w / 2) * img_width)
            y1 = int((y - h / 2) * img_height)
            x2 = int((x + w / 2) * img_width)
            y2 = int((y + h / 2) * img_height)
            
            if label['class_id'] == 0:  # dipstick
                dipsticks.append([x1, y1, x2, y2])
            elif label['class_id'] == 1:  # test_pad
                test_pads.append([x1, y1, x2, y2])
        
        if not dipsticks or not test_pads:
            return {}
        
        all_dipsticks_data = {}
        
        for dipstick_idx, dipstick in enumerate(dipsticks):
            dx1, dy1, dx2, dy2 = dipstick
            
            # 이 딥스틱 내부의 패드들 찾기
            pads_in_dipstick = []
            for pad in test_pads:
                px1, py1, px2, py2 = pad
                pad_center_x = (px1 + px2) / 2
                pad_center_y = (py1 + py2) / 2
                
                if (dx1 <= pad_center_x <= dx2 and dy1 <= pad_center_y <= dy2):
                    pads_in_dipstick.append(pad)
            
            # 11개 패드가 있는 딥스틱만 처리
            if len(pads_in_dipstick) == 11:
                dipstick_width = dx2 - dx1
                dipstick_height = dy2 - dy1
                
                # 딥스틱 방향에 따라 정렬
                if dipstick_width > dipstick_height:  # 수평
                    ordered_pads = sorted(pads_in_dipstick, key=lambda p: p[0])
                else:  # 수직
                    ordered_pads = sorted(pads_in_dipstick, key=lambda p: p[1])
                
                # 패드 정렬 방향 보정
                if len(ordered_pads) > 1:
                    center_x0 = (ordered_pads[0][0] + ordered_pads[0][2]) / 2
                    center_x1 = (ordered_pads[1][0] + ordered_pads[1][2]) / 2
                    center_y0 = (ordered_pads[0][1] + ordered_pads[0][3]) / 2
                    center_y1 = (ordered_pads[1][1] + ordered_pads[1][3]) / 2

                    if dipstick_width > dipstick_height:  # 수평
                        if center_x1 < center_x0:
                            ordered_pads.reverse()
                    else:  # 수직
                        if center_y1 < center_y0:
                            ordered_pads.reverse()
                
                # 패드 번호 할당
                pad_coords = {}
                for i, pad in enumerate(ordered_pads):
                    pad_coords[i] = {
                        'bbox': pad,
                        'center': ((pad[0] + pad[2]) // 2, (pad[1] + pad[3]) // 2)
                    }
                
                all_dipsticks_data[dipstick_idx] = pad_coords
        
        return all_dipsticks_data
    
    def extract_pad_region(self, image, bbox):
        """패드 영역 추출"""
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]
    
    def paste_pad_with_target_size(self, target_image, source_pad_crop, target_bbox):
        """패드를 목표 크기로 리사이즈하여 붙여넣기"""
        if source_pad_crop.size == 0:
            return target_image
        
        target_x1, target_y1, target_x2, target_y2 = target_bbox
        target_w = target_x2 - target_x1
        target_h = target_y2 - target_y1
        
        if target_w <= 0 or target_h <= 0:
            return target_image
        
        try:
            resized_source = cv2.resize(source_pad_crop, (target_w, target_h))
            img_h, img_w = target_image.shape[:2]
            
            if (target_x1 >= 0 and target_y1 >= 0 and 
                target_x2 <= img_w and target_y2 <= img_h):
                target_image[target_y1:target_y2, target_x1:target_x2] = resized_source
        except Exception as e:
            print(f"패드 붙여넣기 오류: {e}")
        
        return target_image
    
    def create_augmented_image(self, base_img_path, base_label_path, other_images_data):
        """증강 이미지 생성"""
        try:
            base_image = self.imread_unicode(base_img_path)
            if base_image is None:
                return None, None
            
            result_image = base_image.copy()
            img_h, img_w = base_image.shape[:2]
            
            # Base 이미지의 패드 좌표 찾기
            base_labels = self.parse_yolo_labels(base_label_path)
            base_dipsticks_data = self.find_pad_coordinates_with_numbering(base_labels, img_w, img_h)
            
            if not base_dipsticks_data:
                return None, None
            
            # 다른 이미지들의 패드 데이터 수집
            other_dipsticks_list = []
            for other_img_path, other_label_path, target_biomarker in other_images_data:
                other_image = self.imread_unicode(other_img_path)
                if other_image is None:
                    continue
                
                other_labels = self.parse_yolo_labels(other_label_path)
                other_img_h, other_img_w = other_image.shape[:2]
                other_dipsticks_data = self.find_pad_coordinates_with_numbering(
                    other_labels, other_img_w, other_img_h)
                
                if not other_dipsticks_data:
                    continue
                
                other_dipsticks_list.append((other_image, other_dipsticks_data, target_biomarker))
            
            # 모든 딥스틱에 대해 패드 교체 수행
            for dipstick_idx in base_dipsticks_data.keys():
                base_pad_coords = base_dipsticks_data[dipstick_idx]
                
                for other_image, other_dipsticks_data, target_biomarker in other_dipsticks_list:
                    if dipstick_idx not in other_dipsticks_data:
                        continue
                    
                    source_pad_coords = other_dipsticks_data[dipstick_idx]
                    target_pad_number = self.get_pad_number_for_biomarker(target_biomarker)
                    
                    if target_pad_number is None:
                        continue
                    
                    if target_pad_number not in source_pad_coords:
                        continue
                    
                    if target_pad_number not in base_pad_coords:
                        continue
                    
                    # 소스 패드 추출
                    source_bbox = source_pad_coords[target_pad_number]['bbox']
                    source_pad_crop = self.extract_pad_region(other_image, source_bbox)
                    
                    # 타겟 위치에 붙여넣기
                    target_bbox = base_pad_coords[target_pad_number]['bbox']
                    result_image = self.paste_pad_with_target_size(
                        result_image, source_pad_crop, target_bbox
                    )
            
            return result_image, base_labels
            
        except Exception as e:
            print(f"증강 이미지 생성 오류: {e}")
            return None, None
    
    def save_augmented_data(self, image, labels, class_name, phone_model, filename_stem):
        """증강된 데이터 저장"""
        try:
            class_dir = os.path.join(self.output_dir, class_name, phone_model)
            os.makedirs(class_dir, exist_ok=True)
            
            img_path = os.path.join(class_dir, filename_stem + ".jpg")
            label_path = os.path.join(class_dir, filename_stem + ".txt")
            
            if not self.imwrite_unicode(img_path, image):
                return False
            
            with open(label_path, 'w') as f:
                for label in labels:
                    f.write(f"{label['class_id']} {label['x']} {label['y']} {label['w']} {label['h']}\n")
            
            return True
            
        except Exception as e:
            print(f"저장 오류: {e}")
            return False
    
    def run_augmentation_by_condition(self, target_usage_per_class=50):
        """메인 증강 실행 함수"""
        print("🚀 (폰, 조건) 매칭 기반 증강 시작...")
        
        # 1. 데이터 수집
        data_map = self.collect_data_by_condition_key()
        if not data_map:
            print("❌ 수집된 데이터가 없습니다.")
            return 0

        # 2. 양성 클래스 수집 (Ctrl 제외)
        all_positive_classes = set()
        for class_data in data_map.values():
            all_positive_classes.update(cls for cls in class_data.keys() if cls != 'Ctrl')
        
        if not all_positive_classes:
            print("❌ 양성 클래스를 찾을 수 없습니다.")
            return 0
            
        # 3. 조합 생성
        print(f"🔬 {len(all_positive_classes)}개의 양성 클래스로 조합 생성...")
        self.generate_balanced_combinations(list(all_positive_classes), target_usage_per_class)
        
        if not self.global_combinations:
            print("❌ 생성된 조합이 없습니다.")
            return 0
            
        print(f"✅ {len(self.global_combinations)}개의 조합 레시피 생성 완료.")
        
        # 4. 증강 실행
        total_augmentations = 0
        print("\n=== 🚀 증강 실행 시작 ===")
        
        for combo_idx, combo in enumerate(self.global_combinations):
            if combo_idx % 100 == 0:
                print(f"  진행률: {combo_idx}/{len(self.global_combinations)} ({combo_idx/len(self.global_combinations)*100:.1f}%)")
            
            for (phone_model, condition_str), class_data in data_map.items():
                # Ctrl(베이스) 이미지 확인
                if 'Ctrl' not in class_data:
                    continue
                
                # 조합의 모든 클래스가 현재 조건에 있는지 확인
                if not all(cls in class_data for cls in combo):
                    continue
                
                base_img_path, base_label_path = class_data['Ctrl']
                base_filename_stem = Path(base_img_path).stem
                
                # 소스 이미지 데이터 준비
                other_images_data = []
                valid_combo = True
                
                for other_class in combo:
                    other_img, other_label = class_data[other_class]
                    biomarker = self.get_biomarker_from_class_name(other_class)
                    if biomarker:
                        other_images_data.append((other_img, other_label, biomarker))
                    else:
                        valid_combo = False
                        break
                
                if not valid_combo:
                    continue
                
                # 증강 실행
                augmented_image, augmented_labels = self.create_augmented_image(
                    base_img_path, base_label_path, other_images_data
                )
                
                # 저장
                if augmented_image is not None:
                    new_class_name = "+".join(sorted(combo))
                    
                    if self.save_augmented_data(
                        augmented_image, augmented_labels,
                        new_class_name, phone_model, base_filename_stem
                    ):
                        total_augmentations += 1
                        self.stats['successful_augmentations'] += 1
                        
                        # 통계 업데이트
                        for class_name in combo:
                            self.stats['class_usage_counts'][class_name] += 1
                            biomarker = self.get_biomarker_from_class_name(class_name)
                            if biomarker:
                                self.stats['biomarker_usage_counts'][biomarker] += 1
                else:
                    self.stats['failed_augmentations'] += 1

        print(f"\n🎉 총 {total_augmentations:,}개의 증강 이미지 생성 완료.")
        return total_augmentations
    
    def print_statistics(self):
        """상세 통계 출력"""
        print("\n" + "="*60)
        print("=== 최종 통계 ===")
        print("="*60)
        
        print(f"✅ 성공적인 증강: {self.stats['successful_augmentations']:,}")
        print(f"❌ 실패한 증강: {self.stats['failed_augmentations']:,}")
        print(f"📱 처리된 핸드폰 기종: {len(self.stats['phone_models'])}개")
        print(f"   {sorted(self.stats['phone_models'])}")
        
        print(f"\n🌟 촬영 조건별 분포:")
        for condition, count in sorted(self.stats['condition_counts'].items()):
            print(f"   {condition}: {count:,}개 파일")
        
        print(f"\n🧪 바이오마커별 사용 횟수:")
        for biomarker in sorted(self.stats['biomarker_usage_counts'].keys()):
            count = self.stats['biomarker_usage_counts'][biomarker]
            print(f"   {biomarker:12s}: {count:,}회")
        
        print(f"\n📊 클래스별 사용 횟수 (Top 20):")
        sorted_usage = sorted(self.stats['class_usage_counts'].items(), 
                            key=lambda x: x[1], reverse=True)
        
        for class_name, count in sorted_usage[:20]:
            biomarker = self.get_biomarker_from_class_name(class_name)
            print(f"   {class_name:20s}: {count:5d}회 ({biomarker})")


def main():
    print("\n" + "="*60)
    print("🧬 개선된 딥스틱 데이터 증강 시스템")
    print("="*60)
    
    # 설정
    root_dir = r"C:\Users\seoja\Desktop\new_dipstick\reconstructed_training_data" 
    output_dir = r"C:\Users\seoja\Desktop\new_dipstick\balanced_combination_more"
    
    # 증강 실행
    augmenter = ImprovedDataAugmentation(root_dir, output_dir)
    total_augmented = augmenter.run_augmentation_by_condition(target_usage_per_class=50)
    
    # 통계 출력
    augmenter.print_statistics()
    
    print(f"\n🎯 결과: {total_augmented:,}개 조합 이미지 생성")
    print(f"📁 저장 위치: {output_dir}")
    print("\n✅ 증강 완료! 이제 train/val/test 분할을 진행하세요.")


if __name__ == "__main__":
    main()