#!/usr/bin/env python3
"""
조건 필터링 + 계층화 데이터셋 분할 스크립트
- 원본 데이터는 절대 건드리지 않음 (복사만 함)
- 3000_100 조건과 Shadow 조건을 제외
- 나머지 조건들을 train/val/test에 공정한 비율(70:15:15)로 분배
"""

import os
import shutil
import glob
import random
from pathlib import Path
from collections import defaultdict
import json
from typing import Dict, List, Tuple

# 설정
RANDOM_SEED = 42
# [설정 필요] 분할하려는 원본 데이터 경로
SOURCE_DIR = r"C:\Users\seoja\Desktop\new_dipstick\reconstructed_training_data"
# [설정 필요] 분할된 데이터가 저장될 경로
TARGET_DIR = r"C:\Users\seoja\Desktop\new_dipstick\dataset_filtered_V1" 

# 70% train / 15% val / 15% test
SPLIT_RATIOS = {
    'train': 0.70,
    'val': 0.15, 
    'test': 0.15
}

# [추가] 제외할 조건들 설정
EXCLUDED_CONDITIONS = [
    '3000_100',  # 3000 색온도, 100 밝기 조건
    'Shadow'     # Shadow 조건
]

# 기기 목록 (필요시 수정)
DEVICES = ['A32', 'A32_w', 'S21+', 'S21+_w', 'Z-Flip', 'Z-Flip_w']

def setup_random_seed():
    """재현가능성을 위한 시드 고정"""
    random.seed(RANDOM_SEED)
    print(f"🎲 Random seed 설정: {RANDOM_SEED}")

def get_all_classes() -> List[str]:
    """모든 클래스(농도별) 폴더 목록 반환"""
    source_path = Path(SOURCE_DIR)
    if not source_path.exists():
        raise FileNotFoundError(f"❌ 소스 경로를 찾을 수 없습니다: {SOURCE_DIR}")
    
    classes = [d.name for d in source_path.iterdir() 
              if d.is_dir() and not d.name.startswith('.')]
    classes.sort()
    print(f"📂 발견된 클래스: {len(classes)}개")
    for cls in classes:
        print(f"   - {cls}")
    return classes

def parse_condition_from_filename(filename: str) -> Tuple[str, str]:
    """파일명에서 조건(색온도, 밝기) 추출"""
    fn_stem = Path(filename).stem # 확장자 제거
    
    if 'Shadow' in fn_stem or 'shadow' in fn_stem:
        return ('Shadow', '')
    
    # Bilirubin_0.5_3000_100.jpg 패턴
    parts = fn_stem.split('_')
    if len(parts) >= 3:
        temp = parts[-2]
        brightness = parts[-1]
        if temp.isdigit() and brightness.isdigit():
             return (temp, brightness)
             
    print(f"   ⚠️  [경고] 조건 파싱 실패: {filename}. 'Unknown'으로 처리.")
    return ('Unknown', '')

def is_condition_excluded(condition_key: str) -> bool:
    """조건이 제외 대상인지 확인"""
    return condition_key in EXCLUDED_CONDITIONS

def collect_files_by_condition(classes: List[str]) -> Dict:
    """조건별로 파일들을 수집 (제외 조건 필터링 적용)"""
    condition_files = defaultdict(lambda: defaultdict(list))
    excluded_count = 0
    
    print(f"\n📊 조건별 파일 수집 중... (제외 조건: {EXCLUDED_CONDITIONS})")
    
    for class_name in classes:
        for device in DEVICES:
            device_path = Path(SOURCE_DIR) / class_name / device
            if not device_path.exists():
                continue
                
            image_files = list(device_path.glob("*.jpg"))
            
            for img_file in image_files:
                temp, brightness = parse_condition_from_filename(img_file.name)
                
                # 'Unknown' 조건은 건너뛰기
                if temp == 'Unknown':
                    continue
                    
                condition_key = f"{temp}_{brightness}" if brightness else temp
                
                # [추가] 제외 조건 확인
                if is_condition_excluded(condition_key):
                    excluded_count += 1
                    print(f"   🚫 제외: {img_file.name} (조건: {condition_key})")
                    continue
                
                txt_file = img_file.with_suffix('.txt')
                if txt_file.exists():
                    condition_files[condition_key][class_name].append({
                        'image': img_file,
                        'label': txt_file,
                        'device': device
                    })
    
    print(f"\n📈 조건별 수집 결과:")
    total_files = 0
    for condition, class_data in condition_files.items():
        condition_total = sum(len(files) for files in class_data.values())
        total_files += condition_total
        print(f"   {condition:12s}: {condition_total:5d}개 파일")
    
    print(f"🎯 총 수집된 파일 쌍: {total_files}개")
    print(f"🚫 제외된 파일 쌍: {excluded_count}개")
    return condition_files

def create_condition_based_split(condition_files: Dict) -> Dict:
    """
    조건별 '계층화 분할' 수행 (필터링된 조건들만 사용)
    - 모든 조건을 train/val/test에 공정한 비율로 분배
    """
    print(f"\n🔄 조건별 계층화 분할 수행 (비율: {SPLIT_RATIOS})")
    
    split_data = {
        'train': defaultdict(list),
        'val': defaultdict(list), 
        'test': defaultdict(list)
    }
    
    all_conditions = list(condition_files.keys())
    print(f"📊 총 {len(all_conditions)}개의 조건을 분할합니다:")
    for condition in sorted(all_conditions):
        print(f"   - {condition}")
    
    # 모든 조건에 대해 루프
    for condition in all_conditions:
        if condition not in condition_files:
            continue
            
        print(f"\n📝 조건 '{condition}' 분할 중...")
        
        # 각 조건 *내부*에서 클래스별로 분할
        for class_name, files in condition_files[condition].items():
            
            files_copy = files.copy()
            random.shuffle(files_copy)
            
            n_total = len(files_copy)
            if n_total == 0:
                continue

            # train, val, test 인덱스 계산
            idx_train_end = int(n_total * SPLIT_RATIOS['train'])
            idx_val_end = idx_train_end + int(n_total * SPLIT_RATIOS['val'])
            
            # 최소 1개씩은 보장하려고 시도 (데이터가 충분할 때만)
            if n_total >= 3:
                if idx_train_end == n_total:  # train이 100%인 경우
                    idx_train_end = max(1, n_total - 2)
                if idx_val_end == idx_train_end:  # val이 0%인 경우
                    idx_val_end = min(n_total - 1, idx_train_end + 1)
            
            # train / val / test로 분할
            train_files = files_copy[:idx_train_end]
            val_files = files_copy[idx_train_end:idx_val_end]
            test_files = files_copy[idx_val_end:]
            
            # 분할 결과 추가
            split_data['train'][class_name].extend(train_files)
            split_data['val'][class_name].extend(val_files)
            split_data['test'][class_name].extend(test_files)
            
            print(f"   {class_name:15s}: {len(train_files):3d} train, {len(val_files):3d} val, {len(test_files):3d} test")
    
    # 분할 결과 통계
    print(f"\n📊 최종 분할 결과:")
    total_all = 0
    for split_name, class_data in split_data.items():
        total = sum(len(files) for files in class_data.values())
        total_all += total
        actual_ratio = total / total_all if total_all > 0 else 0
        print(f"   {split_name.upper():5s}: {total:6d}개 파일 ({actual_ratio:.1%})")
    
    print(f"   --------------------")
    print(f"   TOTAL: {total_all:6d}개 파일")
    
    return split_data

def copy_files_to_target(split_data: Dict):
    """분할된 파일들을 타겟 디렉토리로 복사"""
    print(f"\n📁 타겟 디렉토리 생성: {TARGET_DIR}")
    
    target_path = Path(TARGET_DIR)
    if target_path.exists():
        print("⚠️  타겟 디렉토리가 이미 존재합니다. 덮어쓰시겠습니까? (y/N): ", end="")
        response = input().strip().lower()
        if response != 'y':
            print("❌ 작업을 취소했습니다.")
            return False
        print("   기존 디렉토리를 삭제합니다...")
        shutil.rmtree(target_path)
    
    target_path.mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    # 분할별 디렉토리 구조 생성 및 파일 복사
    for split_name, class_data in split_data.items():
        print(f"\n📋 {split_name.upper()} 세트 생성 중...")
        split_total = 0
        
        for class_name, files in class_data.items():
            if not files:
                continue
                
            for file_info in files:
                # 타겟 경로 구성: TARGET_DIR/train/Bilirubin_0.5/A32/
                target_class_device_dir = target_path / split_name / class_name / file_info['device']
                target_class_device_dir.mkdir(parents=True, exist_ok=True)
                
                # 이미지 파일 복사
                target_img = target_class_device_dir / file_info['image'].name
                shutil.copy2(file_info['image'], target_img)
                
                # 라벨 파일 복사  
                target_label = target_class_device_dir / file_info['label'].name
                shutil.copy2(file_info['label'], target_label)
                split_total += 1
        
        print(f"   ✅ {split_total}개 파일 쌍 복사 완료")
        total_copied += split_total
    
    print(f"\n✨ 총 {total_copied}개의 파일 쌍이 새 위치로 복사되었습니다.")
    return True

def create_split_report(split_data: Dict):
    """분할 결과 리포트 생성"""
    report_path = Path(TARGET_DIR) / "split_report.json"
    
    report = {
        "split_method": "condition_based_stratified_filtered",
        "split_ratios_target": SPLIT_RATIOS,
        "random_seed": RANDOM_SEED,
        "excluded_conditions": EXCLUDED_CONDITIONS,
        "statistics": {}
    }
    
    grand_total = 0
    
    for split_name, class_data in split_data.items():
        split_stats = {}
        split_total_files = 0
        
        for class_name, files in class_data.items():
            device_counts = defaultdict(int)
            condition_counts = defaultdict(int)
            
            for file_info in files:
                device_counts[file_info['device']] += 1
                temp, brightness = parse_condition_from_filename(file_info['image'].name)
                condition = f"{temp}_{brightness}" if brightness else temp
                condition_counts[condition] += 1
            
            split_stats[class_name] = {
                "total_files": len(files),
                "devices": dict(device_counts),
                "conditions": dict(condition_counts)
            }
            split_total_files += len(files)
        
        report["statistics"][split_name] = {
            "total_files": split_total_files,
            "by_class": split_stats
        }
        grand_total += split_total_files
    
    report["grand_total_files"] = grand_total
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📊 분할 리포트 저장: {report_path}")

def main():
    """메인 실행 함수"""
    print("🚀 조건 필터링 + 계층화 분할 시작!")
    print("=" * 60)
    print(f"🚫 제외할 조건: {EXCLUDED_CONDITIONS}")
    print("=" * 60)
    
    try:
        # 1. 초기 설정
        setup_random_seed()
        
        # 2. 클래스 수집
        classes = get_all_classes()
        
        # 3. 조건별 파일 수집 (필터링 적용)
        condition_files = collect_files_by_condition(classes)
        
        if not condition_files:
            print("❌ 필터링 후 사용 가능한 데이터가 없습니다!")
            return False
        
        # 4. 계층화 분할 수행
        split_data = create_condition_based_split(condition_files)
        
        # 5. 파일 복사
        success = copy_files_to_target(split_data)
        if not success:
            return False
            
        # 6. 리포트 생성
        create_split_report(split_data)
        
        print("\n🎉 조건 필터링 + 계층화 분할 완료!")
        print("=" * 60)
        print("📂 결과물 위치:")
        print(f"   📁 {TARGET_DIR}")
        print(f"   📊 {Path(TARGET_DIR) / 'split_report.json'}")
        print("\n💡 필터링 결과:")
        print(f"   - 제외된 조건: {EXCLUDED_CONDITIONS}")
        print("   - 나머지 조건들이 train/val/test에 공정하게 분배됨")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🏆 모든 작업이 성공적으로 완료되었습니다!")
    else:
        print("\n💥 작업 중 오류가 발생했거나 취소되었습니다.")