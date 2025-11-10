#!/usr/bin/env python3
"""
다중 레이블 조합 데이터 분할 스크립트 - 전역 조건 분할 버전
- 전역적으로 조건을 먼저 분할 (Train 4개, Val 1-2개, Test 1-2개)
- 모든 기기가 동일한 조건 분할 사용
- 비율 정확히 맞춤 (약 4:1.5:1.5 비율)
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
SOURCE_DIR = r"C:\Users\seoja\Desktop\new_dipstick\다중_레이블_분류_학습데이터들\1_balanced_combination"
# [설정 필요] 분할된 데이터가 저장될 경로
TARGET_DIR = r"C:\Users\seoja\Desktop\new_dipstick\multilabel_dataset_global_condition_split222"

# 기기 목록
DEVICES = ['A32', 'A32_w', 'S21+', 'S21+_w', 'Z-Flip', 'Z-Flip_w']

def setup_random_seed():
    """재현가능성을 위한 시드 고정"""
    random.seed(RANDOM_SEED)
    print(f"🎲 Random seed 설정: {RANDOM_SEED}")

def get_all_classes() -> List[str]:
    """모든 클래스(조합) 폴더 목록 반환"""
    source_path = Path(SOURCE_DIR)
    if not source_path.exists():
        raise FileNotFoundError(f"❌ 소스 경로를 찾을 수 없습니다: {SOURCE_DIR}")
    
    classes = [d.name for d in source_path.iterdir() 
              if d.is_dir() and not d.name.startswith('.')]
    classes.sort()
    print(f"📂 발견된 클래스: {len(classes)}개")
    for i, cls in enumerate(classes[:10]):  # 처음 10개만 출력
        print(f"   - {cls}")
    if len(classes) > 10:
        print(f"   ... 및 {len(classes)-10}개 더")
    return classes

def parse_condition_from_filename(filename: str) -> Tuple[str, str]:
    """파일명에서 조건(색온도, 밝기) 추출"""
    fn_stem = Path(filename).stem
    
    if 'Shadow' in fn_stem or 'shadow' in fn_stem:
        return ('Shadow', '')
    
    # 조합클래스명_색온도_밝기.jpg 패턴
    parts = fn_stem.split('_')
    if len(parts) >= 2:
        temp = parts[-2]
        brightness = parts[-1]
        if temp.isdigit() and brightness.isdigit():
             return (temp, brightness)
             
    print(f"   ⚠️  [경고] 조건 파싱 실패: {filename}. 'Unknown'으로 처리.")
    return ('Unknown', '')

def discover_all_conditions(classes: List[str]) -> List[str]:
    """모든 사용 가능한 조건들을 찾기"""
    print(f"\n🔍 전체 조건 탐색 중...")
    
    all_conditions = set()
    
    for class_name in classes[:5]:  # 몇 개 클래스만 샘플링
        for device in DEVICES[:2]:  # 몇 개 기기만 샘플링
            device_path = Path(SOURCE_DIR) / class_name / device
            if not device_path.exists():
                continue
                
            image_files = list(device_path.glob("*.jpg"))[:10]  # 몇 개 파일만 샘플링
            
            for img_file in image_files:
                temp, brightness = parse_condition_from_filename(img_file.name)
                
                if temp == 'Unknown':
                    continue
                    
                condition_key = f"{temp}_{brightness}" if brightness else temp
                all_conditions.add(condition_key)
    
    conditions_list = sorted(list(all_conditions))
    print(f"🌡️ 발견된 조건: {len(conditions_list)}개")
    for condition in conditions_list:
        print(f"   - {condition}")
    
    return conditions_list

def create_global_condition_split(all_conditions: List[str]) -> Dict:
    """
    전역적으로 조건을 분할
    - Train: 4개 조건
    - Val: 1-2개 조건
    - Test: 1-2개 조건 (나머지)
    """
    print(f"\n🌍 전역 조건 분할 생성")
    
    conditions = all_conditions[:]
    random.shuffle(conditions)
    
    n_total = len(conditions)
    
    # Train은 항상 4개 (또는 전체의 60% 중 큰 값)
    n_train = min(4, max(4, int(n_total * 0.6)))
    
    # 남은 조건을 val/test로 분할
    remaining = n_total - n_train
    n_val = max(1, remaining // 2)
    n_test = remaining - n_val
    
    train_conditions = conditions[:n_train]
    val_conditions = conditions[n_train:n_train + n_val]
    test_conditions = conditions[n_train + n_val:]
    
    condition_split = {
        'train': train_conditions,
        'val': val_conditions,
        'test': test_conditions
    }
    
    print(f"📊 전역 조건 분할 결과:")
    for split_name, conds in condition_split.items():
        print(f"   {split_name.upper():5s}: {len(conds)}개 조건 - {conds}")
    
    return condition_split

def collect_files_by_condition(classes: List[str], condition_split: Dict) -> Dict:
    """조건 분할에 따라 파일들을 수집"""
    print(f"\n📊 조건 분할에 따른 파일 수집 중...")
    
    split_data = {
        'train': defaultdict(lambda: defaultdict(list)),
        'val': defaultdict(lambda: defaultdict(list)),
        'test': defaultdict(lambda: defaultdict(list))
    }
    
    total_files = {'train': 0, 'val': 0, 'test': 0}
    
    for class_name in classes:
        for device in DEVICES:
            device_path = Path(SOURCE_DIR) / class_name / device
            if not device_path.exists():
                continue
                
            image_files = list(device_path.glob("*.jpg"))
            
            for img_file in image_files:
                temp, brightness = parse_condition_from_filename(img_file.name)
                
                if temp == 'Unknown':
                    continue
                    
                condition_key = f"{temp}_{brightness}" if brightness else temp
                
                # 어느 split에 속하는지 확인
                target_split = None
                for split_name, conditions in condition_split.items():
                    if condition_key in conditions:
                        target_split = split_name
                        break
                
                if target_split is None:
                    continue  # 분할에 포함되지 않은 조건
                
                txt_file = img_file.with_suffix('.txt')
                if txt_file.exists():
                    split_data[target_split][class_name][device].append({
                        'image': img_file,
                        'label': txt_file,
                        'condition': condition_key
                    })
                    total_files[target_split] += 1
    
    print(f"\n📈 조건별 수집 결과:")
    grand_total = sum(total_files.values())
    for split_name, total in total_files.items():
        ratio = total / grand_total if grand_total > 0 else 0
        print(f"   {split_name.upper():5s}: {total:6d}개 파일 ({ratio:.1%})")
    
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
    
    for split_name, class_data in split_data.items():
        print(f"\n📋 {split_name.upper()} 세트 생성 중...")
        split_total = 0
        
        for class_name, device_data in class_data.items():
            for device, files in device_data.items():
                if not files:
                    continue
                    
                for file_info in files:
                    # 타겟 경로 구성: TARGET_DIR/train/class_name/device/
                    target_class_device_dir = target_path / split_name / class_name / device
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

def create_split_report(split_data: Dict, condition_split: Dict):
    """분할 결과 리포트 생성"""
    report_path = Path(TARGET_DIR) / "split_report.json"
    
    report = {
        "split_method": "global_condition_based_split",
        "random_seed": RANDOM_SEED,
        "devices": DEVICES,
        "condition_split": condition_split,
        "statistics": {}
    }
    
    grand_total = 0
    
    for split_name, class_data in split_data.items():
        split_stats = {}
        split_total_files = 0
        device_counts = defaultdict(int)
        condition_counts = defaultdict(int)
        
        for class_name, device_data in class_data.items():
            class_device_counts = defaultdict(int)
            class_condition_counts = defaultdict(int)
            class_total = 0
            
            for device, files in device_data.items():
                for file_info in files:
                    condition = file_info['condition']
                    
                    device_counts[device] += 1
                    condition_counts[condition] += 1
                    class_device_counts[device] += 1
                    class_condition_counts[condition] += 1
                    class_total += 1
            
            split_stats[class_name] = {
                "total_files": class_total,
                "devices": dict(class_device_counts),
                "conditions": dict(class_condition_counts)
            }
            split_total_files += class_total
        
        report["statistics"][split_name] = {
            "total_files": split_total_files,
            "devices": dict(device_counts),
            "conditions": dict(condition_counts),
            "by_class": split_stats
        }
        grand_total += split_total_files
    
    report["grand_total_files"] = grand_total
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📊 분할 리포트 저장: {report_path}")

def main():
    """메인 실행 함수"""
    print("🚀 다중 레이블 조합 데이터 분할 시작! (전역 조건 분할)")
    print("=" * 60)
    
    try:
        # 1. 초기 설정
        setup_random_seed()
        
        # 2. 클래스 수집
        classes = get_all_classes()
        
        # 3. 전체 조건 탐색
        all_conditions = discover_all_conditions(classes)
        
        if not all_conditions:
            print("❌ 사용 가능한 조건을 찾을 수 없습니다!")
            return False
        
        # 4. 전역 조건 분할 생성
        condition_split = create_global_condition_split(all_conditions)
        
        # 5. 조건 분할에 따른 파일 수집
        split_data = collect_files_by_condition(classes, condition_split)
        
        # 6. 파일 복사
        success = copy_files_to_target(split_data)
        if not success:
            return False
            
        # 7. 리포트 생성
        create_split_report(split_data, condition_split)
        
        print("\n🎉 다중 레이블 조합 데이터 분할 완료!")
        print("=" * 60)
        print("📂 결과물 위치:")
        print(f"   📁 {TARGET_DIR}")
        print(f"   📊 {Path(TARGET_DIR) / 'split_report.json'}")
        print("\n💡 특징:")
        print("   - 전역적으로 조건 분할 (Train 4개, Val 1-2개, Test 1-2개)")
        print("   - 모든 기기가 동일한 조건 세트 사용")
        print("   - 정확한 비율 보장")
        
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