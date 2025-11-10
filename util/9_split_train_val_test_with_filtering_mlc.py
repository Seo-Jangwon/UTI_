#!/usr/bin/env python3
"""
다중 레이블 조합 데이터 분할 스크립트 - 조건 필터링 버전
- 기기별 + 조명조건별 분할로 완전한 컨닝 방지
- 3000_100, Shadow 조건 제외
- 각 기기마다 서로 다른 조명조건을 train/val/test에 배정
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
SOURCE_DIR = r"C:\Users\seoja\Desktop\new_dipstick\다중_레이블_분류_학습데이터들\balanced_combination_more"
# [설정 필요] 분할된 데이터가 저장될 경로
TARGET_DIR = r"C:\Users\seoja\Desktop\new_dipstick\multilabel_dataset_device_condition_split_filtered"

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

def is_condition_excluded(condition_key: str) -> bool:
    """조건이 제외 대상인지 확인"""
    return condition_key in EXCLUDED_CONDITIONS

def collect_files_by_device_condition(classes: List[str]) -> Dict:
    """기기별 + 조건별로 파일들을 수집 (제외 조건 필터링 적용)"""
    device_condition_files = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    print(f"\n📊 기기별 + 조건별 파일 수집 중... (제외 조건: {EXCLUDED_CONDITIONS})")
    
    total_files = 0
    excluded_count = 0
    condition_stats = defaultdict(int)
    
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
                
                # [추가] 제외 조건 확인
                if is_condition_excluded(condition_key):
                    excluded_count += 1
                    continue
                
                condition_stats[condition_key] += 1
                
                txt_file = img_file.with_suffix('.txt')
                if txt_file.exists():
                    device_condition_files[device][condition_key][class_name].append({
                        'image': img_file,
                        'label': txt_file
                    })
                    total_files += 1
    
    print(f"\n📈 수집 결과:")
    print(f"🎯 총 수집된 파일 쌍: {total_files}개")
    print(f"🚫 제외된 파일 쌍: {excluded_count}개")
    print(f"📱 기기별 분포:")
    for device in DEVICES:
        device_total = sum(len(files) for condition_data in device_condition_files[device].values() 
                          for files in condition_data.values())
        print(f"   {device:10s}: {device_total:5d}개 파일")
    
    print(f"🌡️ 사용된 조건별 분포:")
    for condition, count in sorted(condition_stats.items()):
        print(f"   {condition:12s}: {count:5d}개 파일")
    
    return device_condition_files

def create_device_condition_split_plan(device_condition_files: Dict) -> Dict:
    """
    기기별 + 조건별 분할 계획 생성 (필터링된 조건들만 사용)
    각 기기마다 서로 다른 조건을 train/val/test에 배정
    """
    print(f"\n🎯 기기별 + 조건별 분할 계획 생성 (필터링된 조건들)")
    
    # 각 기기에서 사용 가능한 조건들 파악
    device_conditions = {}
    for device in DEVICES:
        conditions = list(device_condition_files[device].keys())
        device_conditions[device] = conditions
        print(f"📱 {device}: {len(conditions)}개 조건 - {conditions}")
    
    # 각 기기별로 조건을 train/val/test에 랜덤 배정
    split_plan = {}
    
    for device in DEVICES:
        conditions = device_conditions[device][:]
        random.shuffle(conditions)
        
        if not conditions:
            print(f"⚠️  {device}에는 사용 가능한 조건이 없습니다.")
            continue
        
        n_total = len(conditions)
        n_train = max(1, int(n_total * SPLIT_RATIOS['train']))
        n_val = max(1, int(n_total * SPLIT_RATIOS['val'])) if n_total > 2 else 0
        
        # 인덱스 계산
        train_conditions = conditions[:n_train]
        val_conditions = conditions[n_train:n_train + n_val] if n_val > 0 else []
        test_conditions = conditions[n_train + n_val:] if n_total > n_train + n_val else []
        
        # 최소 1개씩은 보장
        if not test_conditions and len(conditions) > 1:
            test_conditions = [train_conditions.pop()]
        if not val_conditions and len(conditions) > 2:
            val_conditions = [train_conditions.pop()]
        
        split_plan[device] = {
            'train': train_conditions,
            'val': val_conditions,
            'test': test_conditions
        }
        
        print(f"\n📋 {device} 분할 계획:")
        print(f"   Train: {train_conditions}")
        print(f"   Val:   {val_conditions}")
        print(f"   Test:  {test_conditions}")
    
    return split_plan

def execute_split_with_plan(device_condition_files: Dict, split_plan: Dict) -> Dict:
    """분할 계획에 따라 실제 파일 분할 실행"""
    print(f"\n🔄 분할 계획 실행 중...")
    
    split_data = {
        'train': defaultdict(list),
        'val': defaultdict(list),
        'test': defaultdict(list)
    }
    
    total_files = {'train': 0, 'val': 0, 'test': 0}
    
    for device in DEVICES:
        if device not in split_plan:
            continue
            
        device_plan = split_plan[device]
        
        for split_name in ['train', 'val', 'test']:
            conditions = device_plan[split_name]
            split_files = 0
            
            for condition in conditions:
                if condition in device_condition_files[device]:
                    condition_data = device_condition_files[device][condition]
                    
                    for class_name, files in condition_data.items():
                        for file_info in files:
                            # 기기 정보 추가
                            file_info_with_device = file_info.copy()
                            file_info_with_device['device'] = device
                            
                            split_data[split_name][class_name].append(file_info_with_device)
                            split_files += 1
            
            total_files[split_name] += split_files
            print(f"   📱 {device} → {split_name.upper()}: {split_files}개 파일")
    
    print(f"\n📊 최종 분할 결과:")
    for split_name, total in total_files.items():
        ratio = total / sum(total_files.values()) if sum(total_files.values()) > 0 else 0
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
        
        for class_name, files in class_data.items():
            if not files:
                continue
                
            for file_info in files:
                # 타겟 경로 구성: TARGET_DIR/train/class_name/device/
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

def create_split_report(split_data: Dict, split_plan: Dict):
    """분할 결과 리포트 생성"""
    report_path = Path(TARGET_DIR) / "split_report.json"
    
    report = {
        "split_method": "device_condition_based_split_filtered",
        "split_ratios_target": SPLIT_RATIOS,
        "random_seed": RANDOM_SEED,
        "devices": DEVICES,
        "excluded_conditions": EXCLUDED_CONDITIONS,
        "split_plan": split_plan,
        "statistics": {}
    }
    
    grand_total = 0
    
    for split_name, class_data in split_data.items():
        split_stats = {}
        split_total_files = 0
        device_counts = defaultdict(int)
        condition_counts = defaultdict(int)
        
        for class_name, files in class_data.items():
            class_device_counts = defaultdict(int)
            class_condition_counts = defaultdict(int)
            
            for file_info in files:
                device = file_info['device']
                temp, brightness = parse_condition_from_filename(file_info['image'].name)
                condition = f"{temp}_{brightness}" if brightness else temp
                
                device_counts[device] += 1
                condition_counts[condition] += 1
                class_device_counts[device] += 1
                class_condition_counts[condition] += 1
            
            split_stats[class_name] = {
                "total_files": len(files),
                "devices": dict(class_device_counts),
                "conditions": dict(class_condition_counts)
            }
            split_total_files += len(files)
        
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
    print("🚀 다중 레이블 조합 데이터 분할 시작! (조건 필터링 버전)")
    print("=" * 60)
    print(f"🚫 제외할 조건: {EXCLUDED_CONDITIONS}")
    print("=" * 60)
    
    try:
        # 1. 초기 설정
        setup_random_seed()
        
        # 2. 클래스 수집
        classes = get_all_classes()
        
        # 3. 기기별 + 조건별 파일 수집 (필터링 적용)
        device_condition_files = collect_files_by_device_condition(classes)
        
        if not device_condition_files:
            print("❌ 필터링 후 사용 가능한 데이터가 없습니다!")
            return False
        
        # 4. 분할 계획 생성
        split_plan = create_device_condition_split_plan(device_condition_files)
        
        # 5. 분할 실행
        split_data = execute_split_with_plan(device_condition_files, split_plan)
        
        # 6. 파일 복사
        success = copy_files_to_target(split_data)
        if not success:
            return False
            
        # 7. 리포트 생성
        create_split_report(split_data, split_plan)
        
        print("\n🎉 다중 레이블 조합 데이터 분할 완료!")
        print("=" * 60)
        print("📂 결과물 위치:")
        print(f"   📁 {TARGET_DIR}")
        print(f"   📊 {Path(TARGET_DIR) / 'split_report.json'}")
        print("\n💡 특징:")
        print(f"   - 제외된 조건: {EXCLUDED_CONDITIONS}")
        print("   - 기기별로 서로 다른 조건을 train/val/test에 배정")
        print("   - 완전한 컨닝 방지 보장")
        
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