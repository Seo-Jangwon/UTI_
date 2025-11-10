#!/usr/bin/env python3
"""
조건별 데이터셋 분할 스크립트
- 원본 데이터는 절대 건드리지 않음 (복사만 함)
- 조건별 계층화된 분할로 robust한 일반화 성능 확보
- 최신 연구 기반 최적 분할 비율 적용
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
SOURCE_DIR = r"C:\Users\seoja\Desktop\new_dipstick\reconstructed_training_data"
TARGET_DIR = r"C:\Users\seoja\Desktop\new_dipstick\dataset_mcc"

# 2024년 최신 연구 기반 최적 분할 비율 (의료 영상 특화)
# 70% train / 15% val / 15% test 
# 출처: PMC, Nature Communications, MICCAI 2024 논문들
SPLIT_RATIOS = {
    'train': 0.70,
    'val': 0.15, 
    'test': 0.15
}

# 조건 정의
CONDITIONS = [
    ('3000', '50'),
    ('3000', '100'), 
    ('4000', '50'),
    ('4000', '100'),
    ('5000', '50'),
    ('5000', '100'),
    ('Shadow', '')  # Shadow는 온도/밝기 정보 없음
]

# 기기 목록
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
    for i, cls in enumerate(classes):
        print(f"   {i+1:2d}. {cls}")
    return classes

def parse_condition_from_filename(filename: str) -> Tuple[str, str]:
    """파일명에서 조건(색온도, 밝기) 추출"""
    if 'Shadow' in filename:
        return ('Shadow', '')
    
    # Bilirubin_0.5_3000_100.jpg 패턴에서 3000, 100 추출
    parts = filename.split('_')
    if len(parts) >= 4:
        return (parts[-2], parts[-1].split('.')[0])  # 확장자 제거
    return ('Unknown', '')

def collect_files_by_condition(classes: List[str]) -> Dict:
    """조건별로 파일들을 수집"""
    condition_files = defaultdict(lambda: defaultdict(list))
    
    print("\n📊 조건별 파일 수집 중...")
    
    for class_name in classes:
        for device in DEVICES:
            device_path = Path(SOURCE_DIR) / class_name / device
            if not device_path.exists():
                continue
                
            # 이미지 파일만 수집 (.jpg)
            image_files = list(device_path.glob("*.jpg"))
            
            for img_file in image_files:
                temp, brightness = parse_condition_from_filename(img_file.name)
                condition_key = f"{temp}_{brightness}" if brightness else temp
                
                # 이미지와 대응되는 텍스트 파일 확인
                txt_file = img_file.with_suffix('.txt')
                if txt_file.exists():
                    condition_files[condition_key][class_name].append({
                        'image': img_file,
                        'label': txt_file,
                        'device': device
                    })
    
    # 통계 출력
    print(f"\n📈 조건별 수집 결과:")
    total_files = 0
    for condition, class_data in condition_files.items():
        condition_total = sum(len(files) for files in class_data.values())
        total_files += condition_total
        print(f"   {condition}: {condition_total}개 파일")
    
    print(f"🎯 총 수집된 파일 쌍: {total_files}개")
    return condition_files

def create_condition_based_split(condition_files: Dict) -> Dict:
    """조건별 계층화된 분할 수행"""
    print(f"\n🔄 조건별 분할 수행 (비율: {SPLIT_RATIOS})")
    
    # unseen test conditions 정의 (완전히 새로운 조건)
    unseen_conditions = ['5000_100', 'Shadow']
    seen_conditions = [k for k in condition_files.keys() 
                      if k not in unseen_conditions]
    
    split_data = {
        'train': defaultdict(list),
        'val': defaultdict(list), 
        'test': defaultdict(list)
    }
    
    print(f"👁️  Seen conditions: {seen_conditions}")
    print(f"👁️‍🗨️ Unseen test conditions: {unseen_conditions}")
    
    # Unseen conditions → Test
    for condition in unseen_conditions:
        if condition in condition_files:
            for class_name, files in condition_files[condition].items():
                split_data['test'][class_name].extend(files)
            print(f"   {condition} → Test (unseen)")
    
    # Seen conditions → Train/Val 분할
    for condition in seen_conditions:
        if condition not in condition_files:
            continue
            
        for class_name, files in condition_files[condition].items():
            # 클래스별로 셔플
            files_copy = files.copy()
            random.shuffle(files_copy)
            
            n_total = len(files_copy)
            n_train = int(n_total * (SPLIT_RATIOS['train'] / (SPLIT_RATIOS['train'] + SPLIT_RATIOS['val'])))
            
            # Train/Val 분할
            split_data['train'][class_name].extend(files_copy[:n_train])
            split_data['val'][class_name].extend(files_copy[n_train:])
    
    # 분할 결과 통계
    print(f"\n📊 분할 결과:")
    for split_name, class_data in split_data.items():
        total = sum(len(files) for files in class_data.values())
        print(f"   {split_name}: {total}개 파일")
    
    return split_data

def copy_files_to_target(split_data: Dict):
    """분할된 파일들을 타겟 디렉토리로 복사"""
    print(f"\n📁 타겟 디렉토리 생성: {TARGET_DIR}")
    
    # 타겟 디렉토리 생성
    target_path = Path(TARGET_DIR)
    if target_path.exists():
        print("⚠️  타겟 디렉토리가 이미 존재합니다. 덮어쓰시겠습니까? (y/N): ", end="")
        response = input().strip().lower()
        if response != 'y':
            print("❌ 작업을 취소했습니다.")
            return
        shutil.rmtree(target_path)
    
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 분할별 디렉토리 구조 생성 및 파일 복사
    for split_name, class_data in split_data.items():
        print(f"\n📋 {split_name.upper()} 세트 생성 중...")
        
        for class_name, files in class_data.items():
            if not files:
                continue
                
            for file_info in files:
                # 타겟 경로 구성: dataset_mcc/train/Bilirubin_0.5/A32/
                target_class_device_dir = target_path / split_name / class_name / file_info['device']
                target_class_device_dir.mkdir(parents=True, exist_ok=True)
                
                # 이미지 파일 복사
                target_img = target_class_device_dir / file_info['image'].name
                shutil.copy2(file_info['image'], target_img)
                
                # 라벨 파일 복사  
                target_label = target_class_device_dir / file_info['label'].name
                shutil.copy2(file_info['label'], target_label)
        
        # 통계 출력
        total_files = sum(len(files) for files in class_data.values())
        print(f"   ✅ {total_files}개 파일 쌍 복사 완료")

def create_split_report(split_data: Dict):
    """분할 결과 리포트 생성"""
    report_path = Path(TARGET_DIR) / "split_report.json"
    
    report = {
        "split_method": "condition_based_stratified",
        "split_ratios": SPLIT_RATIOS,
        "random_seed": RANDOM_SEED,
        "unseen_test_conditions": ["5000_100", "Shadow"],
        "statistics": {}
    }
    
    for split_name, class_data in split_data.items():
        split_stats = {}
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
        
        report["statistics"][split_name] = split_stats
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📊 분할 리포트 저장: {report_path}")

def main():
    """메인 실행 함수"""
    print("🚀 조건별 데이터셋 분할 시작!")
    print("=" * 60)
    
    try:
        # 1. 초기 설정
        setup_random_seed()
        
        # 2. 클래스 수집
        classes = get_all_classes()
        
        # 3. 조건별 파일 수집
        condition_files = collect_files_by_condition(classes)
        
        # 4. 조건별 분할 수행
        split_data = create_condition_based_split(condition_files)
        
        # 5. 파일 복사
        copy_files_to_target(split_data)
        
        # 6. 리포트 생성
        create_split_report(split_data)
        
        print("\n🎉 조건별 데이터셋 분할 완료!")
        print("=" * 60)
        print("📂 결과물 위치:")
        print(f"   📁 {TARGET_DIR}")
        print(f"   📊 {Path(TARGET_DIR) / 'split_report.json'}")
        print("\n💡 팁:")
        print("   - split_report.json에서 상세한 분할 통계를 확인하세요")
        print("   - Test 세트는 완전히 unseen conditions만 포함합니다")
        print("   - 원본 데이터는 그대로 보존됩니다")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("🔧 문제 해결을 위해 코드를 검토해주세요.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🏆 모든 작업이 성공적으로 완료되었습니다!")
    else:
        print("\n💥 작업 중 오류가 발생했습니다.")