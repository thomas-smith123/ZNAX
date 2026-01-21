#!/usr/bin/env python3
"""
检查数据集标签分布
"""
import numpy as np
import h5py
from collections import Counter

def main():
    print("Analyzing dataset label distribution...")
    
    # 加载数据
    dataset_path = '/home/jiangrundong/RF Fingerprinting/lora/dataset/dataset_training_no_aug.h5'
    with h5py.File(dataset_path, 'r') as f:
        data = np.array(f['data'])    
        labels = np.array(f['label']) 

    print(f"Original data shape: {data.shape}")
    print(f"Original labels shape: {labels.shape}")
    
    labels_flat = labels.flatten()
    print(f"Flattened labels shape: {labels_flat.shape}")
    
    # 统计标签分布
    label_counts = Counter(labels_flat)
    print(f"Total unique labels: {len(label_counts)}")
    print(f"Label range: {min(label_counts.keys()):.0f} to {max(label_counts.keys()):.0f}")
    
    # 显示每个标签的样本数量
    print("\nLabel distribution:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        print(f"Label {label:.0f}: {count} samples")
    
    # 计算每个类别的样本索引范围
    labels_processed = labels_flat - 1  # 从0开始索引
    print(f"\nProcessed labels range: {labels_processed.min():.0f} to {labels_processed.max():.0f}")
    
    # 检查前1000个样本的标签分布
    print(f"\nFirst 1000 samples label distribution:")
    first_1000_labels = labels_processed[:1000]
    first_1000_counts = Counter(first_1000_labels)
    for label in sorted(first_1000_counts.keys()):
        count = first_1000_counts[label]
        print(f"Label {label:.0f}: {count} samples")
    
    # 计算每个类别的样本数
    samples_per_class = len(labels_flat) // len(label_counts)
    print(f"\nExpected samples per class: {samples_per_class}")
    
    # 找到每个类别的起始索引
    print(f"\nClass boundaries (approximate):")
    for i in range(0, min(10, len(label_counts))):  # 只显示前10个类别
        start_idx = i * samples_per_class
        end_idx = (i + 1) * samples_per_class
        actual_label = labels_processed[start_idx] if start_idx < len(labels_processed) else "N/A"
        print(f"Class {i}: indices {start_idx}-{end_idx-1}, actual label: {actual_label}")

if __name__ == "__main__":
    main()