#!/usr/bin/env -S uv run python
"""
擬似ラベル（Pseudo-labeling）付与スクリプト

不完全なデータセット（身体のみ、または顔のみ）に対して、
既存モデルを使用して足りないラベルを自動補完します。

機能:
  - COCO画像（personのみ）に対して顔検出を実行し追記
  - WIDERFace画像（faceのみ）に対して人検出を実行し追記
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
from tqdm import tqdm

def calculate_iou(box1, box2):
    """YOLO形式(center_x, center_y, width, height)のIoUを計算"""
    b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    b1_area = box1[2] * box1[3]
    b2_area = box2[2] * box2[3]
    union_area = b1_area + b2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def augment_labels(dataset_dir, face_model_path, person_model_path, conf=0.3, iou_threshold=0.5):
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    # モデルのロード
    print(f"モデルをロード中...")
    face_model = YOLO(face_model_path)
    person_model = YOLO(person_model_path)

    # クラスID定義
    PERSON_CLASS_ID = 0
    FACE_CLASS_ID = 1

    for split in ["train", "val"]:
        split_images = images_dir / split
        split_labels = labels_dir / split
        
        if not split_images.exists():
            continue

        image_files = list(split_images.glob("*.jpg")) + list(split_images.glob("*.png"))
        print(f"\n{split} セットの処理中 ({len(image_files)} 枚)...")

        for img_path in tqdm(image_files):
            label_path = split_labels / f"{img_path.stem}.txt"
            
            # 現在のラベルを読み込み
            existing_labels = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            existing_labels.append({
                                'class': int(parts[0]),
                                'bbox': [float(x) for x in parts[1:5]]
                            })

            # 既にどのようなラベルを持っているか（統計用）
            has_person = any(l['class'] == PERSON_CLASS_ID for l in existing_labels)
            has_face = any(l['class'] == FACE_CLASS_ID for l in existing_labels)

            new_labels_to_write = []

            # 1. 顔検出 (全画像に対して実行し、既存ラベルと重複しないものだけ追加)
            results = face_model.predict(img_path, conf=conf, verbose=False)
            for res in results:
                for box in res.boxes:
                    xywhn = box.xywhn[0].tolist()
                    # 既存の全顔ラベルと比較
                    is_duplicate = False
                    for existing in existing_labels:
                        if existing['class'] == FACE_CLASS_ID:
                            if calculate_iou(xywhn, existing['bbox']) > iou_threshold:
                                is_duplicate = True
                                break
                    if not is_duplicate:
                        new_labels_to_write.append(f"{FACE_CLASS_ID} {' '.join(map(str, xywhn))}")
                        # 重複チェックのために既存リストにも追加
                        existing_labels.append({'class': FACE_CLASS_ID, 'bbox': xywhn})

            # 2. 人検出 (全画像に対して実行)
            results = person_model.predict(img_path, conf=conf, verbose=False)
            for res in results:
                for box in res.boxes:
                    if int(box.cls[0]) == 0: #person
                        xywhn = box.xywhn[0].tolist()
                        is_duplicate = False
                        for existing in existing_labels:
                            if existing['class'] == PERSON_CLASS_ID:
                                if calculate_iou(xywhn, existing['bbox']) > iou_threshold:
                                    is_duplicate = True
                                    break
                        if not is_duplicate:
                            new_labels_to_write.append(f"{PERSON_CLASS_ID} {' '.join(map(str, xywhn))}")
                            existing_labels.append({'class': PERSON_CLASS_ID, 'bbox': xywhn})

            # 新しいラベルがあれば追記
            if new_labels_to_write:
                with open(label_path, 'a') as f:
                    for nl in new_labels_to_write:
                        # ファイルが空でない場合に改行を入れる
                        if label_path.stat().st_size > 0:
                            f.write(f"\n{nl}")
                        else:
                            f.write(nl)
                            f.write("\n") # ensure newline for next

    print("\n✓ ラベル補正が完了しました。")

def main():
    parser = argparse.ArgumentParser(description="不完全なデータセットに擬似ラベルを付与する")
    parser.add_argument("--dataset", type=str, default="datasets/person_face", help="対象データセットのディレクトリ")
    parser.add_argument("--face-model", type=str, default="face-yolo11m.pt", help="顔検出用モデル")
    parser.add_argument("--person-model", type=str, default="yolo11m.pt", help="人検出用モデル")
    parser.add_argument("--conf", type=float, default=0.3, help="信頼度しきい値")
    
    args = parser.parse_args()
    
    augment_labels(args.dataset, args.face_model, args.person_model, args.conf)

if __name__ == "__main__":
    main()
