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

def augment_labels(dataset_dir, face_model_path, person_model_path, conf=0.3, iou_threshold=0.5, batch_size=16):
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
        if not image_files:
            continue

        print(f"\n{split} セットの処理中 ({len(image_files)} 枚, batch_size={batch_size})...")

        # バッチ処理
        for i in tqdm(range(0, len(image_files), batch_size)):
            batch_paths = image_files[i : i + batch_size]
            
            # 1. 各画像の現状ラベルを読み込む & 補完が必要か判定
            batch_info = []
            for img_path in batch_paths:
                label_path = split_labels / f"{img_path.stem}.txt"
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
                
                has_person = any(l['class'] == PERSON_CLASS_ID for l in existing_labels)
                has_face = any(l['class'] == FACE_CLASS_ID for l in existing_labels)
                
                batch_info.append({
                    'img_path': img_path,
                    'label_path': label_path,
                    'existing': existing_labels,
                    'needs_face': not has_face,
                    'needs_person': not has_person,
                    'new_to_write': []
                })

            # 2. 一括顔検出
            face_targets = [info['img_path'] for info in batch_info if info['needs_face']]
            if face_targets:
                face_results = face_model.predict(face_targets, conf=conf, verbose=False, batch=batch_size)
                
                # 結果を元の画像に紐付け
                target_idx = 0
                for info in batch_info:
                    if info['needs_face']:
                        res = face_results[target_idx]
                        for box in res.boxes:
                            xywhn = box.xywhn[0].tolist()
                            # 重複チェック
                            is_duplicate = False
                            for existing in info['existing']:
                                if existing['class'] == FACE_CLASS_ID:
                                    if calculate_iou(xywhn, existing['bbox']) > iou_threshold:
                                        is_duplicate = True
                                        break
                            if not is_duplicate:
                                info['new_to_write'].append(f"{FACE_CLASS_ID} {' '.join(map(str, xywhn))}")
                                info['existing'].append({'class': FACE_CLASS_ID, 'bbox': xywhn})
                        target_idx += 1

            # 3. 一括人検出
            person_targets = [info['img_path'] for info in batch_info if info['needs_person']]
            if person_targets:
                person_results = person_model.predict(person_targets, conf=conf, verbose=False, batch=batch_size)
                
                target_idx = 0
                for info in batch_info:
                    if info['needs_person']:
                        res = person_results[target_idx]
                        for box in res.boxes:
                            if int(box.cls[0]) == 0: # person
                                xywhn = box.xywhn[0].tolist()
                                is_duplicate = False
                                for existing in info['existing']:
                                    if existing['class'] == PERSON_CLASS_ID:
                                        if calculate_iou(xywhn, existing['bbox']) > iou_threshold:
                                            is_duplicate = True
                                            break
                                if not is_duplicate:
                                    info['new_to_write'].append(f"{PERSON_CLASS_ID} {' '.join(map(str, xywhn))}")
                                    info['existing'].append({'class': PERSON_CLASS_ID, 'bbox': xywhn})
                        target_idx += 1

            # 4. 書き出し
            for info in batch_info:
                if info['new_to_write']:
                    with open(info['label_path'], 'a') as f:
                        for nl in info['new_to_write']:
                            if info['label_path'].stat().st_size > 0:
                                f.write(f"\n{nl}")
                            else:
                                f.write(nl)
                                f.write("\n")

    print("\n✓ ラベル補正が完了しました。")

def main():
    parser = argparse.ArgumentParser(description="不完全なデータセットに擬似ラベルを付与する（バッチ処理対応）")
    parser.add_argument("--dataset", type=str, default="datasets/person_face", help="対象データセットのディレクトリ")
    parser.add_argument("--face-model", type=str, default="yolov12m-face.pt", help="顔検出用モデル")
    parser.add_argument("--person-model", type=str, default="yolo12m.pt", help="人検出用モデル")
    parser.add_argument("--conf", type=float, default=0.3, help="信頼度しきい値")
    parser.add_argument("--batch", type=int, default=16, help="バッチサイズ")
    
    args = parser.parse_args()
    
    augment_labels(args.dataset, args.face_model, args.person_model, args.conf, batch_size=args.batch)

if __name__ == "__main__":
    main()
