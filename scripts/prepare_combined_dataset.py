#!/usr/bin/env -S uv run python
"""
COCO（personクラス）とWIDERFace（faceクラス）を統合したYOLOトレーニング用データセットを作成

使用方法:
    # COCO8を使用（デフォルト、テスト用）
    uv run python scripts/prepare_combined_dataset.py --dataset coco8
    
    # COCO全体を使用（本番用）
    uv run python scripts/prepare_combined_dataset.py --dataset coco
    
    # dry-runモード（実際にはコピーしない）
    uv run python scripts/prepare_combined_dataset.py --dataset coco8 --dry-run
注意: WIDERFaceアノテーションファイルは別途ダウンロードが必要です
"""

import argparse
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path


class DatasetPreparer:
    def __init__(self, base_dir=None, dataset_type="coco8"):
        if base_dir is None:
            # スクリプトの親ディレクトリ（yolo-face/）をbase_dirとする
            base_dir = Path(__file__).parent.parent.resolve()
        self.base_dir = Path(base_dir)
        self.downloads_dir = self.base_dir / "downloads"
        self.datasets_dir = self.base_dir / "datasets"
        self.output_dir = self.datasets_dir / "person_face"
        self.dataset_type = dataset_type
        
        # COCO8のURL
        self.coco8_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
        
    def download_file(self, url, dest_path):
        """ファイルをダウンロード（進捗表示付き）"""
        print(f"\nダウンロード中: {url}")
        print(f"保存先: {dest_path}")
        
        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(block_num * block_size * 100 / total_size, 100)
                print(f"\r進捗: {percent:.1f}% ({block_num * block_size / 1024 / 1024:.1f} MB)", end='')
        
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print()  # 改行
        print(f"✓ ダウンロード完了")
        
    def extract_zip(self, zip_path, extract_to):
        """ZIPファイルを解凍"""
        print(f"\n解凍中: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ 解凍完了: {extract_to}")
        
    def prepare_coco8(self):
        """COCO8データセットをダウンロード・準備"""
        print("\n" + "=" * 60)
        print("COCO8データセット準備")
        print("=" * 60)
        
        coco8_zip = self.downloads_dir / "coco8.zip"
        coco8_dir = self.downloads_dir / "coco8"
        
        # ダウンロード
        if not coco8_zip.exists():
            self.download_file(self.coco8_url, coco8_zip)
        else:
            print(f"\n✓ COCO8データセットは既にダウンロード済み: {coco8_zip}")
        
        # 解凍
        if not coco8_dir.exists():
            self.extract_zip(coco8_zip, self.downloads_dir)
        else:
            print(f"✓ COCO8データセットは既に解凍済み: {coco8_dir}")
        
        # personクラスのみを抽出
        print("\npersonクラスを抽出中...")
        person_count = 0
        person_images = 0
        
        for split in ["train", "val"]:
            images_dir = coco8_dir / "images" / split
            labels_dir = coco8_dir / "labels" / split
            
            if not images_dir.exists():
                print(f"Warning: {images_dir} が見つかりません")
                continue
            
            if not labels_dir.exists():
                print(f"Warning: {labels_dir} が見つかりません")
                continue
            
            for label_file in labels_dir.glob("*.txt"):
                # ラベルファイルを読み込み
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                # personクラス（class_id=0）のみをフィルタ
                person_lines = [line for line in lines if line.strip() and line.strip().startswith('0 ')]
                
                if not person_lines:
                    continue
                
                # 対応する画像ファイルを探す
                image_file = None
                for ext in ['.jpg', '.jpeg', '.png']:
                    potential_image = images_dir / f"{label_file.stem}{ext}"
                    if potential_image.exists():
                        image_file = potential_image
                        break
                
                if not image_file:
                    print(f"Warning: {label_file.stem} の画像が見つかりません")
                    continue
                
                # コピー先
                dest_image = self.output_dir / "images" / split / image_file.name
                dest_label = self.output_dir / "labels" / split / label_file.name
                
                # コピー
                shutil.copy(image_file, dest_image)
                
                # personクラスのみのラベルを保存
                with open(dest_label, 'w') as f:
                    f.writelines(person_lines)
                
                person_count += len(person_lines)
                person_images += 1
        
        print(f"✓ COCO8から{person_images}枚の画像、{person_count}件のpersonデータを抽出")
        return person_count
    
    def prepare_coco(self):
        """COCO全体データセットからpersonクラスを抽出"""
        print("\n" + "=" * 60)
        print("COCOデータセット準備")
        print("=" * 60)
        
        coco_dir = self.downloads_dir / "coco"
        
        if not coco_dir.exists():
            print(f"✗ COCOデータセットが見つかりません: {coco_dir}")
            print("\n次のコマンドでダウンロードしてください:")
            print("python3 scripts/download_datasets.py --coco")
            return 0
        
        person_count = 0
        person_images = 0
        
        # COCOのpersonクラスID（COCO 80クラスの中でpersonはID=1）
        # 注: YOLO TXT形式（coco8など）では0ですが、JSONアノテーションでは1です
        PERSON_CLASS_ID = 1
        
        for split in ["train", "val"]:
            split_name = f"{split}2017"
            images_dir = coco_dir / split_name
            annotations_file = coco_dir / "annotations" / f"instances_{split_name}.json"
            
            if not images_dir.exists():
                print(f"Warning: {images_dir} が見つかりません")
                continue
            
            if not annotations_file.exists():
                print(f"Warning: {annotations_file} が見つかりません")
                continue
            
            print(f"\n{split}セットを処理中...")
            
            # COCOアノテーションJSONを読み込み
            with open(annotations_file, 'r') as f:
                coco_data = json.load(f)
            
            # 画像ID to ファイル名のマッピング
            image_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
            
            # 画像ID to 幅・高さのマッピング
            image_id_to_size = {img['id']: (img['width'], img['height']) for img in coco_data['images']}
            
            # 画像IDごとにpersonアノテーションを集める
            image_annotations = {}
            for ann in coco_data['annotations']:
                if ann['category_id'] == PERSON_CLASS_ID:
                    image_id = ann['image_id']
                    if image_id not in image_annotations:
                        image_annotations[image_id] = []
                    image_annotations[image_id].append(ann)
            
            print(f"  {len(image_annotations)}枚の画像にpersonが含まれています")
            
            # 各画像を処理
            for image_id, annotations in image_annotations.items():
                file_name = image_id_to_file[image_id]
                image_path = images_dir / file_name
                
                if not image_path.exists():
                    continue
                
                width, height = image_id_to_size[image_id]
                
                # YOLO形式のラベルを生成
                yolo_labels = []
                for ann in annotations:
                    # COCO bbox: [x, y, width, height] (左上座標 + サイズ)
                    x, y, w, h = ann['bbox']
                    
                    # YOLO形式に変換: [class_id, x_center, y_center, width, height] (正規化)
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_norm = w / width
                    h_norm = h / height
                    
                    # クラスID=0（person）
                    yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                    person_count += 1
                
                if yolo_labels:
                    # 画像をコピー
                    dest_image = self.output_dir / "images" / split / file_name
                    shutil.copy(image_path, dest_image)
                    
                    # ラベルを保存
                    label_file = dest_image.stem + ".txt"
                    dest_label = self.output_dir / "labels" / split / label_file
                    with open(dest_label, 'w') as f:
                        f.writelines(yolo_labels)
                    
                    person_images += 1
            
            print(f"  ✓ {split}から{person_images}枚の画像を抽出")
        
        print(f"\n✓ COCOから合計{person_images}枚の画像、{person_count}件のpersonデータを抽出")
        return person_count
    
    def convert_wider_to_yolo(self, wider_annotation_file, image_dir_base, output_dir_base, split="train"):
        """
        WIDERFaceアノテーションをYOLO形式に変換
        
        WIDERFaceフォーマット:
        <image_path>
        <num_faces>
        <x> <y> <w> <h> <blur> <expression> <illumination> <invalid> <occlusion> <pose>
        ...
        """
        print(f"\n=== WIDERFace {split}アノテーション変換 ===")
        
        if not wider_annotation_file.exists():
            print(f"Warning: {wider_annotation_file} が見つかりません")
            return 0
        
        face_count = 0
        image_count = 0
        current_image = None
        num_faces = 0
        
        with open(wider_annotation_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # 画像パス（相対パス形式: "0--Parade/0_Parade_marchingband_1_849.jpg"）
            if line and not line.isdigit():
                current_image = line
                i += 1
                
                if i >= len(lines):
                    break
                    
                # 顔の数
                try:
                    num_faces = int(lines[i].strip())
                except ValueError:
                    i += 1
                    continue
                    
                i += 1
                
                # 画像ファイルパス
                image_path = image_dir_base / current_image
                if not image_path.exists():
                    # num_facesの行数だけスキップ
                    i += num_faces
                    continue
                
                # 画像サイズを取得（標準ライブラリのみ）
                try:
                    def get_image_size(filepath):
                        """標準ライブラリのみで画像サイズを取得"""
                        with open(filepath, 'rb') as f:
                            head = f.read(24)
                            
                        # JPEG
                        if len(head) >= 2 and head[0:2] == b'\xff\xd8':
                            f = open(filepath, 'rb')
                            try:
                                f.seek(0)
                                size = 2
                                ftype = 0
                                while not 0xc0 <= ftype <= 0xcf or ftype in (0xc4, 0xc8, 0xcc):
                                    f.seek(size, 1)
                                    byte = f.read(1)
                                    while ord(byte) == 0xff:
                                        byte = f.read(1)
                                    ftype = ord(byte)
                                    size = int.from_bytes(f.read(2), 'big') - 2
                                f.seek(1, 1)
                                height = int.from_bytes(f.read(2), 'big')
                                width = int.from_bytes(f.read(2), 'big')
                                return width, height
                            finally:
                                f.close()
                        
                        # PNG
                        elif head[0:8] == b'\x89PNG\r\n\x1a\n' and head[12:16] == b'IHDR':
                            w, h = int.from_bytes(head[16:20], 'big'), int.from_bytes(head[20:24], 'big')
                            return w, h
                        
                        return None, None
                    
                    img_width, img_height = get_image_size(str(image_path))
                    if not img_width or not img_height:
                        # サイズ取得失敗
                        i += num_faces
                        continue
                except Exception as e:
                    # 画像読み込み失敗
                    i += num_faces
                    continue
                
                yolo_annotations = []
                
                # 各顔のアノテーション
                for _ in range(num_faces):
                    if i >= len(lines):
                        break
                    
                    parts = lines[i].strip().split()
                    if len(parts) < 4:
                        i += 1
                        continue
                    
                    try:
                        x, y, w, h = map(int, parts[:4])
                    except ValueError:
                        i += 1
                        continue
                    
                    # 無効なBBoxをスキップ
                    if w <= 0 or h <= 0 or x < 0 or y < 0:
                        i += 1
                        continue
                    
                    # YOLO形式に変換
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    w_norm = w / img_width
                    h_norm = h / img_height
                    
                    # 範囲チェック
                    if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < w_norm <= 1 and 0 < h_norm <= 1:
                        # クラスID=1（face）
                        yolo_annotations.append(f"1 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                        face_count += 1
                    
                    i += 1
                
                # ラベルファイルに書き込み
                if yolo_annotations:
                    # 出力パスを構築
                    # 画像ファイル名から拡張子を除いてラベルファイル名を作成
                    image_filename = Path(current_image).name
                    label_filename = Path(image_filename).stem + ".txt"
                    
                    dest_label = output_dir_base / "labels" / split / label_filename
                    with open(dest_label, 'w') as lf:
                        lf.writelines(yolo_annotations)
                    
                    # 画像をコピー
                    dest_image = output_dir_base / "images" / split / image_filename
                    shutil.copy(image_path, dest_image)
                    image_count += 1
            else:
                i += 1
        
        print(f"✓ WIDERFace {split}から{image_count}枚の画像、{face_count}件のfaceデータを変換")
        return face_count
    
    def prepare_wider(self):
        """WIDERFaceデータセットを準備"""
        print("\n" + "=" * 60)
        print("WIDERFaceデータセット準備")
        print("=" * 60)
        
        wider_split_dir = self.downloads_dir / "wider_face_split"
        wider_train_images = self.downloads_dir / "WIDER_train" / "images"
        wider_val_images = self.downloads_dir / "WIDER_val" / "images"
        
        face_count = 0
        
        # Train annotations
        train_annotation = wider_split_dir / "wider_face_train_bbx_gt.txt"
        if train_annotation.exists() and wider_train_images.exists():
            face_count += self.convert_wider_to_yolo(
                train_annotation,
                wider_train_images,
                self.output_dir,
                "train"
            )
        else:
            if not train_annotation.exists():
                print(f"Warning: {train_annotation} が見つかりません")
            if not wider_train_images.exists():
                print(f"Warning: {wider_train_images} が見つかりません")
        
        # Val annotations
        val_annotation = wider_split_dir / "wider_face_val_bbx_gt.txt"
        if val_annotation.exists() and wider_val_images.exists():
            face_count += self.convert_wider_to_yolo(
                val_annotation,
                wider_val_images,
                self.output_dir,
                "val"
            )
        
        return face_count
    
    def create_data_yaml(self, person_count, face_count):
        """data.yaml設定ファイルを作成"""
        print("\n" + "=" * 60)
        print("data.yaml作成")
        print("=" * 60)
        
        yaml_content = f"""# Person + Face統合データセット ({self.dataset_type.upper()})
path: {self.output_dir}
train: images/train
val: images/val

# クラス定義
nc: 2
names:
  0: person
  1: face

# 統計情報
# person: {person_count}件
# face: {face_count}件
"""
        
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"✓ {yaml_path} を作成")
        print(f"\n統計:")
        print(f"  - person: {person_count}件")
        print(f"  - face: {face_count}件")
        
    def print_wider_instructions(self):
        """WIDERFaceアノテーションのダウンロード手順を表示"""
        print("\n" + "=" * 60)
        print("⚠️  WIDERFaceアノテーションの追加手順")
        print("=" * 60)
        print("\n現在、personクラスのみのデータセットが作成されました。")
        print("faceクラスを追加するには、以下の手順でWIDERFaceアノテーションを取得してください:")
        print("\n1. WIDERFaceアノテーションをダウンロード:")
        print("   http://shuoyang1213.me/WIDERFACE/")
        print("   'Face annotations' セクションから 'WIDER Face Training Images' アノテーションをダウンロード")
        print("\n2. ダウンロードしたファイルを解凍:")
        print(f"   解凍先: {self.downloads_dir}/wider_face_split/")
        print("\n3. 期待されるファイル:")
        print(f"   {self.downloads_dir}/wider_face_split/wider_face_train_bbx_gt.txt")
        print("\n次のステップ:")
        print(f"uv run python scripts/train_yolo.py --data {self.output_dir}/data.yaml")
        print("\nアノテーションが見つかれば、自動的にfaceクラスも統合されます。")
        print("=" * 60)
        
    def run(self):
        """メイン処理"""
        print("\n" + "=" * 70)
        print(f"COCO・WIDERFace統合データセット作成 ({self.dataset_type.upper()})")
        print("=" * 70)
        
        # ディレクトリ作成
        self.downloads_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for split in ["train", "val"]:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        # データセットタイプに応じて処理
        if self.dataset_type == "coco8":
            person_count = self.prepare_coco8()
        elif self.dataset_type == "coco":
            person_count = self.prepare_coco()
        else:
            print(f"✗ 未対応のデータセットタイプ: {self.dataset_type}")
            return
        
        # WIDERFace処理
        face_count = 0
        wider_train_annotation = self.downloads_dir / "wider_face_split" / "wider_face_train_bbx_gt.txt"
        
        if wider_train_annotation.exists():
            face_count = self.prepare_wider()
        
        # data.yaml作成
        self.create_data_yaml(person_count, face_count)
        
        # 完了メッセージ
        print("\n" + "=" * 70)
        print("✓ データセット作成完了！")
        print("=" * 70)
        print(f"\n出力先: {self.output_dir}")
        print(f"\n次のコマンドでトレーニングを開始できます:")
        print(f"yolo task=detect mode=train model=yolo11n.pt \\")
        print(f"  data={self.output_dir}/data.yaml \\")
        print(f"  epochs=50 imgsz=640")
        
        # WIDERFace手順の表示
        if not wider_train_annotation.exists():
            self.print_wider_instructions()


def main():
    parser = argparse.ArgumentParser(
        description="COCO・WIDERFace統合データセット作成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # COCO8を使用（デフォルト、テスト用）
  uv run python scripts/prepare_combined_dataset.py --dataset coco8
  
  # COCO全体を使用（本番用）
  uv run python scripts/prepare_combined_dataset.py --dataset coco
  
  # dry-runモード（実際にはコピーしない）
  uv run python scripts/prepare_combined_dataset.py --dataset coco8 --dry-run"""
    )
    
    parser.add_argument(
        "--dataset",
        choices=["coco8", "coco"],
        default="coco8",
        help="使用するCOCOデータセット (coco8: 小規模テスト用, coco: 本番用全体データ)"
    )
    
    args = parser.parse_args()
    
    preparer = DatasetPreparer(dataset_type=args.dataset)
    preparer.run()


if __name__ == "__main__":
    main()
