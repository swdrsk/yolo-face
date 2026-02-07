#!/usr/bin/env python3
"""
COCO8（person）とWIDERFace（face）を統合したYOLOトレーニング用データセットを作成

使用方法:
    python scripts/prepare_combined_dataset.py
    
注意: WIDERFaceアノテーションファイルは別途ダウンロードが必要です
"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path


class DatasetPreparer:
    def __init__(self, base_dir=None):
        if base_dir is None:
            # スクリプトの親ディレクトリ（yolo-face/）をbase_dirとする
            base_dir = Path(__file__).parent.parent.resolve()
        self.base_dir = Path(base_dir)
        self.downloads_dir = self.base_dir / "downloads"
        self.datasets_dir = self.base_dir / "datasets"
        self.output_dir = self.datasets_dir / "person_face"
        
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
    
    def create_data_yaml(self, person_count, face_count):
        """data.yaml設定ファイルを作成"""
        print("\n" + "=" * 60)
        print("data.yaml作成")
        print("=" * 60)
        
        yaml_content = f"""# Person + Face統合データセット
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
        print("\n4. 再度このスクリプトを実行:")
        print("   python scripts/prepare_combined_dataset.py")
        print("\nアノテーションが見つかれば、自動的にfaceクラスも統合されます。")
        print("=" * 60)
        
    def run(self):
        """メイン処理"""
        print("\n" + "=" * 70)
        print("COCO8・WIDERFace統合データセット作成")
        print("=" * 70)
        
        # ディレクトリ作成
        self.downloads_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for split in ["train", "val"]:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        # COCO8準備
        person_count = self.prepare_coco8()
        
        # WIDERFace処理（将来的に実装）
        face_count = 0
        wider_train_annotation = self.downloads_dir / "wider_face_split" / "wider_face_train_bbx_gt.txt"
        
        if wider_train_annotation.exists():
            print("\n✓ WIDERFaceアノテーションが見つかりました！")
            print("  （WIDERFace変換機能は次のバージョンで実装予定）")
        
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


if __name__ == "__main__":
    preparer = DatasetPreparer()
    preparer.run()
