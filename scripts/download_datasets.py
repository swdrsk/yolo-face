#!/usr/bin/env python3
"""
COCO全体データセットとWIDERFaceデータセットをダウンロード

使用方法:
    # 全てダウンロード
    python scripts/download_datasets.py --all
    
    # COCOのみ
    python scripts/download_datasets.py --coco
    
    # WIDERFaceのみ
    python scripts/download_datasets.py --wider
    
    # ドライラン（実際にダウンロードしない）
    python scripts/download_datasets.py --all --dry-run
"""

import argparse
import os
import subprocess
from pathlib import Path


class DatasetDownloader:
    def __init__(self, base_dir=None, dry_run=False):
        if base_dir is None:
            # スクリプトの親ディレクトリ（yolo-face/）をbase_dirとする
            base_dir = Path(__file__).parent.parent.resolve()
        self.base_dir = Path(base_dir)
        self.downloads_dir = self.base_dir / "downloads"
        self.dry_run = dry_run
        
        # COCO Dataset URLs
        self.coco_urls = {
            "train_images": "http://images.cocodataset.org/zips/train2017.zip",  # ~18GB
            "val_images": "http://images.cocodataset.org/zips/val2017.zip",  # ~1GB
            "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",  # ~241MB
        }
        
        # WIDERFace Dataset URLs
        self.wider_urls = {
            "train_images": "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip",  # ~1.5GB
            "val_images": "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip",  # ~345MB
            "annotations": "http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip",  # ~3MB
        }
    
    def download_file(self, url, dest_path, description):
        """curlを使用してファイルをダウンロード"""
        print(f"\n{'[DRY RUN] ' if self.dry_run else ''}ダウンロード: {description}")
        print(f"  URL: {url}")
        print(f"  保存先: {dest_path}")
        
        if self.dry_run:
            print(f"  ✓ ドライラン: 実際にはダウンロードしません")
            return
        
        # 既に存在する場合はスキップ
        if dest_path.exists():
            file_size = dest_path.stat().st_size / (1024 * 1024 * 1024)  # GB
            print(f"  ✓ 既にダウンロード済み ({file_size:.2f} GB)")
            return
        
        # curlでダウンロード
        cmd = [
            "curl",
            "-L",  # リダイレクトをフォロー
            "-#",  # プログレスバー表示
            "-o", str(dest_path),
            url
        ]
        
        print(f"  実行中: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            file_size = dest_path.stat().st_size / (1024 * 1024 * 1024)  # GB
            print(f"  ✓ ダウンロード完了 ({file_size:.2f} GB)")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ ダウンロード失敗: {e}")
            raise
    
    def extract_zip(self, zip_path, extract_to, description):
        """ZIPファイルを解凍"""
        print(f"\n{'[DRY RUN] ' if self.dry_run else ''}解凍: {description}")
        print(f"  ファイル: {zip_path}")
        print(f"  解凍先: {extract_to}")
        
        if self.dry_run:
            print(f"  ✓ ドライラン: 実際には解凍しません")
            return
        
        if not zip_path.exists():
            print(f"  ✗ ZIPファイルが見つかりません: {zip_path}")
            return
        
        # unzipコマンドで解凍
        cmd = ["unzip", "-q", "-o", str(zip_path), "-d", str(extract_to)]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"  ✓ 解凍完了")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ 解凍失敗: {e}")
            raise
    
    def download_coco(self):
        """COCOデータセットをダウンロード"""
        print("\n" + "=" * 70)
        print("COCOデータセット ダウンロード")
        print("=" * 70)
        print("\n合計サイズ: ~19GB")
        
        coco_dir = self.downloads_dir / "coco"
        coco_dir.mkdir(parents=True, exist_ok=True)
        
        # Train Images (~18GB)
        train_zip = coco_dir / "train2017.zip"
        self.download_file(
            self.coco_urls["train_images"],
            train_zip,
            "COCO Train Images 2017 (~18GB)"
        )
        self.extract_zip(train_zip, coco_dir, "COCO Train Images")
        
        # Validation Images (~1GB)
        val_zip = coco_dir / "val2017.zip"
        self.download_file(
            self.coco_urls["val_images"],
            val_zip,
            "COCO Val Images 2017 (~1GB)"
        )
        self.extract_zip(val_zip, coco_dir, "COCO Val Images")
        
        # Annotations (~241MB)
        anno_zip = coco_dir / "annotations_trainval2017.zip"
        self.download_file(
            self.coco_urls["annotations"],
            anno_zip,
            "COCO Annotations 2017 (~241MB)"
        )
        self.extract_zip(anno_zip, coco_dir, "COCO Annotations")
        
        print("\n✓ COCOデータセット準備完了")
        print(f"  場所: {coco_dir}")
    
    def download_wider(self):
        """WIDERFaceデータセットをダウンロード"""
        print("\n" + "=" * 70)
        print("WIDERFaceデータセット ダウンロード")
        print("=" * 70)
        print("\n合計サイズ: ~1.9GB")
        
        wider_dir = self.downloads_dir / "wider_face"
        wider_dir.mkdir(parents=True, exist_ok=True)
        
        # Train Images (~1.5GB)
        train_zip = self.downloads_dir / "WIDER_train.zip"
        self.download_file(
            self.wider_urls["train_images"],
            train_zip,
            "WIDERFace Train Images (~1.5GB)"
        )
        # 既存のWIDER_trainを使用するため、解凍先はdownloadsに
        if not self.dry_run and train_zip.exists() and not (self.downloads_dir / "WIDER_train").exists():
            self.extract_zip(train_zip, self.downloads_dir, "WIDERFace Train Images")
        
        # Validation Images (~345MB)
        val_zip = self.downloads_dir / "WIDER_val.zip"
        self.download_file(
            self.wider_urls["val_images"],
            val_zip,
            "WIDERFace Val Images (~345MB)"
        )
        if not self.dry_run and val_zip.exists() and not (self.downloads_dir / "WIDER_val").exists():
            self.extract_zip(val_zip, self.downloads_dir, "WIDERFace Val Images")
        
        # Annotations (~3MB)
        anno_zip = self.downloads_dir / "wider_face_split.zip"
        self.download_file(
            self.wider_urls["annotations"],
            anno_zip,
            "WIDERFace Annotations (~3MB)"
        )
        if not self.dry_run and anno_zip.exists() and not (self.downloads_dir / "wider_face_split").exists():
            self.extract_zip(anno_zip, self.downloads_dir, "WIDERFace Annotations")
        
        print("\n✓ WIDERFaceデータセット準備完了")
        print(f"  場所: {self.downloads_dir}")
    
    def print_summary(self):
        """ダウンロードサマリーを表示"""
        print("\n" + "=" * 70)
        print("ダウンロード完了サマリー")
        print("=" * 70)
        
        print("\n期待されるディレクトリ構造:")
        print("""
downloads/
├── coco/
│   ├── train2017/          # COCO train images
│   ├── val2017/            # COCO val images
│   ├── annotations/        # COCO annotations (JSON)
│   ├── train2017.zip
│   ├── val2017.zip
│   └── annotations_trainval2017.zip
├── WIDER_train/
│   └── images/             # WIDERFace train images
├── WIDER_val/
│   └── images/             # WIDERFace val images
├── wider_face_split/
│   ├── wider_face_train_bbx_gt.txt
│   ├── wider_face_val_bbx_gt.txt
│   └── ...
├── WIDER_train.zip
├── WIDER_val.zip
└── wider_face_split.zip
        """)
        
        print("\n次のステップ:")
        print("1. prepare_combined_dataset.py を更新してCOCO全体に対応")
        print("2. WIDERFace変換機能を実装")
        print("3. 統合データセットを作成")


def main():
    parser = argparse.ArgumentParser(
        description="COCO・WIDERFaceデータセットをダウンロード",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 全てダウンロード
  python scripts/download_datasets.py --all
  
  # COCOのみ
  python scripts/download_datasets.py --coco
  
  # WIDERFaceのみ
  python scripts/download_datasets.py --wider
  
  # ドライラン（実際にはダウンロードしない）
  python scripts/download_datasets.py --all --dry-run
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="全てのデータセットをダウンロード"
    )
    parser.add_argument(
        "--coco",
        action="store_true",
        help="COCOデータセットのみダウンロード"
    )
    parser.add_argument(
        "--wider",
        action="store_true",
        help="WIDERFaceデータセットのみダウンロード"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ドライラン（実際にはダウンロードしない）"
    )
    
    args = parser.parse_args()
    
    # 引数チェック
    if not (args.all or args.coco or args.wider):
        parser.print_help()
        return
    
    downloader = DatasetDownloader(dry_run=args.dry_run)
    
    # ディレクトリ作成
    downloader.downloads_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 70)
    print("データセットダウンロードツール")
    print("=" * 70)
    
    if args.dry_run:
        print("\n⚠️  ドライランモード: 実際にはダウンロードしません\n")
    
    # ダウンロード実行
    if args.all or args.coco:
        downloader.download_coco()
    
    if args.all or args.wider:
        downloader.download_wider()
    
    # サマリー表示
    downloader.print_summary()
    
    print("\n" + "=" * 70)
    print("✓ 全ての処理が完了しました")
    print("=" * 70)


if __name__ == "__main__":
    main()
