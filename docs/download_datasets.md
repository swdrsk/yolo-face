# データセットダウンロードスクリプト

## 概要

`scripts/download_datasets.py` を作成しました。COCO全体データセット（~19GB）とWIDERFaceデータセット（~1.9GB）を`downloads/`フォルダにダウンロード・解凍するスクリプトです。

## 使用方法

### 全てダウンロード
```bash
python scripts/download_datasets.py --all
```

### COCOのみ
```bash
python scripts/download_datasets.py --coco
```

### WIDERFaceのみ
```bash
python scripts/download_datasets.py --wider
```

### ドライラン（実際にダウンロードしない）
```bash
python scripts/download_datasets.py --all --dry-run
```

## ダウンロード内容

### COCOデータセット（~19GB）
- **Train Images**: `train2017.zip` (~18GB) - 118,287枚の画像
- **Val Images**: `val2017.zip` (~1GB) - 5,000枚の画像
- **Annotations**: `annotations_trainval2017.zip` (~241MB) - JSON形式のアノテーション

### WIDERFaceデータセット（~1.9GB）
- **Train Images**: `WIDER_train.zip` (~1.5GB) - 顔検出用トレーニング画像
- **Val Images**: `WIDER_val.zip` (~345MB) - 顔検出用検証画像
- **Annotations**: `wider_face_split.zip` (~3MB) - テキスト形式のアノテーション

## ダウンロード後のディレクトリ構造

```
downloads/
├── coco/
│   ├── train2017/          # 118,287枚の画像
│   ├── val2017/            # 5,000枚の画像
│   ├── annotations/
│   │   ├── instances_train2017.json
│   │   ├── instances_val2017.json
│   │   └── ...
│   ├── train2017.zip
│   ├── val2017.zip
│   └── annotations_trainval2017.zip
├── WIDER_train/
│   └── images/
│       ├── 0--Parade/
│       ├── 1--Handshaking/
│       └── ...
├── WIDER_val/
│   └── images/
├── wider_face_split/
│   ├── wider_face_train_bbx_gt.txt
│   ├── wider_face_val_bbx_gt.txt
│   └── ...
├── WIDER_train.zip
├── WIDER_val.zip
└── wider_face_split.zip
```

## 特徴

- ✅ `curl`を使用した堅牢なダウンロード
- ✅ 既存ファイルの自動スキップ
- ✅ プログレスバー表示
- ✅ ドライランモード（テスト用）
- ✅ 自動解凍

## 注意事項

- ダウンロードには合計 **~21GB** のディスク容量が必要です
- ダウンロード時間は回線速度に依存します（100Mbps接続で約30分程度）
- 既にダウンロード済みのファイルは自動的にスキップされます

## 次のステップ

ダウンロード完了後:
1. `prepare_combined_dataset.py` を更新してCOCO全体に対応
2. WIDERFace変換機能を実装
3. person + face の統合データセットを作成
