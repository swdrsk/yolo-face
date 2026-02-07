# YOLOトレーニングスクリプト

統合データセット（person + face）を使用してYOLOモデルをトレーニングするスクリプトです。

## 前提条件

### 1. 依存関係のインストール

uvを使用している場合（推奨）：

```bash
uv pip sync requirements.txt
```

またはpipを使用：

```bash
pip install -r requirements.txt
```

### 2. データセットの準備

先にデータセットを作成してください：

```bash
# COCO8 + WIDERFace（テスト用）
uv run python scripts/prepare_combined_dataset.py --dataset coco8

# または COCO全体 + WIDERFace（本番用）
uv run python scripts/prepare_combined_dataset.py --dataset coco
```

## 使用方法

### 基本的な使用

デフォルト設定（**YOLO26m**、50エポック）でトレーニング：

```bash
uv run python scripts/train_yolo.py
```

### 短時間学習（Freeze推奨）

**Freeze学習**は、バックボーン部分をフリーズしてヘッド部分のみを学習することで、学習時間を大幅に短縮できます：

```bash
# Freeze学習（推奨、30エポック程度で十分）
uv run python scripts/train_yolo.py --freeze 10 --epochs 30
```

### カスタム設定

#### より大きなモデルで長時間トレーニング（通常学習）

```bash
uv run python scripts/train_yolo.py --model yolo26l --epochs 100
```

#### バッチサイズと画像サイズの調整

```bash
uv run python scripts/train_yolo.py --batch 32 --imgsz 640
```

#### 事前学習済みモデルから継続

```bash
uv run python scripts/train_yolo.py --weights runs/detect/train/weights/best.pt --epochs 50
```

#### デバイス指定

```bash
# CPU使用
uv run python scripts/train_yolo.py --device cpu

# GPU 0を使用
uv run python scripts/train_yolo.py --device 0

# 複数GPU使用
uv run python scripts/train_yolo.py --device 0,1
```

## パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `--data` | `datasets/person_face/data.yaml` | データセット設定ファイルのパス |
| `--model` | `yolo26m` | YOLOモデル（26シリーズ/11シリーズ、n/s/m/l/x） |
| `--weights` | `None` | 事前学習済み重みファイルのパス |
| `--epochs` | `50` | トレーニングエポック数 |
| `--batch` | `16` | バッチサイズ |
| `--imgsz` | `640` | 入力画像サイズ |
| `--device` | `""` | 使用デバイス（自動/cpu/0/0,1等） |
| `--project` | `runs/detect` | プロジェクトディレクトリ |
| `--name` | `train` | 実験名 |
| `--exist-ok` | `False` | 既存ディレクトリを上書き |
| `--no-pretrained` | `False` | COCO事前学習済みモデルを使用しない |
| `--freeze` | `None` | フリーズするレイヤー数（10推奨、0で無効） |

## モデルサイズ

### YOLO26シリーズ（推奨、デフォルト）

| モデル | パラメータ数 | 速度 | 精度 | 推奨用途 |
|--------|------------|------|------|----------|
| yolo26n | ~3M | 最速 | 低 | 軽量デバイス |
| yolo26s | ~11M | 速い | 中 | 一般用途 |
| **yolo26m** | ~25M | 中 | 高 | **バランス型（デフォルト）** |
| yolo26l | ~43M | 遅い | 高 | 高精度要求 |
| yolo26x | ~56M | 最遅 | 最高 | 最高精度要求 |

### YOLO11シリーズ（互換性用）

| モデル | パラメータ数 | 速度 | 精度 |
|--------|------------|------|------|
| yolo11n | 2.6M | 最速 | 低 |
| yolo11s | 9.4M | 速い | 中 |
| yolo11m | 20.1M | 中 | 高 |
| yolo11l | 25.3M | 遅い | 高 |
| yolo11x | 56.9M | 最遅 | 最高 |

## Freeze学習について

### Freeze学習とは？

**Freeze学習**は、事前学習済みモデルのバックボーン部分（特徴抽出層）をフリーズし、ヘッド部分（検出層）のみを学習する手法です。

**メリット**:
- ⚡ **学習時間が大幅に短縮**（通常の1/3～1/5程度）
- 💾 **GPU/CPUメモリ使用量が削減**
- 📊 **少ないエポックで良好な結果**（30エポック程度で十分）

**使用例**:
```bash
# Freeze学習（推奨）
uv run python scripts/train_yolo.py --freeze 10 --epochs 30
```

**推奨設定**:
- `--freeze 10`: 最初の10レイヤーをフリーズ（バックボーン固定）
- `--epochs 30`: 30エポック程度で十分な精度

## 出力

トレーニング結果は以下のディレクトリに保存されます：

```
runs/detect/train/
├── weights/
│   ├── best.pt       # 最良モデル
│   └── last.pt       # 最終モデル
├── results.png       # トレーニング結果グラフ
├── confusion_matrix.png
├── F1_curve.png
├── P_curve.png
├── R_curve.png
└── PR_curve.png
```

## トレーニング後の推論

トレーニングが完了したら、ベストモデルを使って推論できます：

```bash
# 画像で推論
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/image.jpg

# 動画で推論
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/video.mp4

# Webカメラ
yolo detect predict model=runs/detect/train/weights/best.pt source=0
```

## トラブルシューティング

### メモリ不足エラー

バッチサイズを減らしてください：

```bash
uv run python scripts/train_yolo.py --batch 8
```

または画像サイズを小さくしてください：

```bash
uv run python scripts/train_yolo.py --imgsz 416
```

### CUDA/GPU エラー

CPUを使用してください：

```bash
uv run python scripts/train_yolo.py --device cpu
```

## 推奨設定

### クイックスタート（COCO8データセット、Freeze学習）

最も高速で効率的な学習：

```bash
uv run python scripts/train_yolo.py --freeze 10 --epochs 30 --batch 16
```

### 開発/テスト用（COCO8データセット、通常学習）

```bash
uv run python scripts/train_yolo.py --model yolo26m --epochs 50 --batch 16
```

### 本番用（COCO全体データセット、Freeze学習）

大規模データセットでも短時間で学習：

```bash
uv run python scripts/train_yolo.py --freeze 10 --epochs 30 --batch 32 --imgsz 640
```

### 本番用（COCO全体データセット、通常学習）

より高精度が必要な場合：

```bash
uv run python scripts/train_yolo.py --model yolo26l --epochs 100 --batch 32 --imgsz 640
```

## 参考

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [YOLO11 Models](https://docs.ultralytics.com/models/yolo11/)
