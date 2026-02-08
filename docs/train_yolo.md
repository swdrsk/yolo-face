# YOLOトレーニングスクリプト

統合データセット（person + face）を使用してYOLOモデルをトレーニングするスクリプトです。

## クイックスタート（初めての方）

ゼロから学習までの最短手順：

```bash
# 1. uvのインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 仮想環境の作成と依存関係のインストール
uv venv
uv pip sync requirements.txt

# 3. データセットのダウンロード（COCO8推奨、数分）
uv run python scripts/download_datasets.py --dataset coco8

# 4. データセットの統合・準備（YOLO形式に変換）
uv run python scripts/prepare_combined_dataset.py --dataset coco8

# 5. （オプション）テスト用の小さいデータセット作成（23枚、高速）
uv run python scripts/subsample_dataset.py --samples 20

# 6A. 小さいデータセットで動作確認（約1分、推奨）
uv run python scripts/train_yolo.py --data datasets/person_face_small/data.yaml --freeze 10 --epochs 5 --batch 4 --imgsz 320

# 6B. または通常データセットでトレーニング（Freeze学習、推奨）
uv run python scripts/train_yolo.py --freeze 10 --epochs 30

# 7. モデルエクスポート
uv run python scripts/export_model.py --weights runs/detect/train/weights/best.pt
```

## 前提条件

### 1. 依存関係のインストール

**推奨Pythonバージョン: 3.10 〜 3.12**
（※3.13以上の最新バージョンはPyTorch等のライブラリが未対応の場合があります）

uvを使用している場合（推奨）：

```bash
# uvのインストール（未インストールの場合）
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 仮想環境の作成（Python 3.12を指定）
uv venv --python 3.12

# 依存関係のインストール
uv pip sync requirements.txt
```

またはpipを使用：

```bash
pip install -r requirements.txt
```

### 2. データセットのダウンロード

データセットを自動でダウンロードします：

```bash
# COCO8 + WIDERFace（テスト用、推奨）
uv run python scripts/download_datasets.py --dataset coco8

# または COCO全体 + WIDERFace（本番用、大容量）
uv run python scripts/download_datasets.py --dataset coco
```

**注意**: COCO全体データセットは数十GB以上のサイズになります。最初はCOCO8で試すことを推奨します。

### 3. データセットの統合・準備

ダウンロードしたデータセットをYOLO形式に変換して統合します：

```bash
# COCO8 + WIDERFaceを統合（テスト用）
uv run python scripts/prepare_combined_dataset.py --dataset coco8

# または COCO全体 + WIDERFaceを統合（本番用）
uv run python scripts/prepare_combined_dataset.py --dataset coco
```

これにより、`datasets/person_face/`ディレクトリに統合データセットが作成されます。

### 4. （オプション）動作確認用の小さいデータセット作成

学習速度を確認するために、小さいデータセットを作成できます：

```bash
# 各クラス20枚程度にサンプリング（高速テスト用）
uv run python scripts/subsample_dataset.py --samples 20

# 出力: datasets/person_face_small/
```

### 5. ラベル自動補正（擬似ラベル付与）

「身体のみ」または「顔のみ」のデータセットに足りないラベルを自動補完します。

#### 補正用モデルの準備
補正に使用する高精度なモデルをダウンロードします：

```bash
# 顔検出用モデル (akanametov/yolo-face)
curl -L -o face-yolo11m.pt https://github.com/akanametov/yolo-face/releases/download/v1.0/yolov11m-face.pt

# 人検出用モデル (Ultralytics)
curl -L -o yolo11m.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt
```

#### 実行
```bash
# 小規模セットでテスト
uv run python scripts/augment_labels.py --dataset datasets/person_face_small

# 本番データセット全体に適用
uv run python scripts/augment_labels.py --dataset datasets/person_face
```

---

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

## トレーニングログの読み方

トレーニング中、ターミナルには以下のような進捗が表示されます。**Loss（損失）の数値は低いほど優秀です。**

| 項目 | 意味 | 理想的な傾向 |
| :--- | :--- | :--- |
| **box_loss** | 枠の位置のズレ。正解と予測の重なり具合。 | 学習が進むにつれ減少 |
| **cls_loss** | クラス分類のミス。物体が何かを当てる精度。 | 学習が進むにつれ減少 |
| **dfl_loss** | 枠の境界線の精密さ。 | 学習が進むにつれ減少 |
| **mAP50** | 精度。0.5（50%）以上の重なりで正解とした時のスコア。 | **高いほど良い (1.0が最高)** |
| **mAP50-95** | より厳しい基準での平均精度。 | **高いほど良い (1.0が最高)** |

これらのLoss（誤差）が右肩下がりに減少していれば、学習は順調です。

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

### モデル保存先

トレーニング結果は以下のディレクトリに保存されます：

**デフォルト（`--name`未指定時）**:
```
runs/detect/train/
```

**カスタム名指定時** (`--name train_custom`の場合):
```
runs/detect/train_custom/
```

**プロジェクトとカスタム名指定時** (`--project my_project --name experiment1`の場合):
```
my_project/experiment1/
```

### ディレクトリ構造

```
runs/detect/train/              # 保存先ディレクトリ（--nameで変更可能）
├── weights/                    # モデル重みファイル
│   ├── best.pt                 # 最良モデル（validation精度最高）
│   └── last.pt                 # 最終エポックのモデル
│
**どちらを使うべき？**
- **学習の「再開」**: フリーズやエラーで止まった学習を「続きから」やり直す場合は **`last.pt`** を使います。
- **後続の「フェーズ移行」**: 第1段階（Headのみ）が終わり、第2段階（全体）へ進むような場合は、最も精度の高い **`best.pt`** を土台にするのが最適です。

├── results.csv                 # トレーニング結果（CSV）
├── results.png                 # トレーニング結果グラフ
├── confusion_matrix.png        # 混同行列
├── F1_curve.png               # F1スコア曲線
├── P_curve.png                # Precision曲線
├── R_curve.png                # Recall曲線
└── PR_curve.png               # Precision-Recall曲線
```

### モデルエクスポート

学習済みモデルを他の形式にエクスポートできます：

```bash
# TorchScript形式（デフォルト）
uv run python scripts/export_model.py --weights runs/detect/train/weights/best.pt

# ONNX形式（広くサポート）vsv
uv run python scripts/export_model.py --weights runs/detect/train/weights/best.pt --format onnx

# Core ML形式（iOS/macOS用）
uv run python scripts/export_model.py --weights runs/detect/train/weights/best.pt --format coreml

# 複数形式を一度にエクスポート
uv run python scripts/export_model.py --weights runs/detect/train/weights/best.pt --format onnx torchscript coreml
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

### Windows でのインストールエラー (unknown compiler)

Windowsで `uv pip sync` 実行時に `ERROR unknown compiler(s)` と表示される場合、C++コンパイラが不足しています。

**解決策1（推奨）**:
[Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) をインストールし、セットアップ内で **「C++ によるデスクトップ開発」** を選択してください。

**解決策2（バイナリ強制）**:
コンパイルを避けてビルド済みバイナリのみをインストールします：
```bash
uv pip install --only-binary :all: -r requirements.txt
```

### Windows で GPU (CUDA) が検出されない

Windows環境でNVIDIA GPUが認識されない（`CUDA available: False`）場合は、CPU版のPyTorchがインストールされている可能性があります。以下のコマンドでCUDA対応版を強制インストールしてください：

```bash
# CUDA 12.4対応版をインストールする場合
uv pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
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
