#!/usr/bin/env -S uv run python
"""
統合データセット（person + face）を使用してYOLOモデルをトレーニング

使用方法:
    # デフォルト設定でトレーニング（YOLO26m、50エポック）
    uv run python scripts/train_yolo.py
    
    # カスタム設定
    uv run python scripts/train_yolo.py --model yolo26l --epochs 100 --batch 16
    
    # Freeze学習（短時間学習、バックボーンをフリーズしてヘッドのみ学習）
    uv run python scripts/train_yolo.py --freeze 10 --epochs 30
    
    # 事前学習済みモデルから
    uv run python scripts/train_yolo.py --weights runs/detect/train/weights/best.pt --epochs 50
    uv run python scripts/train_yolo.py --weights runs/detect/train/weights/best.pt --data datasets/person_face_small/data.yaml  --freeze 10 --epochs 10
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(
    data_yaml,
    model="yolo26m",
    weights=None,
    epochs=50,
    batch=16,
    imgsz=640,
    workers=8,
    device="",
    project="runs/detect",
    name="train",
    exist_ok=False,
    pretrained=True,
    freeze=None,
    resume=False,
    verbose=True
):
    """
    YOLOモデルをトレーニング
    
    Args:
        data_yaml: データセット設定ファイルのパス
        model: 使用するYOLOモデル（yolo26n, yolo26s, yolo26m, yolo26l, yolo26x, yolo11n等）
        weights: 事前学習済み重みファイルのパス（Noneの場合は公式の事前学習済みを使用）
        epochs: トレーニングエポック数
        batch: バッチサイズ
        imgsz: 入力画像サイズ
        workers: データロードのワーカー数（Windowsでエラーが出る場合は0を推奨）
        device: 使用デバイス（""=自動、"cpu", "0", "0,1"など）
        project: プロジェクトディレクトリ
        name: 実験名
        exist_ok: 既存のプロジェクトディレクトリを上書き
        pretrained: COCO事前学習済みモデルを使用するか
        freeze: フリーズするレイヤー数（Noneまたは0で無効、10推奨。短時間学習用）
        verbose: 詳細ログ出力
    """
    
    print("=" * 70)
    print("YOLOトレーニング開始")
    print("=" * 70)
    print(f"\nデータセット: {data_yaml}")
    print(f"モデル: {model}")
    print(f"エポック数: {epochs}")
    print(f"バッチサイズ: {batch}")
    print(f"画像サイズ: {imgsz}")
    print(f"デバイス: {device if device else '自動検出'}")
    if freeze is not None and freeze > 0:
        print(f"フリーズレイヤー数: {freeze}（短時間学習モード）")
    print()
    
    # モデルをロード
    if weights:
        print(f"カスタム重みをロード: {weights}")
        yolo_model = YOLO(weights)
    else:
        # 事前学習済みモデルまたは新規モデル
        model_name = f"{model}.pt" if pretrained else f"{model}.yaml"
        print(f"モデルをロード: {model_name}")
        yolo_model = YOLO(model_name)
    
    # トレーニング開始
    print("\n" + "=" * 70)
    print("トレーニング実行中...")
    print("=" * 70 + "\n")
    
    # トレーニングパラメータ準備
    train_params = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "workers": workers,
        "device": device,
        "project": project,
        "name": name,
        "exist_ok": exist_ok,
        "verbose": verbose,
        # その他の推奨設定
        "patience": 10,  # Early stopping patience
        "save": True,  # 重みを保存
        "save_period": 10,  # 10エポックごとに保存
        "plots": True,  # トレーニングプロットを作成
        "val": True,  # 検証を実行
        "resume": resume,  # 中断した場所から再開
    }
    
    # Freezeオプション（短時間学習用）
    if freeze is not None and freeze > 0:
        train_params["freeze"] = freeze
        print(f"\n⚡ Freezeモード有効: 最初の{freeze}レイヤーをフリーズ")
        print("   → バックボーンは固定、ヘッド部分のみ学習（学習時間短縮）\n")
    
    results = yolo_model.train(**train_params)
    
    print("\n" + "=" * 70)
    print("✓ トレーニング完了！")
    print("=" * 70)
    print(f"\n保存先: {results.save_dir}")
    print(f"最良モデル: {results.save_dir}/weights/best.pt")
    print(f"最終モデル: {results.save_dir}/weights/last.pt")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="YOLOモデルトレーニング（person + face統合データセット）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルト設定（YOLO26m、50エポック）
  uv run python scripts/train_yolo.py
  
  # Freeze学習（短時間学習、推奨）
  uv run python scripts/train_yolo.py --freeze 10 --epochs 30
  
  # より大きなモデルで長時間トレーニング
  uv run python scripts/train_yolo.py --model yolo26l --epochs 100
  
  # バッチサイズとワーカー数の指定
  uv run python scripts/train_yolo.py --batch 32 --workers 0
  
  # 事前学習済みモデルから継続
  uv run python scripts/train_yolo.py --weights runs/detect/train/weights/best.pt --epochs 50
  
  # CPU使用
  uv run python scripts/train_yolo.py --device cpu
  
  # 学習を中断した場所から再開
  uv run python scripts/train_yolo.py --resume
        """
    )
    
    # スクリプトの親ディレクトリ（yolo-face/）を取得
    script_dir = Path(__file__).parent.parent.resolve()
    default_data = script_dir / "datasets" / "person_face" / "data.yaml"
    
    parser.add_argument(
        "--data",
        type=str,
        default=str(default_data),
        help=f"データセット設定ファイルのパス (デフォルト: {default_data})"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26m",
        choices=["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x", 
                 "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
        help="使用するYOLOモデル (n=nano, s=small, m=medium, l=large, x=xlarge。デフォルト: yolo26m)"
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="事前学習済み重みファイルのパス（指定しない場合は公式の事前学習済みモデルを使用）"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="トレーニングエポック数 (デフォルト: 50)"
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="バッチサイズ (デフォルト: 16)"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="入力画像サイズ (デフォルト: 640)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="データロードのワーカー数 (Windowsでエラーが出る場合は0を指定。デフォルト: 8)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="使用デバイス: '' (自動), 'cpu', '0', '0,1,2,3' など (デフォルト: 自動)"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="プロジェクトディレクトリ (デフォルト: runs/detect)"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="実験名 (デフォルト: train)"
    )
    
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="既存のプロジェクトディレクトリを上書き"
    )
    
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="COCO事前学習済みモデルを使用しない（ランダム初期化）"
    )
    
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="フリーズするレイヤー数（短時間学習用。10推奨。0で無効）"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="中断した場所から学習を再開 (last.ptが必要)"
    )
    
    args = parser.parse_args()
    
    # データファイルの存在確認
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"✗ エラー: データ設定ファイルが見つかりません: {data_path}")
        print("\n次のコマンドでデータセットを作成してください:")
        print("uv run python scripts/prepare_combined_dataset.py --dataset coco8")
        return
    
    # トレーニング実行
    train(
        data_yaml=data_path,
        model=args.model,
        weights=args.weights,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        pretrained=not args.no_pretrained,
        freeze=args.freeze,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
