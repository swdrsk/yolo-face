#!/usr/bin/env -S uv run python
"""
YOLOモデルをエクスポート（ONNX、TorchScript、CoreMLなど）

使用方法:
    # デフォルト設定（ONNX形式）
    uv run python scripts/export_model.py --weights runs/detect/train/weights/best.pt
    
    # 複数形式でエクスポート
    uv run python scripts/export_model.py --weights runs/detect/train/weights/best.pt --format onnx torchscript coreml
    
    # 画像サイズを指定
    uv run python scripts/export_model.py --weights runs/detect/train/weights/best.pt --imgsz 640
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def export_model(
    weights,
    formats=None,
    imgsz=640,
    half=False,
    int8=False,
    device="cpu",
    simplify=True
):
    """
    YOLOモデルをエクスポート
    
    Args:
        weights: 重みファイルのパス
        formats: エクスポート形式のリスト
        imgsz: 入力画像サイズ
        half: FP16量子化（GPU only）
        int8: INT8量子化
        device: デバイス（cpu/cuda/mps）
        simplify: ONNXモデルを簡略化
    """
    
    weights_path = Path(weights)
    
    if not weights_path.exists():
        print(f"✗ エラー: 重みファイルが見つかりません: {weights_path}")
        return
    
    print("=" * 70)
    print("YOLOモデルエクスポート")
    print("=" * 70)
    print(f"\n重みファイル: {weights_path}")
    print(f"エクスポート形式: {', '.join(formats)}")
    print(f"画像サイズ: {imgsz}")
    print(f"デバイス: {device}")
    print()
    
    # モデルをロード
    print("モデルをロード中...")
    model = YOLO(str(weights_path))
    
    # 各形式でエクスポート
    exported_files = []
    
    for fmt in formats:
        print(f"\n{'='*70}")
        print(f"{fmt.upper()}形式でエクスポート中...")
        print(f"{'='*70}")
        
        try:
            # エクスポート実行
            export_path = model.export(
                format=fmt,
                imgsz=imgsz,
                half=half,
                int8=int8,
                device=device,
                simplify=simplify if fmt == "onnx" else False
            )
            
            exported_files.append((fmt, export_path))
            print(f"✓ {fmt.upper()}エクスポート完了: {export_path}")
            
        except Exception as e:
            print(f"✗ {fmt.upper()}エクスポート失敗: {e}")
    
    # 結果サマリー
    print("\n" + "=" * 70)
    print("✓ エクスポート完了！")
    print("=" * 70)
    print("\nエクスポートされたファイル:")
    for fmt, path in exported_files:
        print(f"  [{fmt.upper()}] {path}")
    
    # 利用例を表示
    print("\n使用例:")
    for fmt, path in exported_files:
        if fmt == "onnx":
            print(f"\n  # ONNX Runtime:")
            print(f"  import onnxruntime as ort")
            print(f"  session = ort.InferenceSession('{path}')")
            
        elif fmt == "torchscript":
            print(f"\n  # TorchScript:")
            print(f"  import torch")
            print(f"  model = torch.jit.load('{path}')")
            
        elif fmt == "coreml":
            print(f"\n  # Core ML (iOS/macOS):")
            print(f"  import coremltools as ct")
            print(f"  model = ct.models.MLModel('{path}')")


def main():
    parser = argparse.ArgumentParser(
        description="YOLOモデルをエクスポート",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルト設定（ONNX形式）
  uv run python scripts/export_model.py --weights runs/detect/train/weights/best.pt
  
  # 複数形式でエクスポート
  uv run python scripts/export_model.py --weights runs/detect/train/weights/best.pt \\
    --format onnx torchscript coreml
  
  # 画像サイズを指定
  uv run python scripts/export_model.py --weights runs/detect/train/weights/best.pt --imgsz 640
  
  # INT8量子化（軽量化）
  uv run python scripts/export_model.py --weights runs/detect/train/weights/best.pt --int8

エクスポート形式:
  - onnx: ONNX (推奨、広くサポート)
  - torchscript: TorchScript (PyTorch)
  - coreml: Core ML (iOS/macOS)
  - tflite: TensorFlow Lite (モバイル)
  - edgetpu: Edge TPU (Google Coral)
  - engine: TensorRT (NVIDIA GPU)
        """
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="重みファイルのパス（.pt）"
    )
    
    parser.add_argument(
        "--format",
        nargs="+",
        default=["torchscript"],
        choices=["onnx", "torchscript", "coreml", "tflite", "edgetpu", "engine", "saved_model"],
        help="エクスポート形式（複数指定可、デフォルト: torchscript）"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="入力画像サイズ（デフォルト: 640）"
    )
    
    parser.add_argument(
        "--half",
        action="store_true",
        help="FP16量子化（GPU only）"
    )
    
    parser.add_argument(
        "--int8",
        action="store_true",
        help="INT8量子化（モデルサイズ削減）"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="デバイス: cpu, cuda, mps（デフォルト: cpu）"
    )
    
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="ONNXモデルを簡略化しない"
    )
    
    args = parser.parse_args()
    
    export_model(
        weights=args.weights,
        formats=args.format,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        device=args.device,
        simplify=not args.no_simplify
    )


if __name__ == "__main__":
    main()
