#!/usr/bin/env -S uv run python
"""
データセットを間引いて小さくする（動作確認用）

使用方法:
    # デフォルト設定（各クラス20枚）
    uv run python scripts/subsample_dataset.py
    
    # カスタム枚数を指定
    uv run python scripts/subsample_dataset.py --samples 50
    
    # 特定のデータセットを指定
    uv run python scripts/subsample_dataset.py --input datasets/person_face --output datasets/person_face_small
"""

import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict


def subsample_dataset(
    input_dir,
    output_dir,
    samples_per_class=20,
    seed=42
):
    """
    データセットを間引いて小さくする
    
    Args:
        input_dir: 入力データセットディレクトリ
        output_dir: 出力データセットディレクトリ
        samples_per_class: 各クラスのサンプル数
        seed: 乱数シード
    """
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    print("=" * 70)
    print("データセット間引き処理")
    print("=" * 70)
    print(f"\n入力: {input_dir}")
    print(f"出力: {output_dir}")
    print(f"各クラスのサンプル数: {samples_per_class}枚")
    print()
    
    # 乱数シード設定
    random.seed(seed)
    
    # ラベルファイルを走査してクラスごとに分類
    print("ラベルファイルを解析中...")
    class_files = defaultdict(list)  # {class_id: [(label_file, image_file), ...]}
    
    for split in ["train", "val"]:
        labels_dir = input_dir / "labels" / split
        images_dir = input_dir / "images" / split
        
        if not labels_dir.exists():
            print(f"⚠ {labels_dir} が見つかりません。スキップします。")
            continue
        
        for label_file in labels_dir.glob("*.txt"):
            image_file = images_dir / f"{label_file.stem}.jpg"
            
            # 画像ファイルが存在しない場合はスキップ
            if not image_file.exists():
                continue
            
            # ラベルファイルを読み込んでクラスIDを取得
            with open(label_file, 'r') as f:
                lines = f.readlines()
                if not lines:
                    continue
                
                # 最初の行のクラスIDを使用（複数ある場合は最初のものを使用）
                class_id = int(lines[0].split()[0])
                class_files[class_id].append((label_file, image_file, split))
    
    # 統計情報を表示
    print("\n元データセットの統計:")
    class_names = {0: "person", 1: "face"}
    for class_id, files in sorted(class_files.items()):
        class_name = class_names.get(class_id, f"class_{class_id}")
        print(f"  {class_name} (class {class_id}): {len(files)}枚")
    
    # 各クラスからサンプリング
    print(f"\n各クラスから{samples_per_class}枚をランダムサンプリング中...")
    sampled_files = {}
    for class_id, files in class_files.items():
        if len(files) <= samples_per_class:
            # サンプル数が指定数以下の場合は全て使用
            sampled_files[class_id] = files
            class_name = class_names.get(class_id, f"class_{class_id}")
            print(f"  {class_name}: {len(files)}枚（全て使用）")
        else:
            # ランダムサンプリング
            sampled_files[class_id] = random.sample(files, samples_per_class)
            class_name = class_names.get(class_id, f"class_{class_id}")
            print(f"  {class_name}: {samples_per_class}枚")
    
    # 出力ディレクトリを作成
    print(f"\n出力ディレクトリを作成中: {output_dir}")
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # ファイルをコピー
    print("\nファイルをコピー中...")
    copied_count = defaultdict(int)
    
    for class_id, files in sampled_files.items():
        class_name = class_names.get(class_id, f"class_{class_id}")
        
        for label_file, image_file, split in files:
            # ラベルファイルをコピー
            dest_label = output_dir / "labels" / split / label_file.name
            shutil.copy2(label_file, dest_label)
            
            # 画像ファイルをコピー
            dest_image = output_dir / "images" / split / image_file.name
            shutil.copy2(image_file, dest_image)
            
            copied_count[class_id] += 1
    
    # data.yamlを作成
    print("\ndata.yamlを作成中...")
    data_yaml = output_dir / "data.yaml"
    
    with open(data_yaml, 'w') as f:
        f.write(f"# Person + Face間引きデータセット（動作確認用、各クラス{samples_per_class}枚程度）\n")
        f.write(f"path: {output_dir.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("\n")
        f.write("# クラス定義\n")
        f.write("nc: 2\n")
        f.write("names:\n")
        f.write("  0: person\n")
        f.write("  1: face\n")
        f.write("\n")
        f.write("# 統計情報\n")
        for class_id in sorted(copied_count.keys()):
            class_name = class_names.get(class_id, f"class_{class_id}")
            f.write(f"# {class_name}: {copied_count[class_id]}件\n")
        f.write("\n")
    
    print(f"✓ data.yamlを作成: {data_yaml}")
    
    # 最終統計
    print("\n" + "=" * 70)
    print("✓ データセット間引き完了！")
    print("=" * 70)
    print(f"\n出力先: {output_dir}")
    print("\n統計:")
    total = 0
    for class_id in sorted(copied_count.keys()):
        class_name = class_names.get(class_id, f"class_{class_id}")
        count = copied_count[class_id]
        print(f"  {class_name}: {count}枚")
        total += count
    print(f"  合計: {total}枚")
    
    print(f"\n次のコマンドでトレーニングを開始できます:")
    print(f"uv run python scripts/train_yolo.py --data {data_yaml} --freeze 10 --epochs 10")


def main():
    parser = argparse.ArgumentParser(
        description="データセットを間引いて小さくする（動作確認用）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルト設定（各クラス20枚）
  uv run python scripts/subsample_dataset.py
  
  # カスタム枚数を指定
  uv run python scripts/subsample_dataset.py --samples 50
  
  # 特定のデータセットを指定
  uv run python scripts/subsample_dataset.py --input datasets/person_face --output datasets/person_face_small
        """
    )
    
    # スクリプトの親ディレクトリ（yolo-face/）を取得
    script_dir = Path(__file__).parent.parent.resolve()
    default_input = script_dir / "datasets" / "person_face"
    default_output = script_dir / "datasets" / "person_face_small"
    
    parser.add_argument(
        "--input",
        type=str,
        default=str(default_input),
        help=f"入力データセットディレクトリ (デフォルト: {default_input})"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_output),
        help=f"出力データセットディレクトリ (デフォルト: {default_output})"
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="各クラスのサンプル数 (デフォルト: 20)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード (デフォルト: 42)"
    )
    
    args = parser.parse_args()
    
    # 入力ディレクトリの存在確認
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ エラー: 入力データセットが見つかりません: {input_path}")
        print("\n次のコマンドでデータセットを作成してください:")
        print("uv run python scripts/prepare_combined_dataset.py --dataset coco8")
        return
    
    # 間引き処理を実行
    subsample_dataset(
        input_dir=input_path,
        output_dir=args.output,
        samples_per_class=args.samples,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
