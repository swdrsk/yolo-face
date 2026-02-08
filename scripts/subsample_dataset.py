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
    val_ratio=0.2,
    seed=42
):
    """
    データセットを間引いて小さくする
    
    Args:
        input_dir: 入力データセットディレクトリ
        output_dir: 出力データセットディレクトリ
        samples_per_class: 各クラスの合計サンプル数
        val_ratio: 検証データの割合
        seed: 乱数シード
    """
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    print("=" * 70)
    print("データセット間引き処理 (サンプリング & 再分割)")
    print("=" * 70)
    print(f"\n入力: {input_dir}")
    print(f"出力: {output_dir}")
    print(f"各クラスの合計サンプル数: {samples_per_class}枚")
    print(f"検証データの割合: {val_ratio * 100}%")
    print()
    
    # 乱数シード設定
    random.seed(seed)
    
    # 全てのラベルファイルを収集（元データセットの構成に寄らず全プールから抽出）
    print("元データセットから全ファイルを収集して解析中...")
    class_files = defaultdict(list)  # {class_id: [(label_file, image_file), ...]}
    
    for split in ["train", "val"]:
        labels_dir = input_dir / "labels" / split
        images_dir = input_dir / "images" / split
        
        if not labels_dir.exists():
            continue
        
        for label_file in labels_dir.glob("*.txt"):
            image_file = images_dir / f"{label_file.stem}.jpg"
            if not image_file.exists():
                image_file = images_dir / f"{label_file.stem}.png" # PNGも一応対応
            
            if not image_file.exists():
                continue
            
            with open(label_file, 'r') as f:
                lines = f.readlines()
                if not lines: continue
                class_id = int(lines[0].split()[0])
                class_files[class_id].append((label_file, image_file))
    
    # 各クラスから合計サンプル数を抽出
    sampled_files = {}
    for class_id, files in class_files.items():
        if len(files) <= samples_per_class:
            sampled_files[class_id] = files
        else:
            sampled_files[class_id] = random.sample(files, samples_per_class)
    
    # 出力ディレクトリを作成
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # ファイルをコピー（train/valに分配）
    print("\nサンプリングしたファイルを分配・コピー中...")
    copied_count = defaultdict(lambda: {"train": 0, "val": 0})
    class_names = {0: "person", 1: "face"}
    
    for class_id, files in sampled_files.items():
        # ファイルをシャッフル
        random.shuffle(files)
        
        # 検証データの件数を決定
        val_count = int(len(files) * val_ratio)
        if val_count == 0 and len(files) > 1: val_count = 1 # 最低1枚はvalへ
        
        val_files = files[:val_count]
        train_files = files[val_count:]
        
        class_name = class_names.get(class_id, f"class_{class_id}")
        print(f"  {class_name}: 合計 {len(files)}枚 (train: {len(train_files)}, val: {len(val_files)})")
        
        for split_files, split_name in [(train_files, "train"), (val_files, "val")]:
            for label_file, image_file in split_files:
                # コピー
                shutil.copy2(label_file, output_dir / "labels" / split_name / label_file.name)
                shutil.copy2(image_file, output_dir / "images" / split_name / image_file.name)
                copied_count[class_id][split_name] += 1
    
    # data.yamlを作成
    print("\ndata.yamlを作成中...")
    data_yaml = output_dir / "data.yaml"
    with open(data_yaml, 'w') as f:
        f.write(f"# Person + Face間引きデータセット (各クラス最大{samples_per_class}枚)\n")
        f.write(f"path: {output_dir.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("\nnc: 2\nnames:\n  0: person\n  1: face\n\n# 統計情報\n")
        for cid in sorted(copied_count.keys()):
            name = class_names.get(cid, f"class_{cid}")
            c = copied_count[cid]
            f.write(f"# {name}: train={c['train']}, val={c['val']} (計 {c['train']+c['val']})\n")
    
    print(f"✓ data.yamlを作成: {data_yaml}")
    
    # 最終統計
    print("\n" + "=" * 70)
    print("✓ データセット間引き・再分割完了！")
    print("=" * 70)
    print(f"\n出力先: {output_dir}")
    print("\n件数統計:")
    grand_total = 0
    for cid in sorted(copied_count.keys()):
        name = class_names.get(cid, f"class_{cid}")
        c = copied_count[cid]
        total = c['train'] + c['val']
        print(f"  {name:7}: {total:6}枚 (train: {c['train']:6}, val: {c['val']:6})")
        grand_total += total
    print(f"  {'合計':7}: {grand_total:6}枚")
    
    print(f"\n次のコマンドでトレーニングを開始できます:")
    print(f"uv run python scripts/train_yolo.py --data {data_yaml} --freeze 10 --epochs 30")


def main():
    parser = argparse.ArgumentParser(
        description="データセットを間引いて再分割する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルト（各20枚）
  uv run python scripts/subsample_dataset.py
  
  # 中規模セット作成（各1万枚、val 20%）
  uv run python scripts/subsample_dataset.py --samples 10000 --output datasets/person_face_middle
  
  # 特定の比率で検証データを分ける
  uv run python scripts/subsample_dataset.py --samples 5000 --val-ratio 0.1
        """
    )
    
    script_dir = Path(__file__).parent.parent.resolve()
    default_input = script_dir / "datasets" / "person_face"
    
    parser.add_argument("--input", type=str, default=str(default_input), help="入力データセットディレクトリ")
    parser.add_argument("--output", type=str, default=None, help="出力ディレクトリ (未指定時は枚数に応じた名前)")
    parser.add_argument("--samples", type=int, default=20, help="各クラスの合計サンプル数")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="検証データの割合 (0.0〜1.0)")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    
    args = parser.parse_args()
    
    # 出力先が未指定の場合、枚数に応じて自動生成
    if args.output is None:
        suffix = "small" if args.samples < 1000 else "middle" if args.samples < 50000 else "large"
        output_path = script_dir / "datasets" / f"person_face_{suffix}"
    else:
        output_path = Path(args.output)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ エラー: 入力データセットが見つかりません: {input_path}")
        return
    
    subsample_dataset(
        input_dir=input_path,
        output_dir=output_path,
        samples_per_class=args.samples,
        val_ratio=args.val_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
