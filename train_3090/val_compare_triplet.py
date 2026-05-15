from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def load_model(weights_path: Path, device: int) -> YOLO:
    model = YOLO(str(weights_path))
    model.to(device)
    return model


def predict_image(model: YOLO, image_path: Path, device: int, imgsz: int, conf: float) -> np.ndarray:
    result = model.predict(source=str(image_path), imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    return result.plot()


def build_triplet(images: list[np.ndarray]) -> np.ndarray:
    heights = [image.shape[0] for image in images]
    widths = [image.shape[1] for image in images]
    if len(set(heights)) != 1 or len(set(widths)) != 1:
        raise ValueError("Three visualization images must have identical shapes before concatenation.")
    return np.concatenate(images, axis=1)


def sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly sample 10 VisDrone val images and compare three YOLO11 runs.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling.")
    parser.add_argument("--count", type=int, default=10, help="Number of validation images to sample.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index used for inference.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for visualization.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the 10 comparison folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    candidate_roots = [repo_root / "VisDrone", Path("/root/datasets/VisDrone")]
    dataset_root = next((root for root in candidate_roots if (root / "images" / "val").exists()), None)

    if dataset_root is None:
        raise SystemExit("未找到 VisDrone 数据集目录，请把数据集放到仓库根目录 VisDrone/ 或 /root/datasets/VisDrone/。")

    val_dir = dataset_root / "images" / "val"

    if args.seed is not None:
        random.seed(args.seed)

    image_paths = sorted(
        [*val_dir.glob("*.jpg"), *val_dir.glob("*.jpeg"), *val_dir.glob("*.png")]
    )
    if len(image_paths) < args.count:
        raise SystemExit(f"验证集图片数量不足，当前只有 {len(image_paths)} 张，无法抽取 {args.count} 张。")

    selected_images = random.sample(image_paths, args.count)

    output_dir = args.output_dir or (repo_root / "runs" / "detect" / "VisDrone_Thesis_200e" / "val_compare_triplet_10")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_specs = [
        ("baseline", repo_root / "runs" / "detect" / "VisDrone_Thesis_200e" / "yolo11n_baseline_scratch_200e2" / "weights" / "best.pt"),
        ("baseline+p2+cbam+wiou", repo_root / "runs" / "detect" / "VisDrone_Thesis_200e" / "yolo11n_p2_cbam_wiou_multi_scratch_200e" / "weights" / "best.pt"),
        ("baseline+p2+cbam", repo_root / "runs" / "detect" / "VisDrone_Thesis_200e" / "yolo11n_p2_cbam_multi_scratch_200e" / "weights" / "best.pt"),
    ]

    for _, weights_path in model_specs:
        if not weights_path.exists():
            raise SystemExit(f"未找到权重文件: {weights_path}")

    models = [(name, load_model(weights_path, args.device)) for name, weights_path in model_specs]

    for index, image_path in enumerate(selected_images, start=1):
        sample_dir = output_dir / f"{index:02d}_{image_path.stem}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(image_path, sample_dir / image_path.name)

        result_images = []
        for model_name, model in models:
            result_image = predict_image(model, image_path, args.device, args.imgsz, args.conf)
            result_images.append(result_image)
            cv2.imwrite(str(sample_dir / f"{sanitize_name(model_name)}.jpg"), result_image)

        triplet_image = build_triplet(result_images)
        cv2.imwrite(str(sample_dir / "compare.jpg"), triplet_image)


if __name__ == "__main__":
    main()