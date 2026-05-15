"""
扫描 runs/detect/VisDrone_Thesis_150e 下所有实验的 best.pt，
加载后跑验证集，输出 P/R/mAP50/mAP50-95 汇总表。

用法:
    python eval_all.py

支持：把其他机器跑的结果（整个实验文件夹）复制到
      runs/detect/VisDrone_Thesis_150e/ 下，脚本会自动识别。
"""

import os
import glob
from pathlib import Path
import torch
from ultralytics import YOLO

# ============================================================
# 配置
# ============================================================
PROJECT_DIR = "runs/detect/VisDrone_Thesis_200e"
DATA = "VisDrone.yaml"
IMGSZ = 640
BATCH = 16

# 自动选择 device：有 GPU 用 0，没有就用 cpu
device = "0" if torch.cuda.is_available() else "cpu"


def find_experiments(project_dir: str):
    """扫描项目目录下所有包含 weights/best.pt 的实验文件夹。"""
    pattern = os.path.join(project_dir, "*", "weights", "best.pt")
    pts = sorted(glob.glob(pattern))
    experiments = []
    for pt in pts:
        exp_name = Path(pt).parent.parent.name  # weights/best.pt 的父目录的父目录
        experiments.append((exp_name, pt))
    return experiments


def run_val(pt_path: str):
    """加载 best.pt 跑验证集，返回指标字典。"""
    print(f"\n[Eval] {pt_path}")
    model = YOLO(pt_path)
    metrics = model.val(
        data=DATA,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        split="val",
        verbose=False,
    )
    return {
        "P": float(metrics.box.mp),
        "R": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }


if __name__ == "__main__":
    print(f"Device: {device}")
    exps = find_experiments(PROJECT_DIR)
    print(f"发现实验: {len(exps)} 个\n")

    results = {}
    for name, pt in exps:
        try:
            res = run_val(pt)
            results[name] = res
            print(f"  -> P={res['P']:.3f}  R={res['R']:.3f}  mAP50={res['mAP50']:.3f}  mAP50-95={res['mAP50_95']:.3f}")
        except Exception as e:
            print(f"  -> ERROR: {e}")

    # 打印汇总表
    print("\n" + "=" * 85)
    print(f"{'实验名':<35s} {'P':>8s} {'R':>8s} {'mAP50':>8s} {'mAP50-95':>10s}")
    print("-" * 85)
    for name in sorted(results.keys()):
        r = results[name]
        print(f"{name:<35s} {r['P']:>8.3f} {r['R']:>8.3f} {r['mAP50']:>8.3f} {r['mAP50_95']:>10.3f}")
    print("=" * 85)
