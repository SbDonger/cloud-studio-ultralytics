from pathlib import Path
import subprocess
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    yolov7_root = repo_root / "yolov7"

    if not yolov7_root.exists():
        raise SystemExit(
            "未找到外部 YOLOv7 仓库。请先把 WongKinYiu/yolov7 克隆到仓库根目录下的 yolov7/，"
            "然后再运行这个脚本。"
        )

    command = [
        sys.executable,
        "train.py",
        "--img",
        "640",
        "--batch",
        "16",
        "--epochs",
        "200",
        "--data",
        str(repo_root / "VisDrone.yaml"),
        "--cfg",
        "cfg/training/yolov7-tiny.yaml",
        "--weights",
        "yolov7-tiny.pt",
        "--device",
        "0",
        "--project",
        str(repo_root / "runs" / "detect" / "VisDrone_Thesis_200e"),
        "--name",
        "yolov7tiny_baseline_scratch_200e",
    ]

    subprocess.run(command, cwd=yolov7_root, check=True)


if __name__ == "__main__":
    main()