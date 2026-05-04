from ultralytics import YOLO
import cv2
import os

# ====== 修改这里 ======
YOLO11N_STANDARD_WEIGHTS = "/workspace/yolo11n.pt"
BASELINE_WEIGHTS = "/workspace/runs/detect/VisDrone_Thesis/yolo11n_baseline/weights/best.pt"
P2_WEIGHTS       = "/workspace/runs/detect/VisDrone_Thesis/yolo11n_p2/weights/best.pt"
IMAGE_PATH       = "/workspace/test5.png"
SAVE_DIR         = "/workspace/compare_results"
# ======================

os.makedirs(SAVE_DIR, exist_ok=True)

def run_inference(model_path, name):
    print(f"\n🚀 Running {name}...")

    model = YOLO(model_path)

    results = model.predict(
        source=IMAGE_PATH,
        imgsz=640,        # 和训练一致
        conf=0.3,
        iou=0.3,
        save=False,
        verbose=False
    )

    r = results[0]

    # 获取检测框数量
    num_boxes = len(r.boxes)
    print(f"{name} 检测到目标数量: {num_boxes}")

    # 画图
    img = cv2.imread(IMAGE_PATH)

    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    save_path = os.path.join(SAVE_DIR, f"{name}.jpg")
    cv2.imwrite(save_path, img)

    print(f"结果已保存: {save_path}")

    return num_boxes


if __name__ == "__main__":
    baseline_count = run_inference(BASELINE_WEIGHTS, "baseline")
    p2_count       = run_inference(P2_WEIGHTS, "p2")
    YOLO11N_STANDARD_count = run_inference(YOLO11N_STANDARD_WEIGHTS, "YOLO11N_STANDARD")

    print("\n======================")
    print(f"baseline 检测数量: {baseline_count}")
    print(f"P2 模型检测数量: {p2_count}")
    print(f"YOLO11N_STANDARD 检测数量: {YOLO11N_STANDARD_count}")

    if p2_count > baseline_count:
        print("👉 P2 对小目标检测更强（检测数量更多）")
    else:
        print("👉 baseline 与 P2 差异不明显或 baseline 更强")