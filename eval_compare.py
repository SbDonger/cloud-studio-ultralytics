from ultralytics import YOLO

# 全部实验的 best.pt 对比
weights = {
    "yolo11n_official": "/workspace/yolo11n.pt",
    "baseline": "/workspace/runs/detect/VisDrone_Thesis/yolo11n_baseline/weights/best.pt",
    "p2": "/workspace/runs/detect/VisDrone_Thesis/yolo11n_p2/weights/best.pt",
    "p2_wiou": "/workspace/runs/detect/VisDrone_Thesis/yolo11n_p2_wiou/weights/best.pt",
    "p2_cbam": "/workspace/runs/detect/VisDrone_Thesis/yolo11n_p2_cbam/weights/best.pt",
}

results = {}
for name, path in weights.items():
    print(f"\n{'='*50}")
    print(f"Evaluating: {name}")
    print('='*50)
    model = YOLO(path)
    metrics = model.val(data="VisDrone.yaml", imgsz=640, batch=4, device=0, verbose=False)
    results[name] = {
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "Precision": metrics.box.mp,
        "Recall": metrics.box.mr,
    }
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")

# 汇总对比表
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
header = f"{'Model':<18} {'mAP50':<10} {'mAP50-95':<10} {'P':<10} {'R':<10}"
print(header)
print("-" * 58)
for name, r in results.items():
    print(f"{name:<18} {r['mAP50']:<10.4f} {r['mAP50-95']:<10.4f} {r['Precision']:<10.4f} {r['Recall']:<10.4f}")
