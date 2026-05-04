"""Evaluate P2 vs P2+WIoU best.pt on VisDrone val set."""
from ultralytics import YOLO

print("=" * 60)
print("Evaluating: P2 (CIoU, best.pt)")
print("=" * 60)
m1 = YOLO("/workspace/runs/detect/VisDrone_Thesis/yolo11n_p2/weights/best.pt")
r1 = m1.val(data="VisDrone.yaml", imgsz=640, batch=4, device="cpu", verbose=True)
print(f"\n>>> P2 (CIoU): mAP50={r1.box.map50:.4f}, mAP50-95={r1.box.map:.4f}")
print(f">>> Precision={r1.box.mp:.4f}, Recall={r1.box.mr:.4f}")

print("\n" + "=" * 60)
print("Evaluating: P2+WIoU (best.pt)")
print("=" * 60)
m2 = YOLO("/workspace/runs/detect/VisDrone_Thesis/yolo11n_p2_wiou/weights/best.pt")
r2 = m2.val(data="VisDrone.yaml", imgsz=640, batch=4, device="cpu", verbose=True)
print(f"\n>>> P2+WIoU: mAP50={r2.box.map50:.4f}, mAP50-95={r2.box.map:.4f}")
print(f">>> Precision={r2.box.mp:.4f}, Recall={r2.box.mr:.4f}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
diff_mAP50 = r2.box.map50 - r1.box.map50
diff_map = r2.box.map - r1.box.map
print(f"{'Metric':<15} {'P2 (CIoU)':<12} {'P2+WIoU':<12} {'Diff':<12}")
print("-" * 55)
print(f"{'mAP50':<14} {r1.box.map50:<12.4f} {r2.box.map50:<12.4f} {diff_mAP50:+.4f}")
print(f"{'mAP50-95':<12} {r1.box.map:<12.4f} {r2.box.map:<12.4f} {diff_map:+.4f}")
print(f"{'Precision':<12} {r1.box.mp:<12.4f} {r2.box.mp:<12.4f} {r2.box.mp - r1.box.mp:+.4f}")
print(f"{'Recall':<12} {r1.box.mr:<12.4f} {r2.box.mr:<12.4f} {r2.box.mr - r1.box.mr:+.4f}")
