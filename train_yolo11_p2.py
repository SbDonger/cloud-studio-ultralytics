from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 从 YAML 创建 P2 模型（自动识别 scale='n'）
    model = YOLO("yolo11-p2.yaml")

    # 2. 加载官方 yolo11n.pt 预训练权重，backbone 匹配部分自动加载，
    #    head 部分（4 尺度 vs 3 尺度）不匹配会自动跳过，不影响训练
    model.load("yolo11n.pt")

    # 3. 启动训练
    results = model.train(
        data="VisDrone.yaml",
        epochs=150,
        imgsz=640,
        batch=16,
        device=0,

        # --- 显存与速度优化 ---
        amp=True,
        workers=4,               # 建议 4，避免数据加载瓶颈且稳定

        # --- 学习率与收敛优化 ---
        cos_lr=True,             # 余弦退火学习率，收敛更平滑
        patience=20,             # 早停：验证 mAP 20 轮不提升则停止，防止过拟合
        close_mosaic=10,         # 最后 10 轮关闭 Mosaic，稳定小目标检测精度

        # --- 数据增强（VisDrone 小目标场景建议关闭 mixup/copy_paste） ---
        mixup=0.0,
        copy_paste=0.0,

        # --- 实验管理 ---
        project="VisDrone_Thesis",
        name="yolo11n_p2",

        # --- 可复现性（⚠️ 会显著降低训练速度，非必须建议注释掉） ---
        # seed=42,
        # deterministic=True,
    )
