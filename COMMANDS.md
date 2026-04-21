# Cloud Studio 训练命令笔记

## 1. 后台运行训练（防止锁屏中断）

```bash
cd /workspace
nohup python train_yolo11_p2.py > train.log 2>&1 &
```

查看实时日志：
```bash
tail -f /workspace/train.log
```

查看所有后台进程：
```bash
ps aux | grep python
```

停止训练：
```bash
kill <PID>
```

---

## 2. 恢复中断的训练

```bash
cd /workspace
nohup python resume_train.py > train.log 2>&1 &
```

`resume_train.py` 内容：
```python
from ultralytics import YOLO
model = YOLO("/workspace/runs/detect/VisDrone_Thesis/yolo11n_p2/weights/last.pt")
model.train(resume=True)
```

---

## 3. 基准实验训练

```bash
cd /workspace
nohup python train_yolo11_baseline.py > baseline.log 2>&1 &
```

---

## 4. 查看训练结果

训练结果目录：
```
/workspace/runs/detect/VisDrone_Thesis/yolo11n_p2/
```

关键文件：
- `results.csv` — 每轮 loss 和 mAP 数据
- `weights/best.pt` — 验证集最优权重
- `weights/last.pt` — 最后一轮权重
- `results.png` — 训练曲线图
- `confusion_matrix.png` — 混淆矩阵

---

## 5. 用 best.pt 做推理测试

```python
from ultralytics import YOLO

model = YOLO("/workspace/runs/detect/VisDrone_Thesis/yolo11n_p2/weights/best.pt")
results = model.predict(source="path/to/test_image.jpg", imgsz=640, conf=0.25)
results[0].show()
```

---

## 6. Git 推送（代码备份）

```bash
cd /workspace
git add .
git commit -m "update"
git push origin main
```

---

## 7. 可选：安装 tmux（如果环境允许）

```bash
apt-get update && apt-get install -y tmux
```

tmux 用法：
```bash
tmux new -s train -d                          # 创建后台会话
tmux send-keys -t train "python train.py" Enter  # 发送命令
tmux attach -t train                          # 进入查看
tmux detach                                   # 退出（Ctrl+B 然后 D）
```

---

## 8. 训练参数速查

| 参数 | 值 | 说明 |
|------|-----|------|
| epochs | 150 | 总轮数 |
| imgsz | 640 | 输入分辨率 |
| batch | 16 | 批次大小 |
| amp | True | 混合精度 |
| cos_lr | True | 余弦学习率 |
| patience | 20 | 早停耐心值 |
| close_mosaic | 10 | 最后10轮关闭Mosaic |
| mixup | 0.0 | 关闭MixUp |
| copy_paste | 0.0 | 关闭CopyPaste |
