from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("/workspace/runs/detect/VisDrone_Thesis/yolo11n_p2/weights/last.pt")
    model.train(resume=True)
