from ultralytics import YOLO
import torch

if __name__ == '__main__':   # BẮT BUỘC trên Windows

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    model = YOLO('yolov8m.pt')

    results = model.train(
        data    = r'C:\Users\Yukih\yolo_project\voc_vehicles.yaml',
        epochs  = 100,
        imgsz   = 640,
        batch   = 8,

        optimizer    = 'SGD',
        lr0          = 0.01,
        lrf          = 0.01,
        momentum     = 0.937,
        weight_decay = 0.0005,
        warmup_epochs= 3.0,

        mosaic  = 1.0,
        mixup   = 0.1,
        fliplr  = 0.5,
        flipud  = 0.0,
        degrees = 0.0,
        scale   = 0.5,

        box     = 7.5,
        cls     = 0.5,
        dfl     = 1.5,

        device  = 0,
        workers = 2,
        amp     = True,
        cache   = False,

        project = r'C:\Users\Yukih\yolo_project\runs',
        name    = 'voc_exp1',
        plots   = True,
        patience= 50,
    )

    print("TRAINING XONG!")
    print(f"Best model: {results.save_dir}\\weights\\best.pt")