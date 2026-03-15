from ultralytics import YOLO
import time, numpy as np, torch

MODELS = [
    'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt',
    'yolov9c.pt', 'yolov10m.pt', 'yolo11m.pt'
]
img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"{'Model':<12} {'ms/img':>8} {'FPS':>8}")
print("-"*30)

for w in MODELS:
    model = YOLO(w)
    for _ in range(10):
        model(img, verbose=False)
    torch.cuda.synchronize()
    times = []
    for _ in range(100):
        t = time.perf_counter()
        model(img, verbose=False)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    ms  = round(np.mean(times), 1)
    fps = round(1000 / ms)
    print(f"{w:<12} {ms:>8} {fps:>8}")
    del model
    torch.cuda.empty_cache()

print("\n>>> XONG! <<<")