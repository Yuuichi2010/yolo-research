from ultralytics import YOLO
import torch

if __name__ == '__main__':

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    MODELS = [
        ('YOLOv8n',  'yolov8n.pt'),
        ('YOLOv8s',  'yolov8s.pt'),
        ('YOLOv8m',  'yolov8m.pt'),
        ('YOLOv9c',  'yolov9c.pt'),
        ('YOLOv10m', 'yolov10m.pt'),
        ('YOLOv11m', 'yolo11m.pt'),
    ]

    import os
    cache_file = r'C:\Users\Yukih\yolo_project\datasets\coco\val2017.cache'
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print("Da xoa cache cu")

    results_all = []

    for name, w in MODELS:
        print(f"\n{'='*50}")
        print(f"=== {name} ===")
        print(f"{'='*50}")

        try:
            model = YOLO(w)

            m = model.val(
                data     = r'C:\Users\Yukih\yolo_project\coco.yaml',
                split    = 'val',
                imgsz    = 640,
                batch    = 4,
                conf     = 0.001,
                iou      = 0.65,
                device   = 0,
                workers  = 0,       # QUAN TRONG: workers=0 tranh crash tren Windows
                plots    = True,
                verbose  = False,
                project  = r'C:\Users\Yukih\yolo_project\runs',
                name     = f'coco_{name}',
                exist_ok = True,
            )

            r = {
                'model':  name,
                'mAP':    round(m.box.map   * 100, 1),
                'mAP50':  round(m.box.map50 * 100, 1),
                'P':      round(m.box.mp    * 100, 1),
                'R':      round(m.box.mr    * 100, 1),
                'mAP_S':  round(m.box.maps[0] * 100, 1) if hasattr(m.box,'maps') and len(m.box.maps)>0 else 0,
                'mAP_M':  round(m.box.maps[1] * 100, 1) if hasattr(m.box,'maps') and len(m.box.maps)>1 else 0,
                'mAP_L':  round(m.box.maps[2] * 100, 1) if hasattr(m.box,'maps') and len(m.box.maps)>2 else 0,
            }
            results_all.append(r)
            print(f"  mAP@[.5:.95] = {r['mAP']}%")
            print(f"  mAP@.50      = {r['mAP50']}%")
            print(f"  mAP_Small    = {r['mAP_S']}%")
            print(f"  mAP_Medium   = {r['mAP_M']}%")
            print(f"  mAP_Large    = {r['mAP_L']}%")
            print(f"  Precision    = {r['P']}%")
            print(f"  Recall       = {r['R']}%")

        except Exception as e:
            print(f"  LOI: {e}")
            results_all.append({'model':name,'mAP':0,'mAP50':0,'P':0,'R':0,'mAP_S':0,'mAP_M':0,'mAP_L':0})

        finally:
            try:
                del model
            except:
                pass
            torch.cuda.empty_cache()

    print("\n" + "="*70)
    print("BANG KET QUA TONG HOP")
    print("="*70)
    print(f"{'Model':<12} {'mAP%':>7} {'mAP50%':>8} {'mAP_S%':>8} {'mAP_M%':>8} {'mAP_L%':>8} {'P%':>6} {'R%':>6}")
    print("-"*70)
    for r in results_all:
        print(f"{r['model']:<12} {r['mAP']:>7} {r['mAP50']:>8} {r['mAP_S']:>8} {r['mAP_M']:>8} {r['mAP_L']:>8} {r['P']:>6} {r['R']:>6}")
    print("="*70)
    print(">>> BANG TONG HOP DA DUOC IN RA, XEM O TREN! <<<")
