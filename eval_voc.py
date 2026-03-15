from ultralytics import YOLO

def main():

    BEST = r'C:\Users\Yukih\yolo_project\runs\voc_exp19\weights\best.pt'
    model = YOLO(BEST)

    m = model.val(
        data  = r'C:\Users\Yukih\yolo_project\voc_vehicles.yaml',
        conf  = 0.001,
        iou   = 0.6,
        plots = True,
        project = r'C:\Users\Yukih\yolo_project\runs',
        name    = 'voc_eval_final',
    )

    f1 = 2 * m.box.mp * m.box.mr / (m.box.mp + m.box.mr + 1e-8)
    CLASSES = ['car', 'bus', 'motorbike', 'bicycle']

    print("="*50)
    print("KET QUA - VOC 2012 VEHICLE DETECTION")
    print("="*50)
    print(f"mAP@0.5      = {m.box.map50*100:.2f}%")
    print(f"mAP@.5:.95   = {m.box.map*100:.2f}%")
    print(f"Precision    = {m.box.mp*100:.2f}%")
    print(f"Recall       = {m.box.mr*100:.2f}%")
    print(f"F1-Score     = {f1*100:.2f}%")
    print()

    print(f"{'Lop':<12}  {'AP@.5':>8}  {'AP@.5:.95':>10}")
    print("-"*35)

    for i, cls in enumerate(CLASSES):
        print(f"{cls:<12}  {m.box.ap50[i]*100:>8.2f}%  {m.box.ap[i]*100:>10.2f}%")

    print("="*50)
    print(">>> XONG! <<<")


if __name__ == "__main__":
    main()
