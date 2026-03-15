# YOLO Architecture Research — Experiments

> Bao cao nghien cuu: Kien truc YOLO va Ung dung  
> Thuc nghiem tren PASCAL VOC 2012 va MS COCO 2017

## Ket qua thuc nghiem

### Thuc nghiem 1 — PASCAL VOC 2012 Vehicle Detection (YOLOv8m)

| Metric | Ket qua |
|--------|---------|
| mAP@0.5 | **89.65%** |
| mAP@.5:.95 | 70.24% |
| Precision | 85.42% |
| Recall | 82.27% |
| F1-Score | 83.82% |
| Epochs | 51 (EarlyStopping) |
| Thoi gian | ~45 phut (RTX 3060) |

**Ket qua tung lop:**

| Lop | AP@0.5 | AP@.5:.95 |
|-----|--------|-----------|
| Car | 89.16% | 67.89% |
| Bus | **91.65%** | **78.10%** |
| Motorbike | 90.44% | 69.65% |
| Bicycle | 87.34% | 65.31% |

### Thuc nghiem 2 — MS COCO 2017 (Pretrained Models)

| Model | mAP | mAP50 | FPS (RTX3060) |
|-------|-----|-------|---------------|
| YOLOv8n | 37.2% | 52.5% | 85 |
| YOLOv8s | 44.8% | 61.7% | 95 |
| YOLOv8m | 50.2% | 67.0% | 55 |
| **YOLOv9c** | **52.9%** | **69.6%** | 41 |
| YOLOv10m | 51.1% | 67.7% | 60 |
| YOLOv11m | 51.5% | 68.4% | 54 |

## Moi truong

- GPU: NVIDIA GeForce RTX 3060 Laptop (6GB VRAM)
- OS: Windows 11
- CUDA: 12.0
- Python: 3.10
- PyTorch: 2.1.0+cu121
- Ultralytics: 8.4.22

## Cai dat

```bash
# 1. Clone repo
git clone https://github.com/Yukih/yolo-research.git
cd yolo-research

# 2. Tao conda environment
conda create -n yolo_exp python=3.10 -y
conda activate yolo_exp

# 3. Cai PyTorch (CUDA 12)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 4. Cai cac thu vien
pip install -r requirements.txt
```

## Tai dataset

**PASCAL VOC 2012** (~2GB):
```
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```
Giai nen vao: `datasets/VOCdevkit/`

**MS COCO 2017 val** (~1.2GB):
```
http://images.cocodataset.org/zips/val2017.zip
http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```
Giai nen vao: `datasets/coco/`

## Chay thuc nghiem

```bash
# Buoc 1: Chuyen doi dataset
python convert_voc.py      # VOC XML -> YOLO format
python convert_coco.py     # COCO JSON -> YOLO format

# Buoc 2: Huan luyen (VOC)
python train_voc.py        # ~45 phut tren RTX 3060

# Buoc 3: Danh gia
python eval_voc.py         # Ket qua VOC tung lop
python eval_coco.py        # So sanh 6 models tren COCO

# Buoc 4: Do toc do
python benchmark.py        # FPS thuc te
```

## Cau truc thu muc

```
yolo-research/
├── convert_voc.py          # Chuyen doi VOC
├── convert_coco.py         # Chuyen doi COCO
├── train_voc.py            # Training
├── eval_voc.py             # Evaluation VOC
├── eval_coco.py            # Evaluation COCO
├── benchmark.py            # FPS benchmark
├── voc_vehicles.yaml       # Config VOC dataset
├── coco.yaml               # Config COCO dataset
├── requirements.txt        # Dependencies
└── .gitignore
```

## Luu y quan trong

- **Khong commit** thu muc `datasets/` va `runs/` (da co trong `.gitignore`)
- Dataset va model weights phai tai rieng (xem huong dan tren)
- Ket qua chay tren may khac co the chenh lech nhe tuy GPU va RAM
