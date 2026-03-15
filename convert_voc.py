import os, xml.etree.ElementTree as ET, shutil
from pathlib import Path
from tqdm import tqdm

VOC_ROOT = r"C:\Users\Yukih\yolo_project\datasets\VOCdevkit\VOC2012"
OUT_ROOT = r"C:\Users\Yukih\yolo_project\datasets\voc_vehicles"
CLASSES  = ['car', 'bus', 'motorbike', 'bicycle']

def to_yolo(W, H, xmin, ymin, xmax, ymax):
    xc = (xmin + xmax) / 2 / W
    yc = (ymin + ymax) / 2 / H
    w  = (xmax - xmin) / W
    h  = (ymax - ymin) / H
    return (max(0, min(1, v)) for v in [xc, yc, w, h])

for split in ['train', 'val']:
    img_out = Path(OUT_ROOT) / 'images' / split
    lbl_out = Path(OUT_ROOT) / 'labels' / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    ids = (Path(VOC_ROOT) / 'ImageSets' / 'Main' / f'{split}.txt').read_text().split()
    count = 0
    for img_id in tqdm(ids, desc=f'[{split}]'):
        xml_p = Path(VOC_ROOT) / 'Annotations' / f'{img_id}.xml'
        if not xml_p.exists(): continue
        root = ET.parse(xml_p).getroot()
        W = int(root.find('size/width').text)
        H = int(root.find('size/height').text)
        if W == 0 or H == 0: continue
        lines = []
        for obj in root.findall('object'):
            nm = obj.find('name').text.strip()
            if nm not in CLASSES: continue
            diff = obj.find('difficult')
            if diff is not None and int(diff.text) == 1: continue
            b = obj.find('bndbox')
            xmin = float(b.find('xmin').text)
            ymin = float(b.find('ymin').text)
            xmax = float(b.find('xmax').text)
            ymax = float(b.find('ymax').text)
            if xmax <= xmin or ymax <= ymin: continue
            xc, yc, w, h = to_yolo(W, H, xmin, ymin, xmax, ymax)
            lines.append(f'{CLASSES.index(nm)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}')
        if lines:
            shutil.copy(Path(VOC_ROOT)/'JPEGImages'/f'{img_id}.jpg', img_out/f'{img_id}.jpg')
            (lbl_out / f'{img_id}.txt').write_text('\n'.join(lines))
            count += 1
    print(f'{split}: {count} anh co phuong tien')