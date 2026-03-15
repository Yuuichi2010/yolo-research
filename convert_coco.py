import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

COCO_ROOT = r"C:\Users\Yukih\yolo_project\datasets\coco"

JSON_PATH = rf"{COCO_ROOT}\annotations\instances_val2017.json"
OUT_DIR   = rf"{COCO_ROOT}\labels\val2017"

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

print("Dang doc file JSON (~50MB)...")
coco = json.load(open(JSON_PATH, encoding='utf-8'))

cats    = sorted(coco['categories'], key=lambda c: c['id'])
id2idx  = {c['id']: i for i, c in enumerate(cats)}
imgs    = {img['id']: img for img in coco['images']}
anns    = defaultdict(list)

for a in coco['annotations']:
    if not a.get('iscrowd', 0):
        anns[a['image_id']].append(a)

converted = 0
for img_id, img_anns in tqdm(anns.items(), desc='Converting COCO'):
    img = imgs[img_id]
    W, H = img['width'], img['height']
    stem = Path(img['file_name']).stem
    lines = []
    for a in img_anns:
        x, y, w, h = a['bbox']
        if w <= 0 or h <= 0: continue
        xc = (x + w / 2) / W
        yc = (y + h / 2) / H
        lines.append(f'{id2idx[a["category_id"]]} {xc:.6f} {yc:.6f} {w/W:.6f} {h/H:.6f}')
    if lines:
        Path(OUT_DIR, f'{stem}.txt').write_text('\n'.join(lines))
        converted += 1

print(f'Xong! {converted} anh co labels')
print(f'Labels luu tai: {OUT_DIR}')