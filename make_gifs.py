"""
Generate a GIF for each D-NeRF scene from its test images,
with the GT timestamp printed on each frame.

Usage:
    /scratch/gpfs/MONA/mt3204/envs/kplanes/bin/python \
        /scratch/gpfs/MONA/Toki/Academic/COS526/final_experiments/make_gifs.py
"""

import os, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio

DATA_ROOT = "/scratch/gpfs/MONA/Toki/Academic/COS526/dnerf/data"
OUT_DIR   = "/scratch/gpfs/MONA/Toki/Academic/COS526/final_experiments/gifs"
os.makedirs(OUT_DIR, exist_ok=True)

SCENES = ['bouncingballs', 'hellwarrior', 'hook', 'jumpingjacks',
          'lego', 'mutant', 'standup', 'trex']

MS_PER_FRAME = 500   # 0.5 seconds per frame

for scene in SCENES:
    transforms_path = os.path.join(DATA_ROOT, scene, 'transforms_test.json')
    if not os.path.exists(transforms_path):
        print(f"  [skip] {scene}: transforms_test.json not found")
        continue

    meta   = json.load(open(transforms_path))
    frames = sorted(meta['frames'], key=lambda f: f['time'])

    pil_frames = []
    for f in frames:
        # Image path: strip leading './' and append .png
        rel_path = f['file_path'].lstrip('./')
        img_path = os.path.join(DATA_ROOT, scene, rel_path + '.png')
        if not os.path.exists(img_path):
            img_path = os.path.join(DATA_ROOT, scene, rel_path)
        if not os.path.exists(img_path):
            print(f"  [warn] missing: {img_path}")
            continue

        img = Image.open(img_path).convert('RGBA')

        # Composite onto white background (handles transparency)
        bg  = Image.new('RGBA', img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert('RGB')

        # Resize to max 400px wide for compact GIF
        w, h    = img.size
        max_w   = 400
        if w > max_w:
            img = img.resize((max_w, int(h * max_w / w)), Image.LANCZOS)

        # Draw timestamp text
        draw  = ImageDraw.Draw(img)
        t_val = f['time']
        label = f"t = {t_val:.3f}"

        # Text box with semi-transparent background
        font_size = max(12, img.width // 22)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        bbox    = draw.textbbox((0, 0), label, font=font)
        tw, th  = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad     = 6
        x, y    = 10, 10
        draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad],
                       fill=(0, 0, 0, 180) if hasattr(img, 'putalpha') else (0, 0, 0))
        draw.text((x, y), label, font=font, fill=(255, 255, 100))

        pil_frames.append(np.array(img))

    if not pil_frames:
        print(f"  [skip] {scene}: no frames loaded")
        continue

    out_path = os.path.join(OUT_DIR, f'{scene}.gif')
    imageio.mimwrite(out_path, pil_frames, format='GIF',
                     duration=MS_PER_FRAME, loop=0)
    print(f"  {scene}: {len(pil_frames)} frames → {out_path}")

print(f"\nDone. GIFs saved to {OUT_DIR}/")
