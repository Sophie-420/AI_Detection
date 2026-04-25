"""
Inference script for SwinSeg (v4).
Usage:
    python infer.py <image_or_folder> [--ckpt best_seg_ft.pth] [--out results/] [--threshold 0.5]

Outputs per image (saved to --out):
    <name>_original.png   — resized 512×512 original
    <name>_dct.png        — DCT frequency channel
    <name>_heatmap.png    — JET colormap of sigmoid probability
    <name>_binary.png     — binary mask (prob > threshold)
    <name>_overlay.png    — original blended with red highlight
    <name>_grid.png       — all 5 panels in one figure
"""
import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms.functional import normalize

sys.path.insert(0, os.path.dirname(__file__))
from v4_train import SwinSeg

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def preprocess(img_path: str):
    """Load image → 4-channel tensor (1, 4, 512, 512)."""
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise ValueError(f"Cannot read image: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (512, 512))

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dct  = cv2.dct(np.float32(gray))
    dct  = np.log(np.abs(dct) + 1e-6)
    dct  = cv2.normalize(dct, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    img_t = normalize(
        torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0,
        MEAN, STD,
    )
    dct_t = torch.from_numpy(dct).float().unsqueeze(0) / 255.0
    x = torch.cat([img_t, dct_t], dim=0).unsqueeze(0)  # [1, 4, 512, 512]
    return x, rgb, dct


def run_model(model, x, device, threshold):
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()   # [512, 512]
    binary = (prob > threshold).astype(np.uint8)
    return prob, binary


def save_results(rgb, dct, prob, binary, out_dir, stem, threshold):
    os.makedirs(out_dir, exist_ok=True)

    # Heatmap: JET colormap on prob
    heatmap_bgr = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Overlay: blend original + red highlight for forged regions
    overlay = rgb.astype(np.float32) / 255.0
    red_mask = binary.astype(np.float32)
    overlay[..., 0] = np.clip(overlay[..., 0] + red_mask * 0.5, 0, 1)
    overlay[..., 1] = np.clip(overlay[..., 1] - red_mask * 0.2, 0, 1)
    overlay[..., 2] = np.clip(overlay[..., 2] - red_mask * 0.2, 0, 1)
    overlay = (overlay * 255).astype(np.uint8)

    forged_ratio = binary.mean() * 100
    max_prob     = prob.max()
    mean_prob    = prob.mean()

    # Individual saves
    cv2.imwrite(os.path.join(out_dir, f'{stem}_original.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, f'{stem}_dct.png'), dct)
    cv2.imwrite(os.path.join(out_dir, f'{stem}_heatmap.png'), heatmap_bgr)
    cv2.imwrite(os.path.join(out_dir, f'{stem}_binary.png'), binary * 255)
    cv2.imwrite(os.path.join(out_dir, f'{stem}_overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Grid figure
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    titles = ['Original', 'DCT Channel', 'Heatmap', f'Binary (thr={threshold})', 'Overlay']
    imgs   = [rgb, dct, heatmap_rgb, binary * 255, overlay]
    cmaps  = [None, 'gray', None, 'gray', None]
    for ax, title, im, cmap in zip(axes, titles, imgs, cmaps):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    fig.suptitle(
        f'{stem}  |  forged={forged_ratio:.1f}%  max={max_prob:.3f}  mean={mean_prob:.3f}',
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    grid_path = os.path.join(out_dir, f'{stem}_grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()

    return forged_ratio, max_prob, mean_prob, grid_path


def collect_images(path: str):
    if os.path.isfile(path):
        return [path]
    paths = []
    for fname in sorted(os.listdir(path)):
        if os.path.splitext(fname)[1].lower() in IMG_EXTS:
            paths.append(os.path.join(path, fname))
    return paths


def main():
    parser = argparse.ArgumentParser(description='SwinSeg v4 inference')
    parser.add_argument('input', help='Image file or folder')
    parser.add_argument('--ckpt', default=os.path.join(os.path.dirname(__file__), 'best_seg_ft.pth'))
    parser.add_argument('--out',  default=os.path.join(os.path.dirname(__file__), 'results'))
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print(f'Loading checkpoint: {args.ckpt}')
    model = SwinSeg().to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    images = collect_images(args.input)
    if not images:
        print(f'No images found at: {args.input}')
        sys.exit(1)
    print(f'Found {len(images)} image(s)')

    for img_path in images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        print(f'\n[{stem}]', end='  ')
        try:
            x, rgb, dct = preprocess(img_path)
            prob, binary = run_model(model, x, device, args.threshold)
            ratio, maxp, meanp, grid = save_results(
                rgb, dct, prob, binary, args.out, stem, args.threshold)
            print(f'forged={ratio:.1f}%  max={maxp:.3f}  mean={meanp:.3f}  → {grid}')
        except Exception as e:
            print(f'ERROR: {e}')

    print(f'\n✅ Done. Results saved to: {args.out}')


if __name__ == '__main__':
    main()
