import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
from pathlib import Path

random.seed(42)

DATASET = Path('./dataset_512')
SAVE_PATH = './Mask_Model/mask_check.png'

CATEGORIES = {
    'ai_gen':       'pure_ai',
    'fake_on_real': 'ai_on_real',
    'random_ai':    'random_ai',
    'real':         'pure_real',
    'real_on_fake': 'real_on_fake',
    'specific':     'specific_position_ai',
}

N_PER_CLASS = 3
COLS = 3   # image | mask | overlay

fig, axes = plt.subplots(
    len(CATEGORIES) * N_PER_CLASS, COLS,
    figsize=(COLS * 4, len(CATEGORIES) * N_PER_CLASS * 3)
)

for row_base, (src_dir, label) in enumerate(CATEGORIES.items()):
    img_dir  = DATASET / src_dir / 'images'
    mask_dir = DATASET / src_dir / 'masks'

    all_imgs = sorted(img_dir.glob('*.png')) + sorted(img_dir.glob('*.jpg'))
    chosen   = random.sample(all_imgs, min(N_PER_CLASS, len(all_imgs)))

    for j, img_path in enumerate(chosen):
        row = row_base * N_PER_CLASS + j

        # --- load ---
        img  = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        msk  = np.load(mask_dir / (img_path.stem + '.npy'))
        img  = cv2.resize(img,  (256, 256))
        msk  = cv2.resize(msk.astype(np.uint8), (256, 256),
                          interpolation=cv2.INTER_NEAREST)

        fg_ratio = msk.mean()

        # --- overlay: green=0(authentic) red=1(forged) ---
        overlay = img.copy().astype(np.float32) / 255.0
        alpha = 0.45
        overlay[msk == 1, 0] = np.clip(overlay[msk == 1, 0] + alpha, 0, 1)  # red
        overlay[msk == 1, 1] = np.clip(overlay[msk == 1, 1] - alpha * 0.3, 0, 1)
        overlay[msk == 0, 1] = np.clip(overlay[msk == 0, 1] + alpha * 0.4, 0, 1)  # green

        # --- plot ---
        axes[row, 0].imshow(img)
        axes[row, 1].imshow(msk, cmap='gray', vmin=0, vmax=1)
        axes[row, 2].imshow(np.clip(overlay, 0, 1))

        for col in range(COLS):
            axes[row, col].axis('off')

        # label only on first sample of each class
        if j == 0:
            axes[row, 0].set_ylabel(label, fontsize=9, rotation=0,
                                    labelpad=80, va='center')

        axes[row, 0].set_title(img_path.name[:22], fontsize=6.5)
        axes[row, 1].set_title(f'mask  fg={fg_ratio:.1%}', fontsize=7)
        axes[row, 2].set_title('overlay (red=forged, green=real)', fontsize=6.5)

# column headers
for col, title in enumerate(['Image', 'Mask (white=1 forged)', 'Overlay']):
    axes[0, col].set_title(title, fontsize=9, fontweight='bold')

# legend
red_patch   = mpatches.Patch(color=(0.8, 0.1, 0.1), label='mask=1  forged region')
green_patch = mpatches.Patch(color=(0.1, 0.7, 0.1), label='mask=0  authentic region')
fig.legend(handles=[red_patch, green_patch], loc='lower center',
           ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.0))

plt.suptitle('dataset_512 — mask annotation check (3 random per class)',
             fontsize=12, fontweight='bold', y=1.002)
plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=130, bbox_inches='tight')
plt.close()
print(f'saved: {SAVE_PATH}')
