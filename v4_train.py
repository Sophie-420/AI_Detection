import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from tqdm import tqdm


# ─────────────────────────── Dataset ────────────────────────────

def sample_dataset(img_dir, mask_dir, num_samples=None, specific_repeat=3):
    img_paths = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    mask_paths = sorted([
        os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
        if f.endswith('.npy')
    ])
    pairs = list(zip(img_paths, mask_paths))

    MODE = {
        'ai_on_real': 0,
        'real_on_fake': 1,
        'random_ai': 2,
        'specific_position_ai': 3,
        'pure_ai': 4,
        'pure_real': 5,
    }

    buckets = {k: [] for k in MODE}
    for img, mask in pairs:
        for key, idx in MODE.items():
            if key in img:
                buckets[key].append((img, mask, idx))
                break

    print("\n原始数据分布：")
    for k, v in buckets.items():
        print(f"  {k}: {len(v)}")

    # Use ALL data; specific_position_ai repeated specific_repeat times
    n_cats = len(MODE)
    sampled = []
    for key, items in buckets.items():
        if num_samples is not None:
            items = random.sample(items, min(num_samples // n_cats, len(items)))
        if key == 'specific_position_ai':
            sampled.extend(items * specific_repeat)
        else:
            sampled.extend(items)
    random.shuffle(sampled)

    print(f"采样后总计: {len(sampled)} (specific ×{specific_repeat})\n")
    imgs, masks, modes = zip(*sampled)
    return list(imgs), list(masks), list(modes)


class MaskDataset(Dataset):
    def __init__(self, img_paths, mask_paths, modes, augment=False):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.modes = modes
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def _augment(img, mask):
        if random.random() > 0.5:
            img = cv2.flip(img, 1); mask = cv2.flip(mask, 1)
        if random.random() > 0.5:
            img = cv2.flip(img, 0); mask = cv2.flip(mask, 0)
        angle = random.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img  = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask.astype(np.float32), M, (w, h),
                              flags=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.uint8)
        # Color jitter — image only
        alpha = random.uniform(0.8, 1.2)
        beta  = random.uniform(-20, 20)
        img   = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
        return img, mask

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.img_paths[idx]), cv2.COLOR_BGR2RGB)
        mask = np.load(self.mask_paths[idx])

        img = cv2.resize(img, (224, 224))
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)  # 统一到 {0,1}，修复 random_ai 里的 {0,255}

        if self.augment:
            img, mask = self._augment(img, mask)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dct = cv2.dct(np.float32(gray))
        dct = np.log(np.abs(dct) + 1e-6)
        dct = cv2.normalize(dct, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        img_t = normalize(transforms.ToTensor()(Image.fromarray(img)),
                          [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        dct_t = torch.from_numpy(dct).float().unsqueeze(0) / 255.0
        x = torch.cat([img_t, dct_t], dim=0)

        return x, torch.from_numpy(mask).float().unsqueeze(0), self.modes[idx]


# ─────────────────────────── Model ──────────────────────────────

class SwinSeg(nn.Module):
    """
    Swin-Tiny backbone + 4-level U-Net decoder.
    Lateral projections include BN+ReLU to normalize backbone features
    before they enter the decoder, preventing gradient explosion.
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            num_classes=0,
            in_chans=4,
            features_only=True,
        )
        ch = self.backbone.feature_info.channels()  # [96, 192, 384, 768]

        # Lateral: normalize + project each stage to fixed width
        self.lat3 = self._lateral(ch[3], 256)
        self.lat2 = self._lateral(ch[2], 256)
        self.lat1 = self._lateral(ch[1], 128)
        self.lat0 = self._lateral(ch[0], 64)

        # Fusion blocks (double conv) after concat
        self.fuse3 = self._fuse(256,       256)
        self.fuse2 = self._fuse(256 + 256, 128)
        self.fuse1 = self._fuse(128 + 128, 64)
        self.fuse0 = self._fuse(64  + 64,  32)

        self.head = nn.Conv2d(32, 1, 1)

    @staticmethod
    def _lateral(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _fuse(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feats = self.backbone(x)
        f0, f1, f2, f3 = [f.permute(0, 3, 1, 2) for f in feats]
        # f0:[B,96,56,56]  f1:[B,192,28,28]  f2:[B,384,14,14]  f3:[B,768,7,7]

        p3 = self.lat3(f3)   # [B, 256,  7,  7]
        p2 = self.lat2(f2)   # [B, 256, 14, 14]
        p1 = self.lat1(f1)   # [B, 128, 28, 28]
        p0 = self.lat0(f0)   # [B,  64, 56, 56]

        d3 = self.fuse3(p3)
        d2 = self.fuse2(torch.cat([self._up(d3), p2], dim=1))
        d1 = self.fuse1(torch.cat([self._up(d2), p1], dim=1))
        d0 = self.fuse0(torch.cat([self._up(d1), p0], dim=1))

        out = self.head(d0)
        return F.interpolate(out, size=(224, 224), mode='bilinear', align_corners=False)

    @staticmethod
    def _up(x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)


# ─────────────────────────── Loss ───────────────────────────────

class DiceLoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.sigmoid(pred).view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        inter = (pred * target).sum(dim=1)
        return 1 - ((2 * inter + 1e-6) / (pred.sum(dim=1) + target.sum(dim=1) + 1e-6)).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.exp(-bce)
        focal = self.alpha * (1 - p_t) ** self.gamma * bce
        return focal.mean()


class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return self.focal(pred, target) + self.dice(pred, target)


# ─────────────────────────── Train / Eval ───────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc='Train', leave=False)
    for x, mask, _ in loop:
        x, mask = x.to(device), mask.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        loop.set_postfix(loss=f'{loss.item():.3f}')
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    cats = {0: ('ai_on_real', 0, 0), 1: ('real_on_fake', 0, 0),
            2: ('random_ai', 0, 0), 3: ('specific_position_ai', 0, 0),
            4: ('pure_ai', 0, 0), 5: ('pure_real', 0, 0)}
    total_iou, total_n = 0.0, 0

    with torch.no_grad():
        for x, mask, modes in loader:
            x, mask = x.to(device), mask.to(device)
            pred = (torch.sigmoid(model(x)) > threshold).float()
            for i in range(x.size(0)):
                inter = (pred[i] * mask[i]).sum()
                union = pred[i].sum() + mask[i].sum() - inter
                iou = (inter / (union + 1e-6)).item()
                total_iou += iou
                total_n += 1
                m = modes[i].item()
                name, s, n = cats[m]
                cats[m] = (name, s + iou, n + 1)

    result = {'mean_iou': total_iou / max(total_n, 1)}
    for name, s, n in cats.values():
        result[name] = s / max(n, 1)
    return result


# ─────────────────────────── Visualizer ─────────────────────────

class Visualizer:
    TASK = {0: 'AI→Real', 1: 'Real→AI', 2: 'Random AI', 3: 'Specific AI',
            4: 'Pure AI', 5: 'Pure Real'}
    TASK_METRIC = {
        'AI→Real':    'ai_on_real',
        'Real→AI':    'real_on_fake',
        'Random AI':  'random_ai',
        'Specific AI':'specific_position_ai',
        'Pure AI':    'pure_ai',
        'Pure Real':  'pure_real',
    }

    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.losses, self.mean_ious = [], []
        self.cat_ious = {k: [] for k in self.TASK.values()}

    def plot_curves(self):
        epochs = range(1, len(self.losses) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.losses, label='Train Loss')
        plt.plot(epochs, self.mean_ious, label='Mean IoU', linewidth=2)
        for k, v in self.cat_ious.items():
            plt.plot(epochs, v, linestyle='--', label=k)
        plt.xlabel('Epoch'); plt.ylabel('Value')
        plt.title('Training Curve'); plt.legend(); plt.grid(True)
        path = os.path.join(self.save_dir, 'curve.png')
        plt.savefig(path, dpi=150); plt.close()
        print(f'曲线已保存: {path}')

    def visualize_segmentation(self, model, loader, device, num_samples=8, tag=''):
        model.eval()
        samples = []
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        with torch.no_grad():
            for x, mask, mode in loader:
                pred = torch.sigmoid(model(x.to(device))).cpu()
                for i in range(x.size(0)):
                    samples.append((x[i], mask[i], pred[i], mode[i].item()))
                if len(samples) >= num_samples:
                    break
        samples = samples[:num_samples]

        fig, axes = plt.subplots(len(samples), 5, figsize=(15, 3 * len(samples)))
        for i, (img, gt, pred, mode) in enumerate(samples):
            rgb = torch.clamp(img[:3] * std + mean, 0, 1).permute(1, 2, 0).numpy()
            gt_map   = gt.squeeze().numpy()
            pred_map = pred.squeeze().numpy()
            pred_bin = (pred_map > 0.5).astype(np.float32)
            inter = (pred_bin * gt_map).sum()
            union = pred_bin.sum() + gt_map.sum() - inter
            iou = inter / (union + 1e-6)

            axes[i, 0].imshow(rgb);                      axes[i, 0].set_title(self.TASK[mode])
            axes[i, 1].imshow(gt_map, cmap='gray');      axes[i, 1].set_title('GT')
            axes[i, 2].imshow(pred_map, cmap='jet');     axes[i, 2].set_title('Heatmap')
            axes[i, 3].imshow(pred_bin, cmap='gray');    axes[i, 3].set_title(f'Binary IoU={iou:.2f}')
            overlay = rgb.copy(); overlay[..., 0] = np.clip(overlay[..., 0] + pred_bin * 0.5, 0, 1)
            axes[i, 4].imshow(overlay);                  axes[i, 4].set_title('Overlay')
            for ax in axes[i]: ax.axis('off')

        plt.tight_layout()
        path = os.path.join(self.save_dir, f'seg_{tag}.png')
        plt.savefig(path, dpi=150); plt.close()
        print(f'可视化已保存: {path}')


# ─────────────────────────── Main ───────────────────────────────

TOTAL_EPOCHS = 20


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_imgs, train_masks, train_modes = sample_dataset(
        './dataset_mask/train/images', './dataset_mask/train/masks')
    val_imgs, val_masks, val_modes = sample_dataset(
        './dataset_mask/val/images', './dataset_mask/val/masks')

    train_loader = DataLoader(
        MaskDataset(train_imgs, train_masks, train_modes, augment=True),
        batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        MaskDataset(val_imgs, val_masks, val_modes, augment=False),
        batch_size=8, shuffle=False, num_workers=0)

    model     = SwinSeg().to(device)
    criterion = SegLoss()

    save_dir = './Mask_Model/v4'
    os.makedirs(save_dir, exist_ok=True)
    vis = Visualizer(save_dir)
    best_iou = 0.0

    # Single-phase: differential lr from epoch 1, no warmup
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 5e-6},
        {'params': [p for n, p in model.named_parameters()
                    if 'backbone' not in n], 'lr': 1e-4},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)

    for epoch in range(TOTAL_EPOCHS):
        lr_bb  = optimizer.param_groups[0]['lr']
        lr_dec = optimizer.param_groups[1]['lr']
        print(f'\nEpoch {epoch+1}/{TOTAL_EPOCHS}  bb_lr={lr_bb:.2e}  dec_lr={lr_dec:.2e}')
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        _log_epoch(model, val_loader, device, vis, loss, save_dir, best_iou)
        best_iou = max(best_iou, vis.mean_ious[-1])

        if (epoch + 1) % 5 == 0:
            vis.visualize_segmentation(model, val_loader, device, tag=f'ep{epoch+1}')

    vis.plot_curves()
    print(f'\n训练完成  best val IoU={best_iou:.4f}')


def _log_epoch(model, val_loader, device, vis, loss, save_dir, best_iou):
    metrics = evaluate(model, val_loader, device)
    vis.losses.append(loss)
    vis.mean_ious.append(metrics['mean_iou'])
    for k in vis.TASK.values():
        vis.cat_ious[k].append(metrics[vis.TASK_METRIC[k]])
    print(
        f"Loss {loss:.4f} | Mean {metrics['mean_iou']:.4f} | "
        f"AI→Real {metrics['ai_on_real']:.4f} | Real→AI {metrics['real_on_fake']:.4f} | "
        f"Random {metrics['random_ai']:.4f} | Specific {metrics['specific_position_ai']:.4f} | "
        f"PureAI {metrics['pure_ai']:.4f} | PureReal {metrics['pure_real']:.4f}"
    )
    if metrics['mean_iou'] > best_iou:
        torch.save(model.state_dict(), os.path.join(save_dir, 'best_seg.pth'))
        print(f'  → 保存最优模型 (IoU={metrics["mean_iou"]:.4f})')


if __name__ == '__main__':
    main()
