import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision.transforms.functional import normalize
import torch.nn.functional as F

def sample_dataset(img_dir, mask_dir, num_samples=None):
    img_paths = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if f.endswith(('.png','.jpg','.jpeg'))
    ])

    mask_paths = sorted([
        os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
        if f.endswith('.npy')
    ])

    pairs = list(zip(img_paths, mask_paths))

    # ===== 按任务拆分 =====
    ai_pairs = []
    real_pairs = []

    def get_mode(path):
        if 'ai_on_real' in path:
            return 0
        elif 'real_on_fake' in path:
            return 1
        else:
            return -1

    for img, mask in pairs:
        mode = get_mode(img)
        if mode == 0:
            ai_pairs.append((img, mask, 0))
        elif mode == 1:
            real_pairs.append((img, mask, 1))

    print(f"AI_on_real: {len(ai_pairs)}, Real_on_fake: {len(real_pairs)}")

    # ===== 1:1 采样 =====
    if num_samples is not None:
        half = num_samples // 2

        # 防止不够
        half = min(half, len(ai_pairs), len(real_pairs))

        ai_sample = random.sample(ai_pairs, half)
        real_sample = random.sample(real_pairs, half)

        pairs = ai_sample + real_sample
    else:
        # 如果不限制数量，就取 min 保持平衡
        min_len = min(len(ai_pairs), len(real_pairs))
        pairs = ai_pairs[:min_len] + real_pairs[:min_len]

    random.shuffle(pairs)

    # ===== 拆回 =====
    img_paths, mask_paths, modes = [], [], []

    for img, mask, mode in pairs:
        img_paths.append(img)
        mask_paths.append(mask)
        modes.append(mode)

    print(f"最终采样: {len(img_paths)} (1:1 平衡)")

    return img_paths, mask_paths, modes

class MaskDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None, modes=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.modes = modes

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # ===== 图像 =====
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ===== mask =====
        mask = np.load(self.mask_paths[idx])  # 0/1

        # resize
        img = cv2.resize(img, (224,224))
        mask = cv2.resize(mask, (224,224), interpolation=cv2.INTER_NEAREST)

        # ===== DCT =====
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dct = cv2.dct(np.float32(gray))
        dct = np.log(np.abs(dct)+1e-6)
        dct = cv2.normalize(dct,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

        # ===== tensor =====
        img = transforms.ToTensor()(Image.fromarray(img))
        img = normalize(img,[0.485,0.456,0.406],[0.229,0.224,0.225])

        dct = torch.from_numpy(dct).float().unsqueeze(0)/255.0

        x = torch.cat([img, dct], dim=0)

        mask = torch.from_numpy(mask).float().unsqueeze(0)  # [1,H,W]

        return x, mask, self.modes[idx]
    

class SwinSeg(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            num_classes=0,
            in_chans=4,
            features_only=True   # ⭐关键
        )

        channels = self.backbone.feature_info.channels()

        self.decoder = nn.Sequential(
            nn.Conv2d(channels[-1], 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        feats = self.backbone(x)
        x = feats[-1]  # 最深层
        x = x.permute(0, 3, 1, 2) 
        x = self.decoder(x)
        x = F.interpolate(x, size=(224,224), mode='bilinear')
        return x  # logits
    

class DiceLoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        smooth = 1e-6

        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2 * intersection + smooth) / (union + smooth)

        return 1 - dice.mean()


class SegLoss(nn.Module):
    def __init__(self, device, pos_weight=3.0):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device)
        )

        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)

        return bce_loss + dice_loss
    
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for x, mask,_ in loader:
        x, mask = x.to(device), mask.to(device)

        optimizer.zero_grad()
        pred = model(x)

        loss = criterion(pred, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()

    iou_total = 0
    count = 0

    # 🔥 分任务统计
    iou_ai_on_real = 0
    cnt_ai = 0

    iou_real_on_fake = 0
    cnt_real = 0

    with torch.no_grad():
        for x, mask, mode in loader:
            x, mask = x.to(device), mask.to(device)

            pred = torch.sigmoid(model(x))
            pred = (pred > 0.5).float()

            for i in range(x.size(0)):
                inter = (pred[i] * mask[i]).sum()
                union = pred[i].sum() + mask[i].sum() - inter

                iou = inter / (union + 1e-6)

                iou_total += iou.item()
                count += 1

                # 🔥 按任务分
                if mode[i] == 0:
                    iou_ai_on_real += iou.item()
                    cnt_ai += 1
                elif mode[i] == 1:
                    iou_real_on_fake += iou.item()
                    cnt_real += 1

    return {
        "mean_iou": iou_total / count,
        "ai_on_real": iou_ai_on_real / max(cnt_ai,1),
        "real_on_fake": iou_real_on_fake / max(cnt_real,1)
    }

class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.train_losses = []
        self.mean_ious = []       # 🔥 新增
        self.ai_ious = []         # 🔥 新增
        self.real_ious = []       # 🔥 新增

    # =========================
    # 1️⃣ Loss + IoU 曲线
    # =========================
    def plot_curves(self):
        epochs = range(1, len(self.train_losses)+1)

        plt.figure(figsize=(10,6))

        plt.plot(epochs, self.train_losses, label='Train Loss')
        # plt.plot(epochs, self.val_ious, label='Val IoU')
        plt.plot(epochs, self.mean_ious, label='Mean IoU')
        plt.plot(epochs, self.ai_ious, label='AI on Real')
        plt.plot(epochs, self.real_ious, label='Real on Fake')

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training Curve (Loss & IoU)")
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(self.save_dir, 'curve.png')
        plt.savefig(save_path, dpi=300)
        # plt.show()
        plt.close()

        print(f"曲线已保存: {save_path}")

    # =========================
    # 2️⃣ segmentation 可视化（增强版）
    # =========================
    def visualize_segmentation(self, model, dataloader, device, num_samples=6, draw_num=0):
        model.eval()

        samples = []

        with torch.no_grad():
            for x, mask, mode in dataloader:   # 🔥 拿到 mode
                x = x.to(device)

                pred = torch.sigmoid(model(x))

                for i in range(x.size(0)):
                    samples.append((
                        x[i].cpu(),
                        mask[i].cpu(),
                        pred[i].cpu(),
                        mode[i].item()   # 🔥 存 mode
                    ))

                if len(samples) >= num_samples:
                    break

        samples = samples[:num_samples]

        fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3*num_samples))

        mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

        for i, (img, gt, pred, mode) in enumerate(samples):

            # ===== 还原 RGB =====
            img_rgb = img[:3] * std + mean
            img_rgb = torch.clamp(img_rgb, 0, 1)

            gt_map = gt.squeeze().numpy()
            pred_map = pred.squeeze().numpy()
            pred_bin = (pred_map > 0.5).astype(np.float32)

            # ===== 🔥 计算 IoU（很重要）=====
            inter = (pred_bin * gt_map).sum()
            union = pred_bin.sum() + gt_map.sum() - inter
            iou = inter / (union + 1e-6)

            # ===== 🔥 任务名称 =====
            task_name = "AI→Real" if mode == 0 else "Real→AI"

            # ===== 原图 =====
            axes[i,0].imshow(img_rgb.permute(1,2,0))
            axes[i,0].set_title(f"{task_name}", fontsize=10)
            axes[i,0].axis('off')

            # ===== GT =====
            axes[i,1].imshow(gt_map, cmap='gray')
            axes[i,1].set_title("GT Mask")
            axes[i,1].axis('off')

            # ===== Heatmap =====
            axes[i,2].imshow(pred_map, cmap='jet')
            axes[i,2].set_title("Heatmap")
            axes[i,2].axis('off')

            # ===== Binary =====
            axes[i,3].imshow(pred_bin, cmap='gray')
            axes[i,3].set_title(f"Binary\nIoU={iou:.2f}")
            axes[i,3].axis('off')

            # ===== Overlay =====
            overlay = img_rgb.permute(1,2,0).numpy().copy()
            overlay[..., 0] += pred_bin * 0.5
            overlay = np.clip(overlay, 0, 1)

            axes[i,4].imshow(overlay)
            axes[i,4].set_title("Overlay")
            axes[i,4].axis('off')

        plt.tight_layout()

        save_path = os.path.join(self.save_dir, f'{draw_num}_seg.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"分割可视化已保存: {save_path}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TRAIN_NUM = 280
    VAL_NUM = 40

    train_imgs, train_masks, train_modes = sample_dataset(
        './dataset_mask/train/images',
        './dataset_mask/train/masks',
        TRAIN_NUM
    )

    val_imgs, val_masks, val_modes = sample_dataset(
        './dataset_mask/val/images',
        './dataset_mask/val/masks',
        VAL_NUM
    )

    train_ds = MaskDataset(train_imgs, train_masks, modes=train_modes)
    val_ds = MaskDataset(val_imgs, val_masks, modes=val_modes)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    model = SwinSeg().to(device)
    criterion = SegLoss(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    best_iou = 0
    os.makedirs('./Mask_Model', exist_ok=True)
    vis = Visualizer('./Mask_Model/vis/improve_loss_and_data_divide')

    for epoch in range(20):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        vis.train_losses.append(train_loss)

        metrics = evaluate(model, val_loader, device)
        vis.mean_ious.append(metrics['mean_iou'])
        vis.ai_ious.append(metrics['ai_on_real'])
        vis.real_ious.append(metrics['real_on_fake'])

        print(
            f"Loss {train_loss:.4f} | "
            f"Mean IoU {metrics['mean_iou']:.4f} | "
            f"AI→Real {metrics['ai_on_real']:.4f} | "
            f"Real→AI {metrics['real_on_fake']:.4f}"
        )

        if metrics['mean_iou'] > best_iou:
            best_iou = metrics['mean_iou']
            torch.save(model.state_dict(), './Mask_Model/best_seg.pth')
        if (epoch+1) % 5 == 0:
            vis.visualize_segmentation(
                model,
                val_loader,
                device,
                num_samples=6,
                draw_num=epoch+1
            )
    vis.plot_curves()
    print("训练完成")

if __name__ == "__main__":
    main()