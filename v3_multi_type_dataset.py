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
from tqdm import tqdm 
from scipy.ndimage import distance_transform_edt as distance

def sample_dataset(img_dir, mask_dir, num_samples=None):
    """
    从数据集中采样，支持四个类别，保持类别平衡
    """
    img_paths = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if f.endswith(('.png','.jpg','.jpeg'))
    ])

    mask_paths = sorted([
        os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
        if f.endswith('.npy')
    ])

    pairs = list(zip(img_paths, mask_paths))

    # ===== 按四个类别分组 =====
    categories = {
        'ai_on_real': [],
        'real_on_fake': [],
        'random_ai': [],
        'specific_position_ai': []
    }

    def get_mode_from_path(path):
        if 'ai_on_real' in path:
            return 'ai_on_real'
        elif 'real_on_fake' in path:
            return 'real_on_fake'
        elif 'random_ai' in path:
            return 'random_ai'
        elif 'specific_position_ai' in path:
            return 'specific_position_ai'
        else:
            return None

    for img, mask in pairs:
        mode = get_mode_from_path(img)
        if mode is not None:
            categories[mode].append((img, mask, mode))

    # 打印各类别数量
    print("\n 原始数据分布：")
    for cat, items in categories.items():
        print(f"   {cat}: {len(items)}")

    # ===== 确定每个类别采样数量 =====
    min_count = min(len(items) for items in categories.values())
    
    if num_samples is None:
        # 未指定：取最少类别的数量，每个类别取同样多
        samples_per_class = min_count
        print(f"\n 未指定采样数量，使用平衡采样：每类 {samples_per_class} 张")
    else:
        # 指定总数：均匀分配到四个类别
        samples_per_class = num_samples // 4
        if samples_per_class > min_count:
            print(f"\n 警告：每类需要 {samples_per_class} 张，但最少类别只有 {min_count} 张")
            print(f"   将使用 {min_count} 张/类，总数 = {min_count * 4}")
            samples_per_class = min_count
        else:
            print(f"\n 按指定数量采样：每类 {samples_per_class} 张，总数 {samples_per_class * 4}")

    # ===== 从每个类别采样 =====
    sampled_pairs = []
    
    for cat, items in categories.items():
        if len(items) >= samples_per_class:
            sampled = random.sample(items, samples_per_class)
        else:
            # 如果不够，就全部取（但理论上不应该发生，因为有 min_count 保证）
            sampled = items
            print(f"   {cat} 只有 {len(items)} 张，不足 {samples_per_class}")
        
        sampled_pairs.extend(sampled)
        print(f"   {cat}: 采样 {len(sampled)} 张")

    # 打乱顺序
    random.shuffle(sampled_pairs)

    # ===== 拆分为三个列表 =====
    img_paths, mask_paths, modes = [], [], []
    
    for img, mask, mode in sampled_pairs:
        img_paths.append(img)
        mask_paths.append(mask)
        # 将类别名转换为数字标签（0,1,2,3）
        mode_to_int = {
            'ai_on_real': 0,
            'real_on_fake': 1,
            'random_ai': 2,
            'specific_position_ai': 3
        }
        modes.append(mode_to_int[mode])

    print(f"\n 最终采样: {len(img_paths)} 张 (4类平衡)\n")
    
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
    
#SwinSeg 改 SwinUNet
class SwinUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # =========================
        # Backbone
        # =========================
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            num_classes=0,
            in_chans=4,
            features_only=True
        )

        channels = self.backbone.feature_info.channels()
        # 一般是: [96, 192, 384, 768]

        # =========================
        # Decoder（U-Net风格）
        # =========================

        self.up3 = self._block(channels[3], channels[2])  # 768 → 384
        self.up2 = self._block(channels[2], channels[1])  # 384 → 192
        self.up1 = self._block(channels[1], channels[0])  # 192 → 96

        # 最后 refinement
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels[0], 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # =========================
        # Backbone features
        # =========================
        feats = self.backbone(x)

        f1, f2, f3, f4 = feats
        # shape:
        # f1: [B,56,56,96]
        # f2: [B,28,28,192]
        # f3: [B,14,14,384]
        # f4: [B,7,7,768]

        # 转成 CNN 格式
        f1 = f1.permute(0, 3, 1, 2)
        f2 = f2.permute(0, 3, 1, 2)
        f3 = f3.permute(0, 3, 1, 2)
        f4 = f4.permute(0, 3, 1, 2)

        # =========================
        # Decoder（U-Net skip）
        # =========================

        x = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up3(x) + f3   # skip

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up2(x) + f2

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up1(x) + f1

        # 再上采样回 224
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        x = self.final_conv(x)

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
    loop = tqdm(loader, desc='Training', leave=False)

    for x, mask, _ in loop:
        x, mask = x.to(device), mask.to(device)

        optimizer.zero_grad()
        pred = model(x)

        loss = criterion(pred, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        
        # 更新进度条显示当前 loss
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader.dataset)

def evaluate(model, loader, device,pred_threshold=0.5):
    model.eval()

    iou_total = 0
    count = 0

    # 四个类别的统计
    iou_ai_on_real = 0
    cnt_ai_on_real = 0
    
    iou_real_on_fake = 0
    cnt_real_on_fake = 0
    
    iou_random_ai = 0
    cnt_random_ai = 0
    
    iou_specific_ai = 0
    cnt_specific_ai = 0

    with torch.no_grad():
        for x, mask, mode in loader:
            x, mask = x.to(device), mask.to(device)

            pred = torch.sigmoid(model(x))
            pred = (pred > pred_threshold).float()

            for i in range(x.size(0)):
                inter = (pred[i] * mask[i]).sum()
                union = pred[i].sum() + mask[i].sum() - inter

                iou = inter / (union + 1e-6)

                iou_total += iou.item()
                count += 1

                # 修改为四个类别的统计
                if mode[i] == 0:      # ai_on_real
                    iou_ai_on_real += iou.item()
                    cnt_ai_on_real += 1
                elif mode[i] == 1:    # real_on_fake
                    iou_real_on_fake += iou.item()
                    cnt_real_on_fake += 1
                elif mode[i] == 2:    # random_ai
                    iou_random_ai += iou.item()
                    cnt_random_ai += 1
                elif mode[i] == 3:    # specific_position_ai
                    iou_specific_ai += iou.item()
                    cnt_specific_ai += 1

    return {
        "mean_iou": iou_total / count,
        "ai_on_real": iou_ai_on_real / max(cnt_ai_on_real,1),
        "real_on_fake": iou_real_on_fake / max(cnt_real_on_fake,1),
        "random_ai":iou_random_ai / max(cnt_random_ai,1),
        "specific_position_ai":iou_specific_ai / max(cnt_specific_ai,1),
    }

class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.train_losses = []
        self.mean_ious = []
        # 四个类别的 IoU 列表
        self.ai_on_real_ious = []
        self.real_on_fake_ious = []
        self.random_ai_ious = []
        self.specific_ai_ious = []

    # =========================
    # 1️⃣ Loss + IoU 曲线
    # =========================
    def plot_curves(self):
        epochs = range(1, len(self.train_losses)+1)

        plt.figure(figsize=(10,6))

        plt.plot(epochs, self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(epochs, self.mean_ious, label='Mean IoU', linewidth=2)
        plt.plot(epochs, self.ai_on_real_ious, label='AI on Real', linestyle='--')
        plt.plot(epochs, self.real_on_fake_ious, label='Real on Fake', linestyle='--')
        plt.plot(epochs, self.random_ai_ious, label='Random AI', linestyle='--')
        plt.plot(epochs, self.specific_ai_ious, label='Specific Position AI', linestyle='--')

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
        task_names = {
                    0: "AI on Real",
                    1: "Real on Fake", 
                    2: "Random AI",
                    3: "Specific Position AI"
                }
        with torch.no_grad():
            for x, mask, mode in dataloader:
                x = x.to(device)

                pred = torch.sigmoid(model(x))

                for i in range(x.size(0)):
                    samples.append((
                        x[i].cpu(),
                        mask[i].cpu(),
                        pred[i].cpu(),
                        mode[i].item()
                    ))

                if len(samples) >= num_samples:
                    break

        samples = samples[:num_samples]

        fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3*num_samples))

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        for i, (img, gt, pred, mode) in enumerate(samples):

            # ===== 还原 RGB =====
            img_rgb = img[:3] * std + mean
            img_rgb = torch.clamp(img_rgb, 0, 1)

            gt_map = gt.squeeze().numpy()
            pred_map = pred.squeeze().numpy()
            pred_bin = (pred_map > 0.5).astype(np.float32)

            # ===== 计算 IoU =====
            inter = (pred_bin * gt_map).sum()
            union = pred_bin.sum() + gt_map.sum() - inter
            iou = inter / (union + 1e-6)

            # ===== 任务名称 =====
            task_name = task_names.get(mode, f"Unknown({mode})")

            # ===== 原图 =====
            axes[i, 0].imshow(img_rgb.permute(1, 2, 0))
            axes[i, 0].set_title(f"{task_name}", fontsize=10)
            axes[i, 0].axis('off')

            # ===== GT =====
            axes[i, 1].imshow(gt_map, cmap='gray')
            axes[i, 1].set_title("GT Mask")
            axes[i, 1].axis('off')

            # ===== Heatmap =====
            axes[i, 2].imshow(pred_map, cmap='jet')
            axes[i, 2].set_title("Heatmap")
            axes[i, 2].axis('off')

            # ===== Binary =====
            axes[i, 3].imshow(pred_bin, cmap='gray')
            axes[i, 3].set_title(f"Binary\nIoU={iou:.2f}")
            axes[i, 3].axis('off')

            # ===== Overlay =====
            overlay = img_rgb.permute(1, 2, 0).numpy().copy()
            overlay[..., 0] += pred_bin * 0.5
            overlay = np.clip(overlay, 0, 1)

            axes[i, 4].imshow(overlay)
            axes[i, 4].set_title("Overlay")
            axes[i, 4].axis('off')

        plt.tight_layout()

        save_path = os.path.join(self.save_dir, f'{draw_num}_seg.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"分割可视化已保存: {save_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TRAIN_NUM = 280
    VAL_NUM = 40
    pred_threshold=0.3
    version="v3_multi_type_dataset"

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

    # model = SwinSeg().to(device)
    model = SwinUNet().to(device)
    criterion = SegLoss(device,pos_weight=20)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    best_iou = 0
    os.makedirs('./Mask_Model', exist_ok=True)
    vis = Visualizer(f'./Mask_Model/{version}')

    for epoch in range(20):
        # 训练进度条
        print(f"\n📚 Epoch {epoch+1}/20")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        vis.train_losses.append(train_loss)

        metrics = evaluate(model, val_loader, device,pred_threshold)
        vis.mean_ious.append(metrics['mean_iou'])
        vis.ai_on_real_ious.append(metrics['ai_on_real'])
        vis.real_on_fake_ious.append(metrics['real_on_fake'])
        vis.random_ai_ious.append(metrics['random_ai'])
        vis.specific_ai_ious.append(metrics['specific_position_ai'])

        print(
            f"Epoch {epoch+1:2d} | "
            f"Loss {train_loss:.4f} | "
            f"Mean IoU {metrics['mean_iou']:.4f} | "
            f"AI→Real {metrics['ai_on_real']:.4f} | "
            f"Real→AI {metrics['real_on_fake']:.4f} | "
            f"Random {metrics['random_ai']:.4f} | "
            f"Specific {metrics['specific_position_ai']:.4f}"
        )

        if metrics['mean_iou'] > best_iou:
            best_iou = metrics['mean_iou']
            torch.save(model.state_dict(), f'./Mask_Model/{version}/best_seg.pth')
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