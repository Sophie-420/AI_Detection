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

def get_balanced_samples(root_dir, total_count):
    """
    从 root_dir/real 和 root_dir/fake 中各抽取 total_count // 2 个样本
    """
    real_dir = os.path.join(root_dir, 'real')
    fake_dir = os.path.join(root_dir, 'fake')
    
    # 获取所有文件路径
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    num_per_class = total_count // 2
    
    # 检查样本量是否足够
    if len(real_files) < num_per_class or len(fake_files) < num_per_class:
        actual_min = min(len(real_files), len(fake_files))
        print(f"警告: 样本不足。请求每类 {num_per_class}，实际最大可用每类 {actual_min}")
        num_per_class = actual_min

    # 随机采样
    sampled_real = random.sample(real_files, num_per_class)
    sampled_fake = random.sample(fake_files, num_per_class)
    
    # 构建样本列表 (路径, 标签) - 假设 real=0, fake=1
    samples = [(p, 0) for p in sampled_real] + [(p, 1) for p in sampled_fake]
    
    # 打乱顺序
    random.shuffle(samples)
    return samples

# --- Dataset 定义 ---
class FrequencyDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: 格式为 [(path, label), ...] 的列表
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        # 1. 读取图片
        img_bgr = cv2.imread(path)
        if img_bgr is None:  # 防止坏图
            return torch.zeros((4, 224, 224)), label
             
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. 提取频域特征 (DCT)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_f = np.float32(gray)
        dct = cv2.dct(gray_f)
        dct_log = np.log(np.abs(dct) + 1e-6)

        dct_norm = cv2.normalize(dct_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dct_resized = cv2.resize(dct_norm, (224, 224))

        # 3. 预处理 RGB
        img_pil = Image.fromarray(img_rgb)
        if self.transform:
            img_tensor = self.transform(img_pil)
        
        # 标准化
        img_tensor = normalize(img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # 4. 预处理频域图并拼接
        freq_tensor = torch.from_numpy(dct_resized).float().unsqueeze(0) / 255.0
        combined_tensor = torch.cat([img_tensor, freq_tensor], dim=0) 

        return combined_tensor, label

# --- 可视化工具类 ---
class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.train_losses = []
        self.val_accs = []
        self.val_losses = []
        
    def plot_training_curves(self):
        """绘制训练损失和验证准确率曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 训练损失曲线
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        if len(self.val_losses) > 0:
            ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epochs', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss Curve', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 验证准确率曲线
        ax2.plot(epochs, self.val_accs, 'g-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epochs', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Validation Accuracy Curve', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"训练曲线已保存至: {os.path.join(self.save_dir, 'training_curves.png')}")
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=['Real', 'Fake']):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"混淆矩阵已保存至: {os.path.join(self.save_dir, 'confusion_matrix.png')}")
        
        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
    def visualize_predictions(self, model, dataloader, device, num_samples=8):
        """可视化模型预测结果：一半显示正确样本，一半显示错误样本"""
        model.eval()
        
        # 准备两个容器
        correct_list = []
        wrong_list = []
        half_target = num_samples // 2
        
        # 1. 收集样本
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                for i in range(images.size(0)):
                    img_cpu = images[i].cpu()
                    true_l = labels[i].item()
                    pred_l = preds[i].item()
                    
                    # 归类逻辑
                    if true_l == pred_l and len(correct_list) < half_target:
                        correct_list.append((img_cpu, true_l, pred_l))
                    elif true_l != pred_l and len(wrong_list) < (num_samples - half_target):
                        wrong_list.append((img_cpu, true_l, pred_l))
                
                # 如果收集齐了就提前停止遍历 dataloader
                if len(correct_list) >= half_target and len(wrong_list) >= (num_samples - half_target):
                    break

        # 合并最终要显示的样本
        samples_to_show = correct_list + wrong_list
        actual_num = len(samples_to_show)
        
        if actual_num == 0:
            print("警告：没有可显示的样本。")
            return

        # 2. 开始绘图
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        # 反标准化设置
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        for i in range(num_samples):
            if i < actual_num:
                img_tensor, true_label_idx, pred_label_idx = samples_to_show[i]
                
                # 提取 RGB 通道并反标准化
                img_rgb = img_tensor[:3] * std + mean
                img_rgb = torch.clamp(img_rgb, 0, 1)
                
                axes[i].imshow(img_rgb.permute(1, 2, 0))
                
                true_str = 'Real' if true_label_idx == 0 else 'Fake'
                pred_str = 'Real' if pred_label_idx == 0 else 'Fake'
                
                # 确定颜色和状态标题
                is_correct = (true_label_idx == pred_label_idx)
                color = 'green' if is_correct else 'red'
                tag = "[Correct]" if is_correct else "[Mistake]"
                
                axes[i].set_title(f'{tag}\nTrue: {true_str}\nPred: {pred_str}', 
                                 color=color, fontsize=10)
            else:
                axes[i].axis('off') # 如果错误样本不够，剩下的格子留空
            axes[i].axis('off')

        plt.tight_layout()
        save_name = 'sample_analysis.png' if len(wrong_list) > 0 else 'sample_predictions.png'
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"对比可视化已保存至: {os.path.join(self.save_dir, save_name)}")
    
    def plot_feature_comparison(self, model, dataloader, device):
        """可视化真实和伪造图像的特征对比"""
        model.eval()
        real_features = []
        fake_features = []
        
        # 注册钩子来获取特征
        features = []
        def hook_fn(module, input, output):
            features.append(output.detach())
        
        # 在最后一个卷积层后注册钩子
        handle = model.patch_embed.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                _ = model(images)
                
                # 提取特征
                feat = features[-1].mean(dim=[1, 2])  # 全局平均池化
                
                for i, label in enumerate(labels):
                    if label == 0 and len(real_features) < 100:
                        real_features.append(feat[i].cpu().numpy())
                    elif label == 1 and len(fake_features) < 100:
                        fake_features.append(feat[i].cpu().numpy())
                
                if len(real_features) >= 100 and len(fake_features) >= 100:
                    break
        
        handle.remove()
        
        # 降维可视化（使用PCA）
        from sklearn.decomposition import PCA
        
        real_features = np.array(real_features)
        fake_features = np.array(fake_features)
        all_features = np.vstack([real_features, fake_features])
        labels = np.array([0] * len(real_features) + [1] * len(fake_features))
        
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(all_features)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[labels==0, 0], features_2d[labels==0, 1], 
                             c='blue', label='Real', alpha=0.6, s=50)
        scatter = plt.scatter(features_2d[labels==1, 0], features_2d[labels==1, 1], 
                             c='red', label='Fake', alpha=0.6, s=50)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
        plt.title('Feature Space Visualization (PCA)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'feature_space.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"特征空间可视化已保存至: {os.path.join(self.save_dir, 'feature_space.png')}")

# --- 评估函数 ---
def evaluate_model(model, dataloader, device):
    """评估模型并返回所有预测和标签"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds)

# --- 主训练代码 ---
def main():
    # 1. 基础配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")

    # 路径设置
    TRAIN_PATH = r'D:\DDA4210\AI_Detection\dataset\train'
    VAL_PATH = r'D:\DDA4210\AI_Detection\dataset\val'
    SAVE_PATH = r'D:\DDA4210\AI_Detection\models'
    VISUALIZATION_PATH = os.path.join(SAVE_PATH, 'visualizations')
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(VISUALIZATION_PATH, exist_ok=True)

    # 训练参数
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 5e-5

    # 数据预处理设置
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]),
    }

    # 采样数据
    TRAIN_COUNT = 500  # 总训练集数量
    VAL_COUNT = 100    # 总验证集数量

    print("正在加载数据集...")
    train_samples = get_balanced_samples(TRAIN_PATH, TRAIN_COUNT)
    val_samples = get_balanced_samples(VAL_PATH, VAL_COUNT)

    # 创建 Dataset 和 DataLoader
    train_dataset = FrequencyDataset(train_samples, transform=data_transforms['train'])
    val_dataset = FrequencyDataset(val_samples, transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"训练集规模: {len(train_dataset)}, 验证集规模: {len(val_dataset)}")

    # 初始化模型
    print("正在加载预训练模型...")
    model = timm.create_model(
        'swin_tiny_patch4_window7_224', 
        pretrained=True, 
        num_classes=2, 
        in_chans=4
    )
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # 初始化可视化器
    visualizer = Visualizer(VISUALIZATION_PATH)

    # 训练循环
    best_acc = 0.0
    
    print("\n开始训练...")
    for epoch in range(EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*50}")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_corrects = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == labels.data)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        epoch_train_loss = train_loss / len(train_dataset)
        epoch_train_acc = train_corrects.double() / len(train_dataset)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
        
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = val_corrects.double() / len(val_dataset)
        
        # 记录数据
        visualizer.train_losses.append(epoch_train_loss)
        visualizer.val_losses.append(epoch_val_loss)
        visualizer.val_accs.append(epoch_val_acc)
        
        # 更新学习率
        scheduler.step(epoch_val_loss)
        
        print(f"\n训练损失: {epoch_train_loss:.4f} | 训练准确率: {epoch_train_acc:.4f}")
        print(f"验证损失: {epoch_val_loss:.4f} | 验证准确率: {epoch_val_acc:.4f}")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_acc,
            }, os.path.join(SAVE_PATH, 'best_swin_model.pth'))
            print(f"✓ 已保存新的最优模型，准确率: {best_acc:.4f}")
    
    print("\n" + "="*50)
    print("训练完成！")
    print(f"最佳验证准确率: {best_acc:.4f}")
    print("="*50)
    
    # 加载最佳模型进行最终评估
    checkpoint = torch.load(os.path.join(SAVE_PATH, 'best_swin_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 最终评估和可视化
    print("\n正在生成可视化结果...")
    
    # 1. 绘制训练曲线
    visualizer.plot_training_curves()
    
    # 2. 混淆矩阵和分类报告
    y_true, y_pred = evaluate_model(model, val_loader, device)
    visualizer.plot_confusion_matrix(y_true, y_pred)
    
    # 3. 样本预测可视化
    visualizer.visualize_predictions(model, val_loader, device, num_samples=8)
    
    # 4. 特征空间可视化（可选，较耗时）
    print("\n生成特征空间可视化...")
    visualizer.plot_feature_comparison(model, val_loader, device)
    
    print(f"\n所有可视化结果已保存至: {VISUALIZATION_PATH}")

if __name__ == "__main__":
    main()