import torch
import os
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from train_model import FrequencyDataset, Visualizer, evaluate_model, get_balanced_samples

def run_test():
    # --- 基础配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = r'D:\DDA4210\AI_Detection\models\best_swin_model.pth'
    TEST_DATA_PATH = r'D:\DDA4210\AI_Detection\dataset\test' # 确保该目录下有 real/ 和 fake/
    RESULT_SAVE_PATH = r'D:\DDA4210\AI_Detection\models\test_results'
    os.makedirs(RESULT_SAVE_PATH, exist_ok=True)

    # --- 加载测试数据 ---
    print("正在加载测试集...")
    test_samples = get_balanced_samples(TEST_DATA_PATH, total_count=400) 
    
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    test_dataset = FrequencyDataset(test_samples, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # --- 加载模型 ---
    print(f"正在从 {MODEL_PATH} 加载模型...")
    model = timm.create_model(
        'swin_tiny_patch4_window7_224', 
        pretrained=False, 
        num_classes=2, 
        in_chans=4
    )
    
    # 加载保存的状态字典
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # --- 执行评估 ---
    print("开始测试...")
    visualizer = Visualizer(RESULT_SAVE_PATH)
    
    # 调用训练脚本里的评估函数
    y_true, y_pred = evaluate_model(model, test_loader, device)

    # --- 可视化 ---
    # 1. 混淆矩阵和分类报告
    print("\n生成混淆矩阵...")
    visualizer.plot_confusion_matrix(y_true, y_pred)
    
    # 2. 预测样本可视化（复用你的画图代码）
    print("生成样本预测图...")
    visualizer.visualize_predictions(model, test_loader, device, num_samples=8)
    
    # 3. 特征空间可视化（查看测试集在空间中的分布情况）
    print("生成特征空间 PCA 图...")
    visualizer.plot_feature_comparison(model, test_loader, device)

    print(f"\n测试完成！结果已保存至: {RESULT_SAVE_PATH}")

if __name__ == "__main__":
    run_test()