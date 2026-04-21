import torch
from torch.utils.data import DataLoader
from mask_train_model import SwinSeg, MaskDataset, evaluate, Visualizer, sample_dataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TEST_NUM = 90  #可以手动

    # =========================
    # 1️⃣ 加载 test 数据
    # =========================
    test_imgs, test_masks, test_modes = sample_dataset(
        './dataset_mask/test/images',
        './dataset_mask/test/masks',
        TEST_NUM
    )

    test_ds = MaskDataset(test_imgs, test_masks, modes=test_modes)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    # =========================
    # 2️⃣ 加载模型
    # =========================
    model = SwinSeg().to(device)
    model.load_state_dict(torch.load('./Mask_Model/best_seg.pth', map_location=device))

    # =========================
    # 3️⃣ 评估
    # =========================
    metrics = evaluate(model, test_loader, device)

    print("\n===== TEST RESULT =====")
    print(f"Mean IoU      : {metrics['mean_iou']:.4f}")
    print(f"AI → Real     : {metrics['ai_on_real']:.4f}")
    print(f"Real → AI     : {metrics['real_on_fake']:.4f}")

    # =========================
    # 4️⃣ 可视化
    # =========================
    vis = Visualizer('./Mask_Model/test_vis')

    vis.visualize_segmentation(
        model,
        test_loader,
        device,
        num_samples=10,
        draw_num=0
    )

    print("✅ Test 完成")

if __name__ == "__main__":
    main()