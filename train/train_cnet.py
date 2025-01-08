import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam
from utils.confidence_map import compute_confidence_map

def train_cnet(cnet, train_loader, val_loader, config):
    # 路径配置
    foldername = config["output_folder"]
    model_folder = os.path.join(foldername, "model_params")
    os.makedirs(model_folder, exist_ok=True)
    best_model_path = os.path.join(model_folder, "Cnet1D_best.pth")
    final_model_path = os.path.join(model_folder, "Cnet1D_final.pth")

    device = config["device"]
    cnet.to(device)

    # 优化器和学习率调度器
    optimizer = Adam(
        cnet.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"]  # 添加权重衰减
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config["train"]["step_size"], 
        gamma=config["train"]["gamma"]
    )

    best_loss = float("inf")
    counter = 0

    for epoch in range(config["train"]["epochs"]):
        cnet.train()
        avg_loss = 0

        # 训练循环
        with tqdm(train_loader, desc=f"Training CNet Epoch {epoch}") as t:
            for clean_batch, noise_batch in t:
                clean_batch, noise_batch = clean_batch.to(device), noise_batch.to(device)
                optimizer.zero_grad()

                # 生成置信度图
                confidence_maps_clean = torch.stack(
                    [compute_confidence_map(clean, device, config) for clean in clean_batch]
                ) * config["confidence"]["confidence_weight"]
                # confidence_maps_noise = torch.stack(
                #     [compute_confidence_map(noise, device, config) for noise in noise_batch]
                # ) * config["confidence"]["confidence_weight"]

                # 前向传播和损失计算
                output = cnet(noise_batch)
                loss = F.mse_loss(output, confidence_maps_clean)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                t.set_postfix(avg_loss=avg_loss / len(train_loader))

        # 学习率调度
        lr_scheduler.step()
        avg_epoch_loss = avg_loss / len(train_loader)
        avg_weighted_loss = avg_epoch_loss * config["train"]["cnetloss_print_weight"]
        print(f"Epoch {epoch}: Avg Train Loss: {avg_weighted_loss:.4f}")

        # 验证
        val_loss = validate_cnet(cnet, val_loader, device, config) 
        val_weighted_loss = val_loss * config["train"]["cnetloss_print_weight"]
        print(f"Epoch {epoch}: Validation Loss: {val_weighted_loss:.4f}")

        # 保存最优模型
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(cnet.state_dict(), best_model_path)
            best_weighted_loss = best_loss*config["train"]["cnetloss_print_weight"]
            print(f"Best Validation Loss updated to {best_weighted_loss:.4f}. Model saved.")
        else:
            counter += 1
            if counter >= config["train"]["early_stopping"]:
                print("Early stopping triggered.")
                break

    # 保存最终模型
    torch.save(cnet.state_dict(), final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")

def validate_cnet(cnet, val_loader, device, config):
    cnet.eval()
    total_loss = 0

    with torch.no_grad():
        for clean_batch, noise_batch in val_loader:
            clean_batch, noise_batch = clean_batch.to(device), noise_batch.to(device)

            # 生成置信度图
            confidence_maps_clean = torch.stack(
                [compute_confidence_map(clean, device, config) for clean in clean_batch]
            ) * config["confidence"]["confidence_weight"]
            # confidence_maps_noise = torch.stack(
            #     [compute_confidence_map(noise, device, config) for noise in noise_batch]
            # ) * config["confidence"]["confidence_weight"]

            # 前向传播和损失计算
            output = cnet(noise_batch)
            loss = F.mse_loss(output, confidence_maps_clean)
            total_loss += loss.item()

    return total_loss / len(val_loader)
