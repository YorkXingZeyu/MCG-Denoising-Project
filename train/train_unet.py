import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam
from utils.confidence_map import compute_confidence_map

def train_unet(unet, cnet, train_loader, val_loader, config):
    # 路径配置
    foldername = config["output_folder"]
    model_folder = os.path.join(foldername, "model_params")
    os.makedirs(model_folder, exist_ok=True)
    best_model_path = os.path.join(model_folder, "Unet1D_best.pth")
    final_model_path = os.path.join(model_folder, "Unet1D_final.pth")
    cnet_model_path = os.path.join(model_folder, "Cnet1D_best.pth")

    device = config["device"]
    
    # 加载 CNet1D_best 预训练模型
    if os.path.exists(cnet_model_path):
        cnet.load_state_dict(torch.load(cnet_model_path, map_location=device, weights_only=True))
        print(f"CNet weights loaded from {cnet_model_path}")
    else:
        raise FileNotFoundError(f"CNet weights not found at {cnet_model_path}")

    unet.to(device)
    cnet.to(device)

    # 优化器 & 学习率调度器
    optimizer = Adam(
        unet.parameters(),
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
        unet.train()
        cnet.eval()  # 确保 cnet 不更新权重
        avg_loss = 0
        

        with tqdm(train_loader, desc=f"Training Unet Epoch {epoch}") as t:
            for clean_batch, noise_batch in t:
                clean_batch, noise_batch = clean_batch.to(device), noise_batch.to(device)
                optimizer.zero_grad()

                # 使用条件网络生成条件
                with torch.no_grad():
                    # # 生成置信度图
                    # confidence_maps_noise = torch.stack(
                    #     [compute_confidence_map(noise, device, config) for noise in noise_batch]
                    # ) * config["confidence"]["confidence_weight"]

                    condition = cnet(noise_batch)

                # 前向传播
                output = unet(noise_batch, x_self_cond=condition)


                # 频域损失
                output_fft = torch.fft.fft(output, dim=-1)
                clean_fft = torch.fft.fft(clean_batch, dim=-1)
                freq_diff = torch.abs(output_fft - clean_fft)
                freq_loss = torch.log1p(torch.mean(freq_diff))

                # 总损失
                loss = F.mse_loss(output, clean_batch) + freq_loss * config["train"]["freq_loss_weight"]
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                t.set_postfix(avg_loss=avg_loss / len(train_loader))

        # 学习率调度
        lr_scheduler.step()
        avg_epoch_loss = avg_loss / len(train_loader)
        avg_weighted_loss = avg_epoch_loss * config["train"]["unetloss_print_weight"]
        print(f"Epoch {epoch}: Avg Loss: {avg_weighted_loss:.4f}")

        # 验证
        val_loss = validate_unet(unet, cnet, val_loader, device, config)
        val_weighted_loss = val_loss * config["train"]["unetloss_print_weight"]
        print(f"Validation Loss: {val_weighted_loss:.4f}")

        # 保存最优模型
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(unet.state_dict(), best_model_path)
            best_weighted_loss = best_loss * config["train"]["unetloss_print_weight"]
            print(f"Best Validation Loss updated to {best_weighted_loss:.4f}. Model saved.")
        else:
            counter += 1
            if counter >= config["train"]["early_stopping"]:
                print("Early stopping triggered.")
                break

    # 保存最终模型
    torch.save(unet.state_dict(), final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")

def validate_unet(unet, cnet, val_loader, device, config):
    unet.eval()
    cnet.eval()
    total_loss = 0

    with torch.no_grad():
        for clean_batch, noise_batch in val_loader:
            clean_batch, noise_batch = clean_batch.to(device), noise_batch.to(device)

            # 生成置信度图
            # confidence_maps_noise = torch.stack(
            #     [compute_confidence_map(noise, device, config) for noise in noise_batch]
            # ) * config["confidence"]["confidence_weight"]

            # 使用条件网络生成条件
            condition = cnet(noise_batch)
            output = unet(noise_batch, x_self_cond=condition)
            
            # 频域损失
            output_fft = torch.fft.fft(output, dim=-1)
            clean_fft = torch.fft.fft(clean_batch, dim=-1)
            freq_diff = torch.abs(output_fft - clean_fft)
            freq_loss = torch.log1p(torch.mean(freq_diff))

            # 总损失
            loss = F.mse_loss(output, clean_batch) + freq_loss * config["train"]["freq_loss_weight"]
            total_loss += loss.item()

    return total_loss / len(val_loader)
