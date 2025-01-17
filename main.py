import argparse
import yaml
import numpy as np
import torch
from train.train_cnet import train_cnet
from train.train_unet import train_unet
from train.evaluate import evaluate_model
from data.data_loader import prepare_data
from models.unet1d import Unet1D
from models.cnet1d import Cnet1D
from utils.helpers import seed_everything
from torch.utils.data import DataLoader, TensorDataset
import warnings
# 忽略所有的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/base.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # 固定随机数
    seed_everything(config["seed"])
    
    # 加载数据集
    train_loader, val_loader, _ = prepare_data(config)
    
    # 加载新制作的数据集
    train_name = "training.npy"
    val_name = "valing.npy"
    test_name = "testing.npy"
    loaded_traindata = np.load(train_name, allow_pickle=True).item()
    loaded_valdata = np.load(val_name, allow_pickle=True).item()
    loaded_testdata = np.load(test_name, allow_pickle=True).item()    

    # # 训练集
    # # 转换为 PyTorch TensorDataset 和 DataLoader
    # train_clean_tensor = torch.tensor(loaded_traindata["clean"], dtype=torch.float32)
    # train_noise_tensor = torch.tensor(loaded_traindata["noise"], dtype=torch.float32)
    # train_transformed_tensor = torch.tensor(loaded_traindata["transformed_noise"], dtype=torch.float32)
    # # 创建两个数据集
    # train_dataset_original = torch.utils.data.TensorDataset(train_clean_tensor, train_noise_tensor)
    # train_dataset_transformed = torch.utils.data.TensorDataset(train_clean_tensor, train_transformed_tensor)  
    # # 合并数据集
    # train_combined_dataset = torch.utils.data.ConcatDataset([train_dataset_original, train_dataset_transformed])   
    # # 创建新的 DataLoader
    # train_loader = torch.utils.data.DataLoader(train_combined_dataset, batch_size=config["train"]["batch_size"], shuffle=True, drop_last=True, num_workers=0)

    # #验证集
    # # 转换为 PyTorch TensorDataset 和 DataLoader
    # val_clean_tensor = torch.tensor(loaded_valdata["clean"], dtype=torch.float32)
    # val_noise_tensor = torch.tensor(loaded_valdata["noise"], dtype=torch.float32)
    # val_transformed_tensor = torch.tensor(loaded_valdata["transformed_noise"], dtype=torch.float32)
    # # 创建两个数据集
    # val_dataset_original = torch.utils.data.TensorDataset(val_clean_tensor, val_noise_tensor)
    # val_dataset_transformed = torch.utils.data.TensorDataset(val_clean_tensor, val_transformed_tensor)  
    # # 合并数据集
    # val_combined_dataset = torch.utils.data.ConcatDataset([val_dataset_original, val_dataset_transformed])   
    # # 创建新的 DataLoader
    # val_loader = torch.utils.data.DataLoader(val_combined_dataset, batch_size=config["train"]["batch_size"], shuffle=False, drop_last=True, num_workers=0)

    #测试集
    # 转换为 PyTorch TensorDataset 和 DataLoader
    test_clean_tensor = torch.tensor(loaded_testdata["clean"], dtype=torch.float32)
    test_noise_tensor = torch.tensor(loaded_testdata["noise"], dtype=torch.float32)
    test_transformed_tensor = torch.tensor(loaded_testdata["transformed_noise"], dtype=torch.float32)

    # 创建两个数据集
    test_dataset_original = torch.utils.data.TensorDataset(test_clean_tensor, test_noise_tensor)
    test_dataset_transformed = torch.utils.data.TensorDataset(test_clean_tensor, test_transformed_tensor)  
    # 合并数据集
    # train_combined_dataset = torch.utils.data.ConcatDataset([train_dataset_original, train_dataset_transformed])   
    # 创建新的 DataLoader
    test_loader = torch.utils.data.DataLoader(test_dataset_transformed, batch_size=config["test"]["batch_size"], drop_last=True, num_workers=0)
    
    
    # 模型设置
    cnet = Cnet1D(dim=64, dim_mults=(1, 2, 4), channels=1, self_condition=config['train']["cnet_condition"])
    unet = Unet1D(dim=64, dim_mults=(1, 2, 4), channels=1, self_condition=config['train']["unet_condition"])
    
    # 加载训练评估方式
    if config['train']['train_cnet']:
        train_cnet(cnet, train_loader, val_loader, config)
    
    if config['train']['train_unet']:
        train_unet(unet, cnet, train_loader, val_loader, config)
    
    if config['test']['evaluate_model']:
        evaluate_model(unet, cnet, test_loader, config)

if __name__ == "__main__":
    main()

