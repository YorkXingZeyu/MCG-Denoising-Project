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
    train_loader, val_loader, test_loader = prepare_data(config)
    
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

