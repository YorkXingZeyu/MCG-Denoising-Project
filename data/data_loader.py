import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Subset
from Data_Preparation.data_preparation import Data_Prepar_1, Data_Prepar_2, Data_Prepar_3, Data_Prepar_4

def prepare_data(config):

    foldername = config["output_folder"]
    
    if foldername == "MCG_019_result":
        X_train, y_train, X_test, y_test = Data_Prepar_1()
    elif foldername == "MCG_101_result":
        X_train, y_train, X_test, y_test = Data_Prepar_2()
    elif foldername == "ECG_Data_result":
        X_train, y_train, X_test, y_test = Data_Prepar_3()
    elif foldername == "MCG_self_result":
        X_train, y_train, X_test, y_test = Data_Prepar_4()
    else:
        raise ValueError(f"Unknown output_folder: {foldername}. Please provide a valid output_folder in config.")

    X_train, X_test = torch.FloatTensor(X_train), torch.FloatTensor(X_test)
    y_train, y_test = torch.FloatTensor(y_train), torch.FloatTensor(y_test)

    # Permute to match input requirements
    X_train, y_train = X_train.permute(0, 2, 1), y_train.permute(0, 2, 1)
    X_test, y_test = X_test.permute(0, 2, 1), y_test.permute(0, 2, 1)

    train_val_set = TensorDataset(y_train, X_train)
    test_set = TensorDataset(y_test, X_test)

    train_idx, val_idx = train_test_split(list(range(len(train_val_set))), test_size=0.2)
    train_set = Subset(train_val_set, train_idx)
    val_set = Subset(train_val_set, val_idx)

    train_loader = DataLoader(train_set, batch_size=config["train"]["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config["train"]["batch_size"], drop_last=True)
    test_loader = DataLoader(test_set, batch_size=config["test"]["batch_size"])

    return train_loader, val_loader, test_loader
