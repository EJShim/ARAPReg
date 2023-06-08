import vtk
import numpy as np
import torch

if __name__ == "__main__":
    train_data = torch.tensor( np.load("data/DFaust/raw/dfaust/train.npy"))

    print(train_data.shape)
    

    train_data_mean = train_data.mean(dim=0)
    train_data_std = train_data.std(dim=0)

    print(train_data_mean.shape)
    print(train_data_std.shape)

    output = {
        'mean' : train_data_mean,
        'std' : train_data_std
    }

    torch.save(output, "data/DFaust/raw/meanstd.pt")
    