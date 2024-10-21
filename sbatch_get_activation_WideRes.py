#%%
import torch
import torchvision
from torchvision.transforms import v2
import matplotlib.pyplot as plt

from utils.diffeo_container import sparse_diffeo_container
from utils.get_model_activation import retrieve_layer_activation, get_flatten_children

import sys
sys.path.append('/scratch/cm6627/diffeo_cnn/experiment/006_RandomLabels/fitting-random-labels')

import model_wideresnet


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


## Clark's scrambled label models

class ModelWeights:
    
    path = '/scratch/cm6627/diffeo_cnn/experiment/006_RandomLabels/ModelWeights/'
    EpochsAmount = [0, 60, 120, 180, 240]
    CorruptAmount = [0.0, 0.5, 1.0]

    @staticmethod
    def load_Model(corrupt: float, epochs: int) -> 'torch.model':
        ### Checks
        if corrupt not in ModelWeights.CorruptAmount:
            raise ValueError(f'`corrupt` must be: {ModelWeights.CorruptAmount}')
        if epochs not in ModelWeights.EpochsAmount:
            raise ValueError(f'`epochs` must be {ModelWeights.EpochsAmount}')

        ### Code
        epochs = str(int(epochs))
        if corrupt == 0.0:
            corrupt = '0p0'
        elif corrupt == 0.5:
            corrupt = '0p5'
        elif corrupt == 1.0:
            corrupt = '1p0'

        if epochs == str(int(0)):  # This is just a randomly initalized model 
            corrupt = '0p0'

        file_name = f'Corrupt-{corrupt}/ModelWeights_{epochs}Epochs.pth'

        # I trained on these parameters, which are default to the paper's code
        depth = 28
        classes = 10
        widen_factor = 1
        drop_rate = 0
        model = model_wideresnet.WideResNet(depth, classes,
                                            widen_factor,
                                            drop_rate=drop_rate)

        model_weights_path = ModelWeights.path + file_name
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        
        return model

#%%
EpochsAvaliable = [0, 60, 120, 180, 240]
CorruptOptions = [0.0, 0.5, 1.0]
layers = [2,3,4,7]
num_of_diffeo = 20
diffeo_strength_list = [0.001, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
batch_size = 200

## load data
normalize = v2.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
inference_trans = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), normalize])

cifar_data = torchvision.datasets.CIFAR10('/vast/xj2173/diffeo/CIFAR10',train = False, download = True, transform=inference_trans)
data_loader = torch.utils.data.DataLoader(cifar_data,
                                          batch_size=batch_size,
                                          shuffle=True)


## create diffeos
diffeos = sparse_diffeo_container(32, 32)
for strength in diffeo_strength_list:
    diffeos.sparse_AB_append(3,3,3,strength,num_of_diffeo)
diffeos.get_all_grid()
diffeos.to(device)

diffeos.get_inverse_grid(base_learning_rate=10)
torch.save(diffeos.inverse, f'/vast/xj2173/diffeo/scratch_data/inv_grids/13-{num_of_diffeo}-3-3-3-32-32_inv_grid_sample.pt')

save_path = '/vast/xj2173/diffeo/scratch_data/WideResNEt_layers/'
ref_path = '/vast/xj2173/diffeo/scratch_data/WideResNEt_layers/reference/'
        
images, _ = next(iter(data_loader))
images = images.to(device)
deformed = diffeos(images.unsqueeze(1).expand(-1,num_of_diffeo,-1,-1,-1))

#%%
for i, label_corruption in enumerate(CorruptOptions):
    for j, model_epoch in enumerate(EpochsAvaliable):
        model = ModelWeights.load_Model(corrupt = label_corruption, epochs= model_epoch)
        model.eval()
        model = model.to(device)
        for param in model.parameters():
            param.requires_grad = False
        
        file_prefix = f'13-{num_of_diffeo}-3-3-3-32-32_label_corrupt{i}-epoch{j}_activation'

        deformed = deformed.reshape(-1, 3, 32, 32)
        activation, _ = retrieve_layer_activation(model, deformed, layers)
        ref_activation, _ = retrieve_layer_activation(model, images, layers)

        for key in activation.keys():
            shape = activation[key].shape[1:]
            new_shape = (batch_size, 13, num_of_diffeo,) + shape
            activation[key] = activation[key].reshape(new_shape)
            torch.save(activation[key], save_path + file_prefix + f'_layer-{int(key):02d}.pt')
            torch.save(ref_activation[key], ref_path + f'label_corrupt{i}-epoch{j}_activation_layer-{int(key):02d}.pt')
        
        del model
# %%
print(get_flatten_children(model))
# %%
