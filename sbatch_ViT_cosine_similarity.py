#%% 
import numpy as np
import matplotlib.pyplot as plt

import torch as t
import torch.nn.functional as F

import os
import sys
sys.path.append('utils')

from utils.diffeo_container import diffeo_container


device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
print(f'Using {device} for inference')

# layer_num_list = list(range(6,44)) + [46]
layer_num_list = range(2,15)
read_path = '/vast/xj2173/diffeo/scratch_data/all_ViT_layers/ViT_b_16/'
ref_path =  '/vast/xj2173/diffeo/scratch_data/all_ViT_layers/ViT_b_16/reference/'
# save_path = '/vast/xj2173/diffeo/process_data/all_ViT_layers/cosine_similarity/'
save_path = '/vast/xj2173/diffeo/process_data/all_ViT_layers/ViT_b_16/cos_sim/'

# inv_grid = t.load(read_path + '15-100-4-4-3-224-224_inv_grid_sample.pt', map_location= t.device('cpu'))
# inv_diffeos = diffeo_container(224, 224)
# inv_diffeos.diffeos = inv_grid

token_index = 0

data_dir_list = []
ref_dir_list = []
for layer_num in layer_num_list:
    data_dir_list.append(sorted([s for s in os.listdir(read_path) if f'layer-{layer_num:02d}' in s and '14-50-4-4-3-224-224' in s]))
    num_of_images = len(data_dir_list[-1])
    ref_dir_list.append(sorted([s for s in os.listdir(ref_path) if f'layer-{layer_num:02d}' in s][:num_of_images]))

#%%
cosine_similarity_list = {}
for counter, (file_path_list, ref_path_list) in enumerate(zip(data_dir_list, ref_dir_list)):
    print(f'processing {layer_num_list[counter]}-th layer')
  
    activation = []
    ref_activation = []
    for file_path, ref_file_path in zip(file_path_list, ref_path_list):
        raw_data = t.load(read_path + file_path, map_location = t.device('cpu'))
        ref_data = t.load(ref_path + ref_file_path, map_location = t.device('cpu'))
        
        activation.append(F.normalize(raw_data[:,token_index,:].reshape(14, 50, -1), dim = -1))
        ref_activation.append(F.normalize(ref_data[:,token_index,:], dim = -1))
    
    activation = t.stack(activation) #img, strength, diffeo, channel, pixels
    ref_activation = t.stack(ref_activation).squeeze()

    cosine_similarity = t.einsum('ip, isdp -> isd', ref_activation, activation)
    t.save(cosine_similarity, save_path + 'cosine_similarity' +f'_layer-{int(layer_num_list[counter]):02d}.pt')

    # cosine_similarity_list[f'{layer_num_list[counter]}'] = t.mean(cosine_similarity, dim = (0,2))



# diffeo_amp_list = [0.01, 0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.45, 0.5]

# plt.figure()

# colors = plt.cm.viridis_r(np.linspace(0,1,len(cosine_similarity_list)))

# for i, key in enumerate(cosine_similarity_list.keys()):
#     layer_num = int(layer_num_list[i])
#     plt.plot(diffeo_amp_list, cosine_similarity_list[key], color=colors[i], label = f'layer {layer_num}')
#     plt.legend()
#     plt.xlabel(r'diffeo strength w/ L1 norm $|A|_1$')
#     plt.title('activation cosine similarity after naive inverse compared to no diffeo')
#     plt.ylabel(r'normalized averaged cosine similarity')

# plt.savefig(f'/vast/xj2173/diffeo/process_data/all_ViT_layers/cosine_similarity/all_layers.png')
# plt.close()
# %%
