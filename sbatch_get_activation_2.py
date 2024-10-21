import torch as t
import torchvision as tv
from torchvision.transforms import v2

from utils.diffeo_container import sparse_diffeo_container
from utils.get_model_activation import retrieve_layer_activation

import sys
sys.path.append('utils')


device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
print(device)

ENV2 = tv.models.efficientnet_v2_s(weights = tv.models.EfficientNet_V2_S_Weights.DEFAULT).to(device)
# ENV2 = tv.models.efficientnet_v2_s().to(device) # random initialization
ENV2.eval();
for param in ENV2.parameters():
    param.requires_grad = False
inference_transform = tv.models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
inference_transforms = v2.Compose([
    lambda x: x.convert('RGB'),
    inference_transform,
])

imagenet_val_inference = tv.datasets.ImageNet('vast/diffeo/imagenet', split = 'val', transform = inference_transforms)

sparse_diffeos = sparse_diffeo_container(224, 224)
diffeo_strength_list = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
for strength in diffeo_strength_list:
    sparse_diffeos.sparse_AB_append(4, 4, 3, strength, 50)
sparse_diffeos.get_all_grid()
sparse_diffeos.to(device)

sparse_diffeos.get_inverse_grid(base_learning_rate=500)
t.save(sparse_diffeos.inverse, 'vast/diffeo/scratch_data/inv_grids/14-50-4-4-3-224-224_inv_grid_sample.pt')

print('diffeo & inverse computed', flush = True)

save_path = 'vast/diffeo/scratch_data/all_cnn_layers/ENV2_s/'
ref_path = 'vast/diffeo/scratch_data/all_cnn_layers/ENV2_s/reference/'

imagenet_val_loader = iter(t.utils.data.DataLoader(imagenet_val_inference, batch_size = 1, shuffle=False))

layers = list(range(3,44)) + [46] + [49]

for i in range(100):
  file_prefix = f'14-50-4-4-3-224-224_image-{i:04d}_activation'
  val_image, _ = next(imagenet_val_loader)
  val_image = val_image.to(device)
  # grid_sample = t.cat(sparse_diffeos.diffeos)
  # distorted_list = t.nn.functional.grid_sample(val_image.repeat(14 * 50,1,1,1).to(device), grid_sample, mode = 'bilinear')
  distorted_list = t.cat(sparse_diffeos(val_image.repeat(50, 1,1,1)))
  activation = retrieve_layer_activation(ENV2, distorted_list, layers)
  ref_activation = retrieve_layer_activation(ENV2, val_image, layers)
  for key in activation.keys():
    t.save(activation[key], save_path + file_prefix + f'_layer-{int(key):02d}.pt')
    t.save(ref_activation[key], ref_path + f'val_image-{i:04d}_activation_layer-{int(key):02d}.pt')
  activation = {}
  handle = []
  print(f'{i+1}th image completed', flush = True)
