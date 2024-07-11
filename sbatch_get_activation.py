import torch as t
import torchvision as tv
from torchvision.transforms import v2

from utils.diffeo_container import sparse_diffeo_container

import sys
sys.path.append('utils')



activation = {}
def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
handles = []

def retrieve_layer_activation(model, input, layer_index):
  if len(input) == 3: input = input[None, :, :, :]

  layers = list(model.children())
  layers_flat = flatten(layers)

  for index in layer_index:
    handles.append(layers_flat[index - 1].register_forward_hook(getActivation(str(index))))

  with t.no_grad(): model(input)
  for handle in handles: handle.remove()

  return

def flatten(array):
    result = []
    for element in array:
        if hasattr(element, "__iter__"):
            result.extend(flatten(element))
        else:
            result.append(element)
    return result


device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
print(device)

ENV2 = tv.models.efficientnet_v2_s(weights = tv.models.EfficientNet_V2_S_Weights.DEFAULT).to(device)
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
diffeo_strength_list = [0.01, 0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.45, 0.5]
for strength in diffeo_strength_list:
    sparse_diffeos.sparse_AB_append(4, 4, 3, strength, 100)
sparse_diffeos.get_all_grid()
sparse_diffeos.to(device)

print('diffeo computed')

save_path = 'vast/diffeo/data/all_cnn_layers/'

imagenet_val_loader = iter(t.utils.data.DataLoader(imagenet_val_inference, batch_size = 1, shuffle=False))

layers = list(range(3,44)) + [46] + [49]

for i in range(100):
  file_prefix = f'15-100-4-4-3-224-224_image-{i:04d}_activation'
  val_image, _ = next(imagenet_val_loader)
  grid_sample = t.cat(sparse_diffeos.diffeos)
  distorted_list = t.nn.functional.grid_sample(val_image.repeat(15 * 100,1,1,1).to(device), grid_sample, mode = 'bilinear')
  retrieve_layer_activation(ENV2, distorted_list, layers)
  for key in activation.keys():
    t.save(activation[key], save_path + file_prefix + f'_layer-{int(key):02d}.pt')
  activation = {}
  handle = []
  print(f'{i+1}th image completed')
