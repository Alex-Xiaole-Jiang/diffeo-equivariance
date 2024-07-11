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

save_path = 'vast/diffeo/data/reference/'

imagenet_val_loader = iter(t.utils.data.DataLoader(imagenet_val_inference, batch_size = 1, shuffle=False))

layers = list(range(1,50))

for i in range(len(imagenet_val_loader)):
  file_prefix = f'val_image-{i:04d}_activation'
  val_image, _ = next(imagenet_val_loader)
  retrieve_layer_activation(ENV2, val_image, layers)
  for key in activation.keys():
    t.save(activation[key], save_path + file_prefix + f'_layer-{int(key):02d}.pt')
  activation = {}
  handle = []
  print(f'{i+1}th image completed')
