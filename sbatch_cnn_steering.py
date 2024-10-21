#%%
import torch as t
import torch.nn.functional as F
import torchvision as tv
from torchvision.transforms import v2

from utils.diffeo_container import sparse_diffeo_container, diffeo_container
from utils.get_model_activation import get_flatten_children

from tqdm import tqdm

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
print(device)

torch_seed = 37
t.manual_seed(torch_seed)

# setting up path and config

ImageNet_path = '/vast/xj2173/diffeo/imagenet'
save_path = '/vast/xj2173/diffeo/scratch_data/steering/ENV2_s_no_inverse/'
ref_path = '/vast/xj2173/diffeo/scratch_data/steering/ENV2_s_no_inverse/reference/'

num_of_images = 100

diffeo_strengths = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# diffeo_strengths = [0.5]
num_of_diffeo = 20

# inv_diffeo_save_path = ('/vast/xj2173/diffeo/scratch_data/inv_grids/'
#                         f'{len(diffeo_strengths)}-{num_of_diffeo}-4-4-3-224-224_inv_grid_sample.pt')

steering_layers = list(range(6,44))
# steering_layers = [8]

# setting up helper function

def get_model():
  model = tv.models.efficientnet_v2_s(weights = tv.models.EfficientNet_V2_S_Weights.DEFAULT).to(device)
  # ENV2 = tv.models.efficientnet_v2_s().to(device) # random initialization
  model.eval();
  for param in model.parameters():
      param.requires_grad = False
  return model

def get_inference_transform():
  inference_transform = tv.models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
  inference_transforms = v2.Compose([
      lambda x: x.convert('RGB'),
      inference_transform,
  ])
  return inference_transforms

def get_ImageNet(transforms = None, batch_size = 1, shuffle = False):
  dataset = tv.datasets.ImageNet(ImageNet_path, split = 'val', transform = transforms)
  dataloader = t.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
  return dataset, dataloader

def get_diffeo_container(diffeo_strength_list = None, num_of_didffeo = None):
  diffeo_container = sparse_diffeo_container(224, 224)
  for strength in diffeo_strength_list:
      diffeo_container.sparse_AB_append(4, 4, 3, strength, num_of_didffeo)
  diffeo_container.get_all_grid()
  diffeo_container.to(device)
  return diffeo_container

def get_inverse_diffeo(diffeo_container, base_learning_rate = 500):
  inv_diffeo = diffeo_container.get_inverse_grid(base_learning_rate=base_learning_rate)
  return inv_diffeo


def get_steering_layer_shapes(model, layers):
  # get layer size
  input_size = (1,3,224,224) # standard ImageNet size
  dummy_input = t.rand(*input_size).to(device)
  hooks = []
  layer_shapes = []

  def get_output_size_hook(module, input, output):
    x, y = output.shape[-2:]
    layer_shapes.append((x,y))
    pass

  for layer in layers:
    hooks.append(layer.register_forward_hook(get_output_size_hook))
  
  with t.no_grad(): model(dummy_input)
  for hook in hooks: hook.remove()

  return layer_shapes

def get_steering_diff(diffeo, layer_shapes):
  down_sampled_diffeo = []
  for layer_shape in layer_shapes:
    x, y = layer_shape
    down_sampled_diffeo.append(diffeo.up_down_sample(x, y, align_corners=True))
  return down_sampled_diffeo

def diff_hook(diffeo, batch_size = None):
    if not isinstance(diffeo, diffeo_container):        
        raise Exception('diffeo is not a diffeo_container')
    def hook(module, input, output):
        if batch_size == None:
        # normal situation
            return diffeo(output, in_inference=True)
        if batch_size != None:
            # stack in the batch dimension of the steered result
            output = t.unflatten(output, 0, (-1, batch_size))
            ref = output[0]
            output = t.flatten(output, start_dim = 0, end_dim = 1)
            output = t.cat([output, diffeo(ref, in_inference = True)], dim = 0)
            return output
    return hook

def steering_hooks(layers, diffs):
  hooks = []
  for layer, diff in zip(layers, diffs):
    hooks.append(layer.register_forward_hook(diff_hook(diff, batch_size = len(diffeo_strengths) * num_of_diffeo)))
  return hooks

#%%
#
# where the code starts
model =get_model().to(device)

inf_transf = get_inference_transform()
dataset, dataloader = get_ImageNet(transforms = inf_transf, batch_size = 1, shuffle = True)
data_iter = iter(dataloader)

print('model & dataset setup finished', flush = True)
#%%
diffeos = get_diffeo_container(diffeo_strength_list = diffeo_strengths, num_of_didffeo = num_of_diffeo)
# try:
#   inv_diffeo = t.load(inv_diffeo_save_path, map_location = device)
# except:
#   inv_diffeo = get_inverse_diffeo(diffeos)
#   # t.save(inv_diffeo, inv_diffeo_save_path)

print('diffeo computed', flush = True)
#%%
val_images = t.cat([next(data_iter)[0] for i in range(num_of_images)], dim = 0).to(device)
ref_output = model(val_images)
t.save(ref_output, ref_path + f'val_image-first-{num_of_images}-images-output.pt')

print('reference output computed & saved', flush = True)
#%%
model_layer = get_flatten_children(model)
steering_model_layers = [model_layer[index] for index in steering_layers]
steering_layer_shapes = get_steering_layer_shapes(model, steering_model_layers)
steering_diffeo = get_steering_diff(diffeos, steering_layer_shapes)
for diffeo in steering_diffeo: diffeo.to(device)
hooks = steering_hooks(steering_model_layers, steering_diffeo)

print('diffeo down sampled', flush = True)
# layers = list(range(3,44)) + [46] + [49]
#%%
for i, image in enumerate(tqdm(val_images)):
  file_prefix = f'{len(diffeo_strengths)}-{num_of_diffeo}-4-4-3-224-224_image-{i:04d}_steered'
  layers_in_string = '-'.join(str(num) for num in steering_layers)

  # get a list of shape [strength * diffeo (batch), channel, x, y]
  # distorted_list = diffeos(image.repeat(num_of_diffeo * len(diffeo_strengths), 1,1,1), in_inference = True)
    
  with t.no_grad(): steered = model(image.expand(num_of_diffeo * len(diffeo_strengths), -1, -1, -1))
  # steered has shape [layer, strength, diffeo, -1]
  steered = t.reshape(steered, (len(steering_layers)+1, len(diffeo_strengths), num_of_diffeo, -1))
  t.save(steered, save_path + file_prefix + '_layer-' + layers_in_string +'.pt')
  
  # print(f'{i+1}th image completed', flush = True)

for hook in hooks: hook.remove()

# %%
print('yes')
# %%
