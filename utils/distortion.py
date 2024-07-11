# %%
import numpy as np
import torch
import torch as t
import torch.nn as nn
import math
from tqdm import tqdm
#from torch_perlin_noise import rand_perlin_2d_octaves
# import matplotlib.pyplot as plt

def get_version(): print('version channel mixing')

def dense_transform_amplitude(x_length, y_length, truncate = False, amplitude = 1, rng = None, seed = 37, alpha = None):
  '''
    returns x_length by y_length matrix where each element is random and has flattened norm of 1 (except the 0,0 element)
    x_length and y_length specifies the frequency bound
    amplitude changes the norm
    truncate zeros already generated higher frequency components
  '''
  if rng == None: rng = np.random.default_rng(seed = seed)

  A_sign = (rng.integers(2, size = x_length * y_length) - 0.5) * 2
  B_sign = (rng.integers(2, size = x_length * y_length) - 0.5) * 2

  if alpha == None:
    A_nm = amplitude * rng.dirichlet(np.ones(x_length * y_length - 1)) * A_sign
    B_nm = amplitude * rng.dirichlet(np.ones(x_length * y_length - 1)) * B_sign
  elif alpha != None:
    A_nm = amplitude * rng.dirichlet(alpha) * A_sign
    B_nm = amplitude * rng.dirichlet(alpha) * B_sign

  # A_nm = np.pad(A_nm, (1,0), 'constant').reshape(x_length, y_length)
  # B_nm = np.pad(B_nm, (1,0), 'constant').reshape(x_length, y_length)

  A_nm = A_nm.reshape(x_length, y_length)
  B_nm = B_nm.reshape(x_length, y_length)

  if truncate == False:
    return A_nm, B_nm

  A_nm = A_nm[0:truncate, 0:truncate]
  B_nm = B_nm[0:truncate, 0:truncate]

  return A_nm, B_nm



def sparse_transform_amplitude(x_cutoff, y_cutoff, num_of_terms, diffeo_amp = 1, rng = None, seed = 37, num_of_diffeo = 1, alpha = None):
  '''
    returns x_length by y_length matrix where num_of_terms elements is random and has flattened norm of 1
    in the case that num_of_terms << x_length * y_length, this matrix is sparse
    x_length and y_length specifies the frequency bound
    amplitude changes the norm
  '''
  if rng == None: rng = np.random.default_rng(seed = seed)

  A = []
  B = []

  total_num_elem = x_cutoff * y_cutoff
  pad_len = total_num_elem - num_of_terms #- 1

  for i in range(num_of_diffeo):

    A_sign = (rng.integers(2, size = num_of_terms) - 0.5) * 2
    B_sign = (rng.integers(2, size = num_of_terms) - 0.5) * 2

    if alpha == None:
      A_nm = diffeo_amp * rng.dirichlet(np.ones(num_of_terms)) * A_sign
      B_nm = diffeo_amp * rng.dirichlet(np.ones(num_of_terms)) * B_sign
    elif alpha != None:
      A_nm = diffeo_amp * rng.dirichlet(alpha * np.ones(num_of_terms)) * A_sign
      B_nm = diffeo_amp * rng.dirichlet(alpha * np.ones(num_of_terms)) * B_sign

    A_nm = np.pad(A_nm, (0, pad_len), 'constant')
    A_nm = rng.permutation(A_nm).reshape(x_cutoff, y_cutoff)
    # A_nm = np.pad(A_nm, (1,0), 'constant').reshape(x_length, y_length)
    A.append(t.Tensor(A_nm))

    B_nm = np.pad(B_nm, (0, pad_len), 'constant')
    B_nm = rng.permutation(B_nm).reshape(x_cutoff, y_cutoff)
    # B_nm = np.pad(B_nm, (1,0), 'constant').reshape(x_length, y_length)
    B.append(t.Tensor(B_nm))

  return t.stack(A), t.stack(B)



def create_grid_sample(x_length: int, y_length: int, A_list: torch.Tensor, B_list: torch.Tensor) -> torch.Tensor:
    '''
    Sin distortion for torch.nn.functional.grid_sample, the grid is from -1 to 1

    Args:
    - x_length (int): Length of x-axis of image.
    - y_length (int): Length of y-axis of image.
    - A_list (torch.Tensor): List of square matrices of coefficients, for x coordinate distortion
    - B_list (torch.Tensor): Same as A_list but for y coordinate distortion

    Returns:
    - torch.Tensor that has shape (N, x_length, y_length, 2) that can be fed into torch.nn.functional.grid_sample
    - the last dimension is length 2 because one is for x and one is for y.
    '''
    flow_grids = []

    for A_nm, B_nm in zip(A_list, B_list):
        non_zero_A_arg = torch.nonzero(A_nm, as_tuple=True)
        freq_A_arg = (torch.stack(non_zero_A_arg) + 1) * math.pi / 2
        max_A_freq = torch.max(torch.transpose(freq_A_arg, 1, 0), dim=1).values

        non_zero_B_arg = torch.nonzero(B_nm, as_tuple=True)
        freq_B_arg = (torch.stack(non_zero_B_arg) + 1) * math.pi / 2
        max_B_freq = torch.max(torch.transpose(freq_B_arg, 1, 0), dim=1).values

        unique_A_x, inv_index_A_x = torch.unique(freq_A_arg[1], return_inverse=True)
        unique_A_y, inv_index_A_y = torch.unique(freq_A_arg[0], return_inverse=True)
        unique_B_x, inv_index_B_x = torch.unique(freq_B_arg[1], return_inverse=True)
        unique_B_y, inv_index_B_y = torch.unique(freq_B_arg[0], return_inverse=True)

        x = torch.linspace(-1, 1, steps=x_length)
        y = torch.linspace(-1, 1, steps=y_length)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        normalization_A = 1 / max_A_freq
        normalization_B = 1 / max_B_freq

        x_basis_A = torch.sin(unique_A_x[None, None, :] * (X[:, :, None] + 1))
        y_basis_A = torch.sin(unique_A_y[None, None, :] * (Y[:, :, None] + 1))
        x_basis_B = torch.sin(unique_B_x[None, None, :] * (X[:, :, None] + 1))
        y_basis_B = torch.sin(unique_B_y[None, None, :] * (Y[:, :, None] + 1))

        eps = torch.finfo(torch.float64).eps * 10
        x_basis_A[torch.abs(x_basis_A) < eps] = 0
        y_basis_A[torch.abs(y_basis_A) < eps] = 0
        x_basis_B[torch.abs(x_basis_B) < eps] = 0
        y_basis_B[torch.abs(y_basis_B) < eps] = 0


        X_pert = torch.einsum('xyi, i, xyi, i -> xy', y_basis_A[:, :, inv_index_A_y], A_nm[non_zero_A_arg], x_basis_A[:, :, inv_index_A_x], normalization_A)
        Y_pert = torch.einsum('xyi, i, xyi, i -> xy', y_basis_B[:, :, inv_index_B_y], B_nm[non_zero_B_arg], x_basis_B[:, :, inv_index_B_x], normalization_B)

        x_map = X + X_pert
        y_map = Y + Y_pert

        flow_grid_tensor = torch.stack((y_map, x_map), dim=-1)
        flow_grids.append(flow_grid_tensor.unsqueeze(0))

    return torch.cat(flow_grids, dim=0)
#%%
class add_bias_to_grid(nn.Module):
  def __init__(self, grid):
    # grid should have the shape of a grid_sample grid, i.e. (Channel, X, Y, 2)
    super().__init__()
    self.grid = grid
    self.bias = nn.Parameter(t.zeros_like(grid[:,1:-1,1:-1,:]))
  def forward(self):
    return self.grid + nn.functional.pad(self.bias, (0,0,1,1,1,1), "constant", 0)
    
def find_inv_grid(flow_grid, mode ='bilinear', learning_rate = 0.001, epochs = 10000, early_stopping = True):
  device = flow_grid.device
  batch, x_length, y_length, _ = flow_grid.shape
  x = t.linspace(-1, 1, steps = x_length)
  y = t.linspace(-1, 1, steps = y_length)
  X, Y = t.meshgrid(x, y, indexing='ij')
  reference = t.stack((X, Y, X * Y), dim=0).unsqueeze(0).repeat(batch, 1, 1, 1).to(device)
  #, t.cos(2*math.pi*X) * t.cos(2*math.pi*Y)
  id_grid = t.stack((Y, X), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1).to(device)
  #2 * id_grid - flow_grid
  find_inv_model = add_bias_to_grid(id_grid).to(device)
  loss_fn = nn.MSELoss()
  optimizer = t.optim.SGD(find_inv_model.parameters(), lr = learning_rate)

  num_epochs = epochs
  loss_hist = []
  min_loss = 1e30
  early_stopping_count = 0
  for epoch in tqdm(range(num_epochs)):
    optimizer.zero_grad()
    output = find_inv_model()
    distort = t.nn.functional.grid_sample(reference, flow_grid, mode = mode)
    #inv_distort = t.nn.functional.grid_sample(reference, output, mode = mode)
    restored_left  = t.nn.functional.grid_sample(distort, output, mode = mode)
    #restored_right = t.nn.functional.grid_sample(inv_distort, flow_grid, mode = mode)
    left_loss = loss_fn(reference, restored_left)
    #right_loss = loss_fn(reference, restored_right)
    loss = left_loss #+ right_loss #+ (t.exp(t.abs(left_loss-right_loss)**2) - 1)
    #loss =  left_loss + right_loss
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
          loss_hist.append(loss.item())
          if loss_hist[-1]/min_loss >= 1: 
            early_stopping_count += 1
            # print(f'Early stopping count: {early_stopping_count}')
          if loss_hist[-1] < min_loss: 
            min_loss = loss_hist[-1]
            early_stopping_count = 0
    if early_stopping and early_stopping_count >=10: break

  with t.no_grad():
    flow_grid_inverse_neural = find_inv_model().detach().clone()
  
  del find_inv_model

  return flow_grid_inverse_neural, loss_hist, epoch

#%%
def compose_diffeo_from_left(diffeo_l: t.tensor, diffeo_r: t.tensor, mode = 'bilinear'):
  '''
  This l stands for left and r for right.
  The composition is defined as first apply r then l, so we will implement l interpolates r.
  shape should be the same: (n, x_res, y_res, 2) where n is interpreted as to be looped over
  '''
  if len(diffeo_l.shape) != 4 or len(diffeo_r.shape) != 4 or diffeo_l.shape[0] != diffeo_r.shape[0]:
    raise Exception(f'shape do not match, left:{diffeo_l.shape}, right:{diffeo_r.shape}')
  img = t.permute(diffeo_r, (0, 3, 1, 2))
  product = t.nn.functional.grid_sample(img, diffeo_l, mode = mode, padding_mode='border') # left multiplication
  product = t.permute(product, (0, 2, 3, 1))
  return product

#%%

class mix_channel_2d(nn.Module):
  '''copied from nn.Linear code
  for every pixel x,y, performs a linear transformation from channel to channel
  input shape bixy where b:batch, i:channel, x:x-coordinate, y:y-coordinate
  weight shape: icxy where i:channel, c:channel, x,y:...
  output is a matrix multiplication along the channel dimension: i, ic -> c
  number of parameter: c * c * x * y --> train on >=c images to be under-parametrized
  '''
  __constants__ = ['channels', 'x_resolution', 'y_resolution']
  channels: int
  x_res: int
  y_res: int
  weight: t.Tensor

  def __init__(self, channels: int, x_res: int, y_res: int, device=None, dtype=None) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super().__init__()
    self.x_res = x_res
    self.y_res = y_res
    self.weight = nn.Parameter(t.empty((channels, channels, x_res, y_res), **factory_kwargs))
    self.reset_parameters()
  def reset_parameters(self) -> None:
    '''see nn.Linear'''
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
  def forward(self, input: t.Tensor) -> t.Tensor:
    return t.einsum('bixy, icxy -> bcxy', input, self.weight) 

#%%
def jacobian_det(x_map, y_map):
  '''
    compute jacobian using finite difference that numpy has built in
    returns the same shape as x_map and y_map
    there could be a minus sign because of transpose
    x, y index is confusing
  '''


  x_dir = np.gradient(x_map, 1, 1)
  y_dir = np.gradient(y_map, 1, 1)

  jacobian = np.transpose(np.stack([y_dir, x_dir]), axis = (3, 2, 0, 1))

  return np.linalg.det(jacobian)
#%%
# from PIL import Image
# val_pic = Image.open('val_pic.png')
# val_pic_tensor = t.load('val_pic_inf.pt')

# A_nm, B_nm = sparse_transform_amplitude(5, 5, 2, amplitude = 1, num_of_diffeo = 1)
# A_nm = np.mean(A_nm, axis = 0)
# B_nm = np.mean(B_nm, axis = 0)


# grid_sample = create_grid_sample(224, 224, A_nm, B_nm)
# inverse_grid_sample, loss_hist = find_inv_grid(grid_sample)
# #%%
# distorted = t.nn.functional.grid_sample(val_pic_tensor.unsqueeze(0), grid_sample, mode = 'bilinear')
# undistorted = t.nn.functional.grid_sample(distorted, inverse_grid_sample, mode = 'bicubic')
# %%
