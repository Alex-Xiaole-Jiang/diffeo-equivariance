# %%
import numpy as np
import torch as t
import torch.nn as nn
import math
#from torch_perlin_noise import rand_perlin_2d_octaves
# import matplotlib.pyplot as plt

def get_version(): print('version neural_test')

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



def sparse_transform_amplitude(x_length, y_length, num_of_terms, amplitude = 1, rng = None, seed = 37, loop = 1, alpha = None):
  '''
    returns x_length by y_length matrix where num_of_terms elements is random and has flattened norm of 1
    in the case that num_of_terms << x_length * y_length, this matrix is sparse
    x_length and y_length specifies the frequency bound
    amplitude changes the norm
  '''
  if rng == None: rng = np.random.default_rng(seed = seed)

  A = []
  B = []

  total_num_elem = x_length * y_length
  pad_len = total_num_elem - num_of_terms #- 1

  for i in range(loop):

    A_sign = (rng.integers(2, size = num_of_terms) - 0.5) * 2
    B_sign = (rng.integers(2, size = num_of_terms) - 0.5) * 2

    if alpha == None:
      A_nm = amplitude * rng.dirichlet(np.ones(num_of_terms)) * A_sign
      B_nm = amplitude * rng.dirichlet(np.ones(num_of_terms)) * B_sign
    elif alpha != None:
      A_nm = amplitude * rng.dirichlet(alpha * np.ones(num_of_terms)) * A_sign
      B_nm = amplitude * rng.dirichlet(alpha * np.ones(num_of_terms)) * B_sign

    A_nm = np.pad(A_nm, (0, pad_len), 'constant')
    A_nm = rng.permutation(A_nm).reshape(x_length, y_length)
    # A_nm = np.pad(A_nm, (1,0), 'constant').reshape(x_length, y_length)
    A.append(A_nm)

    B_nm = np.pad(B_nm, (0, pad_len), 'constant')
    B_nm = rng.permutation(B_nm).reshape(x_length, y_length)
    # B_nm = np.pad(B_nm, (1,0), 'constant').reshape(x_length, y_length)
    B.append(B_nm)

  return np.array(A), np.array(B)


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




def create_grid_sample(x_length: int, y_length: int, A_list: np.array, B_list: np.array) -> t.Tensor:
  '''
  cos distortion for t.nn.functional.grid_sample, using cos b/c the grid is from -1 to 1

  Args:
  - x_length (int): Length of x-axis of image.
  - y_length (int): Length of y-axis of image.
  - A_nm (np.array): Square matrix of coefficents. Sets size of cut off
    Following np.meshgrid, the second index is the x frequency coefficient

  Returns:
  - x_map (np.array): Size `x_length` * `y_length`.
  - y_map (np.array): Size `x_length` * `y_length`.
  '''
  flow_grid = []

  for A_nm, B_nm in zip(A_list,B_list):

    # find non-zero coefficient and convert to frequency
    non_zero_A_arg = np.nonzero(A_nm)
    freq_A_arg = (np.array(non_zero_A_arg) + 1) * np.pi / 2  # n pi / L
    max_A_freq = np.max(np.transpose(freq_A_arg), axis = 1)
    non_zero_B_arg = np.nonzero(B_nm)
    freq_B_arg = (np.array(non_zero_B_arg) + 1) * np.pi / 2
    max_B_freq = np.max(np.transpose(freq_B_arg), axis = 1)

    # save time by only computing non-duplicate index
    unique_A_x, inv_index_A_x = np.unique(freq_A_arg[1], return_inverse = True)
    unique_A_y, inv_index_A_y = np.unique(freq_A_arg[0], return_inverse = True)
    unique_B_x, inv_index_B_x = np.unique(freq_B_arg[1], return_inverse = True)
    unique_B_y, inv_index_B_y = np.unique(freq_B_arg[0], return_inverse = True)

    # Create Coordinates
    x = np.linspace(-1, 1, num = x_length, endpoint = True)
    y = np.linspace(-1, 1, num = y_length, endpoint = True)
    X, Y = np.meshgrid(x, y, indexing='ij') # first index is x and second y

    # Normalization
    # max_length = max((x_length - 1), (y_length - 1))
    normalization_A = 1 / (max_A_freq)
    normalization_B = 1 / (max_B_freq)
    #np.nan_to_num(normalization_A, copy = False, nan = 1)
    #np.nan_to_num(normalization_B, copy = False, nan = 1)

    # basis to do outer product, truncate smaller than 1e-15
    x_basis_A = np.sin(unique_A_x[None, None, :] * (X[:, :, None] + 1))
    y_basis_A = np.sin(unique_A_y[None, None, :] * (Y[:, :, None] + 1))
    x_basis_B = np.sin(unique_B_x[None, None, :] * (X[:, :, None] + 1))
    y_basis_B = np.sin(unique_B_y[None, None, :] * (Y[:, :, None] + 1))

    eps = np.finfo(np.float64).eps * 10
    x_basis_A[np.abs(x_basis_A) < eps] = 0
    y_basis_A[np.abs(y_basis_A) < eps] = 0
    x_basis_B[np.abs(x_basis_B) < eps] = 0
    y_basis_B[np.abs(y_basis_B) < eps] = 0

    # fully vectorized fourier sum
    X_pert = np.einsum('xyi, i, xyi, i -> xy', y_basis_A[:,:,inv_index_A_y], A_nm[non_zero_A_arg], x_basis_A[:,:,inv_index_A_x], normalization_A)
    Y_pert = np.einsum('xyi, i, xyi, i -> xy', y_basis_B[:,:,inv_index_B_y], B_nm[non_zero_B_arg], x_basis_B[:,:,inv_index_B_x], normalization_B)


    x_map = (X + X_pert)
    y_map = (Y + Y_pert)

    flow_grid_np = np.stack((np.float32(y_map), np.float32(x_map)), axis = -1)
    flow_grid.append(t.from_numpy(flow_grid_np).unsqueeze(0))

  flow_grid = t.cat(flow_grid, dim = 0)
  
  return flow_grid

class get_diffeomorphism:
  def __init__(self, x_res: int, y_res: int, A = None, B = None, rng = None, seed = 37):
    self.x_res = x_res
    self.y_res = y_res
    if rng == None:
      self.rng = np.random.default_rng(seed = seed)
    else: 
      self.rng = rng 
    self.A = A
    self.B = B
    if A == None: self.A = []
    if B == None: self.B = []
    self.diffeos = []
    self.sparse_AB_param = []

  def sparse_AB_append(self, x_cutoff, y_cutoff, num_of_terms, diffeo_amp, num_of_diffeo, rng = None, seed = 37, alpha = None):
    if rng == 'self': rng = self.rng
    A_nm, B_nm = sparse_transform_amplitude(x_cutoff, y_cutoff, num_of_terms, diffeo_amp, loop = num_of_diffeo, rng = rng, seed = seed, alpha = alpha)
    
    self.sparse_AB_param.append({'x_cutoff': x_cutoff, 'y_cutoff': y_cutoff, 'num_of_diffeo':num_of_diffeo, 'diffeo_amp':diffeo_amp, 'sparsity': num_of_terms, 'rng':rng, 'seed':seed, 'alpha':alpha})
    self.A.append(A_nm)
    self.B.append(B_nm)
  
  def get_all_grid(self):
    self.diffeos = []
    for A, B in zip(self.A, self.B):
      self.diffeos.append(create_grid_sample(self.x_res, self.y_res, A, B))
  
  def __getitem__(self, index):
    return self.diffeos[index[0]][index[1:]]


class BiasOnly_gridInv(nn.Module):
  def __init__(self, grid):
    super().__init__()
    self.grid = grid
    self.bias = nn.Parameter(t.zeros_like(grid))
  def forward(self):
    return self.grid + self.bias
    
def find_inv_grid(flow_grid, mode ='bilinear', learning_rate = 0.001, epochs = 10000, early_stopping = True):
  x_length, y_length, _ = flow_grid.squeeze().shape
  x = t.linspace(-1, 1, steps = x_length)
  y = t.linspace(-1, 1, steps = y_length)
  X, Y = t.meshgrid(x, y, indexing='ij')
  reference = t.stack((X, Y, X * Y, t.cos(2*math.pi*X) * t.cos(2*math.pi*Y)), dim=0).unsqueeze(0)
    
  find_inv_model = BiasOnly_gridInv(t.stack((Y, X), dim=-1).unsqueeze(0))
  loss_fn = nn.MSELoss()
  optimizer = t.optim.Adam(find_inv_model.parameters(), lr = learning_rate)

  num_epochs = epochs
  loss_hist = []
  min_loss = 1e30
  early_stopping_count = 0
  for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = find_inv_model()
    #distort = t.nn.functional.grid_sample(reference, flow_grid, mode = mode)
    inv_distort = t.nn.functional.grid_sample(reference, output, mode = mode)
    #restored_left  = t.nn.functional.grid_sample(distort, output, mode = mode)
    restored_right = t.nn.functional.grid_sample(inv_distort, flow_grid, mode = mode)
    #left_loss = loss_fn(reference, restored_left)
    right_loss = loss_fn(reference, restored_right)
    loss = right_loss #+ left_loss #+ (t.exp(t.abs(left_loss-right_loss)**2) - 1)
    #loss =  left_loss + right_loss
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 500 == 0:
          loss_hist.append(loss.item())
          if loss_hist[-1]/min_loss >= 0.95: early_stopping_count += 1
          if loss_hist[-1] < min_loss: min_loss = loss_hist[-1]
    if early_stopping and early_stopping_count >=5: break

  with t.no_grad():
    flow_grid_inverse_neural = find_inv_model().detach().clone()

  return flow_grid_inverse_neural, loss_hist


#%%
# from PIL import Image
# val_pic = Image.open('val_pic.png')
# val_pic_tensor = t.load('val_pic_inf.pt')

# A_nm, B_nm = sparse_transform_amplitude(5, 5, 2, amplitude = 1, loop = 1)
# A_nm = np.mean(A_nm, axis = 0)
# B_nm = np.mean(B_nm, axis = 0)


# grid_sample = create_grid_sample(224, 224, A_nm, B_nm)
# inverse_grid_sample, loss_hist = find_inv_grid(grid_sample)
# #%%
# distorted = t.nn.functional.grid_sample(val_pic_tensor.unsqueeze(0), grid_sample, mode = 'bilinear')
# undistorted = t.nn.functional.grid_sample(distorted, inverse_grid_sample, mode = 'bicubic')
# %%
