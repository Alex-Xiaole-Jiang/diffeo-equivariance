import torch as t
import torch.nn as nn 
import torch.nn.functional as F

from .distortion import compose_diffeo_from_left, create_grid_sample, get_id_grid
from tqdm import tqdm

torch_seed = 37
t.manual_seed(torch_seed)


class add_bias_to_AB(nn.Module):
  def __init__(self, AB, extra_freq_scaling = 1):
    # grid should have the shape of a grid_sample grid, i.e. (Channel, X, Y, 2)
    super().__init__()
    _, _, x_cutoff, y_cutoff = AB.shape
    self.AB = F.pad(AB,(0,(extra_freq_scaling) * y_cutoff, 0, (extra_freq_scaling) * x_cutoff),mode = 'constant', value = 0)
    # self.bias = nn.Parameter(self.AB.abs().sum() * (t.randint_like(self.AB, 1) - 1/2)/self.AB.numel())
    # self.bias = nn.Parameter(t.rand_like(self.AB))
    self.bias = nn.Parameter(t.rand_like(self.AB)/self.AB.numel())
    self.result = AB
  def forward(self):
    self.result = self.bias + self.AB
    return self.result
  @property
  def result_magnitude(self):
    return self.result[0].detach().abs().sum(), self.result[1].detach().abs().sum()
  @property
  def bias_magnitude(self):
    return self.bias[0].detach().abs().sum(), self.bias[1].detach().abs().sum()
  @property
  def original_magnitude(self):
    return self.AB[0].detach().abs().sum(), self.AB[1].detach().abs().sum()
  
# AB_original = t.cat([t.stack(sparse_diffeos.A), t.stack(sparse_diffeos.B)])
def find_param_inverse(AB: t.Tensor, extra_freq_scaling = 5, num_epochs = 500, device = t.device('cpu')) -> t.Tensor:
  '''
  input: parameters of the diffeo as a t.Tensor([A,B])
  output: t.Tensor([A_inverse, B_inverse]) that gives the inverse of the diffeo
  the algo uses Adagrad
  '''
  inv_loss_hist = []
  AB_mag = []

  grid = create_grid_sample(224, 224, AB[0], AB[1]).to(device)
  AB = add_bias_to_AB(-AB.to(device), extra_freq_scaling = extra_freq_scaling).to(device)
  loss_fn = nn.MSELoss()

  optimizer = t.optim.Adagrad(AB.parameters(), lr = 0.1)
  
  id_grid = get_id_grid(224, 224, device)
  
  for epoch in tqdm(range(num_epochs)):
      optimizer.zero_grad()
      new_AB = AB()
      with t.device(device):
          inv_grid = create_grid_sample(224, 224, new_AB[0], new_AB[1]).to(device)
      un_distorted = compose_diffeo_from_left(inv_grid, grid)
      unreg_loss = loss_fn(un_distorted, id_grid)
      loss = unreg_loss * (1 + 0.1 * AB.result.abs().sum()/AB.AB.abs().sum())
      loss.backward()
      optimizer.step()
      # scheduler.step(loss)

      if (epoch) % 50 == 0:
            with t.no_grad(): 
              inv_loss_hist.append(unreg_loss.item())
              AB_mag.append(AB.result.abs().sum().item()/2)
              AB.bias[t.abs(AB.bias) < 1e-7] = 0
            # print(loss.item())
  
  return AB.result.detach() , inv_loss_hist, AB_mag



# AB_original = t.cat([t.stack(sparse_diffeos.A), t.stack(sparse_diffeos.B)])
def find_img_inverse(distorted_img: t.Tensor, target_img:t.Tensor, AB: t.Tensor, extra_freq_scaling = 5, num_epochs = 500, device = t.device('cpu'), grid_sample_kwargs = {}) -> t.Tensor:
  '''
  input:
    distorted_img: distorted image
    target_img: image we want the distorted image to look like after applying diffeo
    parameters of the diffeo as a t.Tensor([A,B])
  output: 
    t.Tensor([A_inverse, B_inverse]) that gives the inverse of the diffeo
  the optimizer is Adagrad
  '''
  inv_loss_hist = []
  AB_mag = []

  batch_size, channel_num, x_res, y_res = distorted_img.shape

  AB = add_bias_to_AB(-AB.to(device), extra_freq_scaling = extra_freq_scaling).to(device)
  loss_fn = nn.MSELoss()

  optimizer = t.optim.Adagrad(AB.parameters(), lr = 0.1)
  
  
  for epoch in tqdm(range(num_epochs)):
      optimizer.zero_grad()
      new_AB = AB()
      with t.device(device):
          inv_grid = create_grid_sample(x_res, y_res, new_AB[0], new_AB[1]).to(device)
      un_distorted = F.grid_sample(distorted_img, inv_grid.expand(batch_size, -1, -1, -1), 
                                   align_corners=True, **grid_sample_kwargs)
      unreg_loss = loss_fn(un_distorted, target_img)
      loss = unreg_loss * (1 + 0.1 * AB.result.abs().sum()/AB.AB.abs().sum())
      loss.backward()
      optimizer.step()
      # scheduler.step(loss)

      if (epoch) % 50 == 0:
            with t.no_grad(): 
              inv_loss_hist.append(unreg_loss.item())
              AB_mag.append(AB.result.abs().sum().item()/2)
              AB.bias[t.abs(AB.bias) < 1e-7] = 0
            # print(loss.item())
  
  return AB.result.detach() , inv_loss_hist, AB_mag