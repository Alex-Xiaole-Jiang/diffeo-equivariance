#%%
import numpy as np
import torch as t


from .distortion import sparse_transform_amplitude, create_grid_sample, compose_diffeo_from_left, find_inv_grid
#%%
class diffeo_container:
  def __init__(self, x_res: int, y_res: int, diffeos = None, device = t.device('cpu')):
    self._x_res = x_res
    self._y_res = y_res
    if diffeos == None: self.diffeos = []
    if diffeos != None: self.diffeos = diffeos
    self.resampled = {} # dict for resampled grid which are container objects
    self.inverse = None # container for inverse grid
    self._find_inverse_loss = []
    self._device = device
    self.to(device)

  @property
  def x_res(self): return self._x_res
  @property
  def y_res(self): return self._y_res 
  @property
  def device(self): return self._device
  @property
  def length(self):
    length = 0
    for diffeo in self.diffeos:
      length += len(diffeo)
    return length

  def __getitem__(self, index):
    if isinstance(index, int): return self.diffeos[index]
    return self.diffeos[index[0]][index[1:]]
  
  def __len__(self):
    return len(self.diffeos)
  
  def __repr__(self):
    return f"{type(self).__name__}(x_res={self.x_res}, y_res={self.y_res}, with {self.length} diffeos)"
  
  def __call__(self, input):
    return [t.nn.functional.grid_sample(input, diffeos) for diffeos in self.diffeos]
  
  def to(self, device):
    self._device = device
    for index, diffeo in enumerate(self.diffeos): 
      self.diffeos[index] = self.diffeos[index].to(device)
    for children in self.resampled.values(): children.to(device)
    if type(self.inverse) == type(diffeo_container): self.inverse.to(device)
  
  def get_id_grid(self, x_res = None, y_res = None):
    if x_res == None: x_res = self.x_res
    if y_res == None: y_res = self.y_res
    x = t.linspace(-1, 1, x_res)
    y = t.linspace(-1, 1, y_res)
    X, Y = t.meshgrid(x, y)
    id_grid = t.cat([Y.unsqueeze(2), X.unsqueeze(2)], dim = 2).unsqueeze(0).to(self.device)
    return id_grid    

  def up_down_sample(self, new_x_res, new_y_res, mode = 'bilinear'):
    id_grid = self.get_id_grid(x_res = new_x_res, y_res = new_y_res).to(self.device)
    new_diffeo = []
    for diffeos in self.diffeos:
      new_diffeo.append(compose_diffeo_from_left(id_grid.repeat(len(diffeos), 1, 1, 1), diffeos, mode = mode))
    self.resampled[f'{new_x_res},{new_y_res}']= diffeo_container(new_x_res,new_y_res,diffeos = new_diffeo)
    return self.resampled[f'{new_x_res},{new_y_res}']
  
  def get_inverse_grid(self, base_learning_rate = 1000, learning_rate_scaling = 1, mode = 'bilinear'):
    inverse = []
    for diffeo in self.diffeos:
      lr = base_learning_rate * (1 + learning_rate_scaling * len(diffeo))
      inv_grid, lost_hist, epoch_num = find_inv_grid(diffeo, learning_rate = lr, mode = mode)
      inverse.append(inv_grid)
      self._find_inverse_loss.append({'loss': lost_hist, 'stopping_epoch': epoch_num, 'lr': lr})
    self.inverse = diffeo_container(self.x_res, self.y_res, diffeos=inverse, device = self.device)
    return self.inverse


#%%
class sparse_diffeo_container(diffeo_container):
  def __init__(self, x_res: int, y_res: int, A = None, B = None, diffeos = None, rng = None, seed = 37, device = t.device('cpu')):
    super().__init__(x_res, y_res, diffeos, device)
    if rng == None:
      self.rng = 'default with seed=37'
      self._rng = np.random.default_rng(seed = seed)
    else: 
      self.rng = 'passed in'
      self._rng = self.rng
    self.A = A
    self.B = B
    if A == None: self.A = []
    if B == None: self.B = []
    self.diffeo_params = []
    self.children = []

  def sparse_AB_append(self, x_cutoff, y_cutoff, num_of_terms, diffeo_amp, num_of_diffeo, rng = None, seed = 37, alpha = None):
    if rng == 'self': rng = self._rng
    
    self.diffeo_params.append({'x_cutoff': x_cutoff, 'y_cutoff': y_cutoff, 'num_of_diffeo':num_of_diffeo, 'diffeo_amp':diffeo_amp, 'num_of_terms': num_of_terms, 'rng':rng, 'seed':seed, 'alpha':alpha})
    A_nm, B_nm = sparse_transform_amplitude(**self.diffeo_params[-1])
    
    self.A.append(A_nm)
    self.B.append(B_nm)
  
  def get_all_grid(self):
    for A, B in zip(self.A, self.B):
      self.diffeos.append(create_grid_sample(self.x_res, self.y_res, A, B))
    
  def clear_all_grid(self):
    self.diffeos = []

  def get_composition(self, level = 1):
    new_container = diffeo_compose_container(self, level = level)
    if new_container in self.children: self.children.remove(new_container)
    self.children.append(new_container)
    return self.children[-1]


#%%
class diffeo_compose_container(diffeo_container):
  def __init__(self, diffeo_container: diffeo_container, level = 0):
    super().__init__(diffeo_container.x_res, diffeo_container.y_res, [t.cat(list(diffeo_container))])
    self._num_of_generators = len(self.diffeos[0])
    self.diffeos.insert(0, self.get_id_grid())
    
    self.element_to_index = {'g0': (0,0)}
    self._generator_list = [f'g{i}' for i in range(1, self._num_of_generators + 1)]
    for index, value in enumerate(self._generator_list):
      self.element_to_index[value] = (1, index)
    self.compose(level = level)

  def compose(self, level = 1):
    if len(self.diffeos) >= 3: self.diffeos = self.diffeos[0:2]
    self.element_to_index = self.element_to_index
    for counter, key in enumerate(self.element_to_index):
      if counter > self._num_of_generators: del self.element_to_index[key]

    for _ in range(level):
      num_of_diffeo_last_level = len(self.diffeos[-1])
      left = self.diffeos[1].unsqueeze(1).repeat(1, num_of_diffeo_last_level, 1, 1, 1).view(-1, self.x_res, self.y_res, 2)
      self.diffeos.append(compose_diffeo_from_left(left, self.diffeos[-1].repeat(self._num_of_generators, 1, 1, 1)))

      left_gen_list = [item for item in self._generator_list for _ in range(num_of_diffeo_last_level)]
      right_gen_list =list(self.element_to_index)[-num_of_diffeo_last_level:] * self._num_of_generators
      all_elem_list = [a + b for a, b in zip(left_gen_list, right_gen_list)]
      for index, value in enumerate(all_elem_list):
        self.element_to_index[value] = (len(self.diffeos)-1, index)


  def __repr__(self):
    return f"{type(self).__name__}(x_res={self.x_res}, y_res={self.y_res}) with strings of generators of length {len(self.diffeos) - 1} and {self.length} diffeos in total"
# %%
