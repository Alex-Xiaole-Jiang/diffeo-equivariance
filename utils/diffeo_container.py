#%%
import numpy as np
import torch as t
import torch.nn as nn
import math

from distortion import sparse_transform_amplitude, create_grid_sample, compose_diffeo_from_left
#%%
class diffeo_container:
  def __init__(self, x_res: int, y_res: int, diffeos = None):
    self._x_res = x_res
    self._y_res = y_res
    if diffeos == None: self.diffeos = []
    if diffeos != None: self.diffeos = diffeos
    self.children = []

  @property
  def x_res(self): return self._x_res
  @property
  def y_res(self): return self._y_res

  def up_down_sample(self, new_x_res, new_y_res):
    x = t.linspace(-1, 1, new_x_res)
    y = t.linspace(-1, 1, new_y_res)
    X, Y = t.meshgrid(x, y)
    id_grid = t.cat([X.unsqueeze(2), Y.unsqueeze(2)], dim = 2).unsqueeze(0)
    new_diffeo = []
    for diffeos in list(self):
      new_diffeo.append(compose_diffeo_from_left(id_grid.repeat(len(diffeos), 1, 1, 1), diffeos))
    self.children.append(diffeo_container(new_x_res,new_y_res,diffeos = new_diffeo))
    return self.children[-1]

  def __getitem__(self, index):
    if isinstance(index, int): return self.diffeos[index]
    return self.diffeos[index[0]][index[1:]]
  
  def __len__(self):
    return len(self.diffeos)
  
  def __repr__(self):
    return f"{type(self).__name__}(x_res={self.x_res}, y_res={self.y_res})"

#%%
class sparse_diffeo_container(diffeo_container):
  def __init__(self, x_res: int, y_res: int, A = None, B = None, diffeos = None, rng = None, seed = 37):
    super().__init__(x_res, y_res, diffeos)
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


  def sparse_AB_append(self, x_cutoff, y_cutoff, num_of_terms, diffeo_amp, num_of_diffeo, rng = None, seed = 37, alpha = None):
    if rng == 'self': rng = self._rng
    
    self.diffeo_params.append({'x_cutoff': x_cutoff, 'y_cutoff': y_cutoff, 'num_of_diffeo':num_of_diffeo, 'diffeo_amp':diffeo_amp, 'num_of_terms': num_of_terms, 'rng':rng, 'seed':seed, 'alpha':alpha})
    A_nm, B_nm = sparse_transform_amplitude(**self.diffeo_params[-1])
    
    self.A.append(A_nm)
    self.B.append(B_nm)
  
  def get_all_grid(self):
    self.diffeos = []
    for A, B in zip(self.A, self.B):
      self.diffeos.append(create_grid_sample(self.x_res, self.y_res, A, B))

  def get_composition(self):
    self.children.append(diffeo_compose_container(self, level = 1))
    return self.children[-1]


#%%
class diffeo_compose_container(diffeo_container):
  def __init__(self, diffeo_container: diffeo_container, level = 0):
    super().__init__(diffeo_container.x_res, diffeo_container.y_res, [t.cat(list(diffeo_container))])
    self._num_of_generators = len(self.diffeos[0])
    self.compose(level = level)

  def compose(self, level = 1):
    for _ in range(level):
      num_of_diffeo_last_level = len(self.diffeos[-1])
      left = self.diffeos[0].unsqueeze(1).repeat(1, num_of_diffeo_last_level, 1, 1, 1).view(-1, self.x_res, self.y_res, 2)
      self.diffeos.append(compose_diffeo_from_left(left, self.diffeos[-1].repeat(self._num_of_generators, 1, 1, 1)))

  def __repr__(self):
    return f"{type(self).__name__}(x_res={self.x_res}, y_res={self.y_res}) with depth={len(self)}"
# %%
