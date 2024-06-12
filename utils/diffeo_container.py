import numpy as np
import torch as t
import torch.nn as nn
import math

from distortion import sparse_transform_amplitude, create_grid_sample, diffeo_composition

#%%
class diffeo_container:
  def __init__(self, x_res: int, y_res: int, diffeos = None):
    self._x_res = x_res
    self._y_res = y_res
    if diffeos == None: self.diffeos = []
    if diffeos != None: self.diffeos = diffeos

  @property
  def x_res(self): return self._x_res
  @property
  def y_res(self): return self._y_res

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

#%%
class diffeo_compose_container(diffeo_container):
  def __init__(self, diffeo_container: diffeo_container):
    super().__init__(diffeo_container.x_res, diffeo_container.y_res, t.cat(list(diffeo_container)))
    self.generators = self.diffeos
    self.diffeos = []

