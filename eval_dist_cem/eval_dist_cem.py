import numpy as np
import matplotlib.pyplot as plt
from eval_dist_cem.DistModel.distModel import DistModel
from eval_dist_cem.CEM.cem import CEM
from eval_dist_cem.CEM.cem_traj import CEM_traj
#from utils import *

class Dist_CEM():
    def __init__(self, task, model_dir):
      self._task = task
      plt_sz = 1
      self._plt_range = [-plt_sz, plt_sz,-plt_sz, plt_sz]
      if task == 'goal':
          self._distModel = DistModel('RAE_16', 5, 2, n_units=64)
          self._distModel.load_model(model_dir)
          self._cem = CEM(self._distModel.pred, 2, v_min=[-1, 0], v_max=[1, 1], maxits1=100)
      else:
          self._distModel = DistModel('RAE_16', 7, 4, n_units=128)
          self._distModel.load_model(model_dir)
          self._cem = CEM_traj(self._distModel.pred, 4, v_min=[-0.5, -0.5, -1, -1], v_max=[0.5, 0.5, 1., 1.], maxits=100, sampleMethod='Uniform')
            
    def map_shuffle(self, i):
        shuffle_idx = [1, 3, 0, 4, 2]
        return shuffle_idx[i]

    def eval_dist_cem(self, i):
        instr = np.zeros([5])
        #instr[map_shuffle(i)] = 1
        instr[i] = 1
        pred_coord = self._cem.eval(instr)
        #print('test: ', i)
        #print('eval: ', instr)
        return pred_coord

    def eval_traj(self, i,j):
        instr = np.zeros([7])
        instr[i] = 1
        instr[j] = 1
        pred_coord = self._cem.eval(instr)
        return pred_coord

if __name__ == '__main__':
    for i in range(5):
        print(eval_dist_cem(i))
