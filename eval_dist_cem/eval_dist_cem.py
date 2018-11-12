import numpy as np
import matplotlib.pyplot as plt
from eval_dist_cem.DistModel.distModel import DistModel
from eval_dist_cem.DistModel.distModel_sgd import DistModel_SGD
from eval_dist_cem.CEM.cem_traj import CEM_traj
from eval_dist_cem.SGD.sgd import SGD
#from DistModel.distModel import DistModel
#from DistModel.distModel_sgd import DistModel_SGD
#from CEM.cem_traj import CEM_traj
#from SGD.sgd import SGD
#from utils import *

class Dist_CEM():
    def __init__(self, task,  env_id, dist_model=None):
      eval_method = 'cem'
      self._task = task
      plt_sz = 1
      self._plt_range = [-plt_sz, plt_sz,-plt_sz, plt_sz]
      if eval_method == 'cem':
        if env_id == 'Mass-point':
          if task == 'goal':
              print ('In mass-point goal')
              if dist_model != None:
                  model_dir = dist_model
              else:
                  print ('model is none')
                  model_dir = './eval_dist_cem/DistModel/mass_point/64/'
              #model_dir = './eval_dist_cem/DistModel/mass_point/64/'
              # model_dir = './eval_dist_cem/DistModel/mass_goal_dist1/'
              # self._distModel = DistModel('RAE_16', 5, 2, n_units=64)
              self._distModel = DistModel('RAE_16', 5, 2, n_units=128)
              self._distModel.load_model(model_dir)
              self._cem = CEM_traj(self._distModel.pred, 2, v_min=[-1, 0], v_max=[1, 1], maxits=100, sampleMethod='Uniform')
          else:
              #model_dir = './eval_traj/DistModel/MPT/128/'
              if dist_model != None:
                  print ('Path of dist func. ', dist_model)
                  #model_dir = "./eval_dist_cem/DistModel/NMPT/noise_0/1/"
                  model_dir = dist_model
              else:
                  print ('model is none')
                  model_dir = './eval_dist_cem/DistModel/MPT/128/'
                 #  model_dir = './eval_dist_cem/DistModel/NMPT/noise_0/1/'
              # model_dir = './eval_dist_cem/DistModel/MPT/128/'
              self._distModel = DistModel('RAE_16', 7, 4, n_units=128)
              self._distModel.load_model(model_dir)
              self._cem = CEM_traj(self._distModel.pred, 4, v_min=[-0.5, -0.5, -1, -1], v_max=[0.5, 0.5, 1., 1.], maxits=1000, sampleMethod='Uniform')
        else:
          if task == 'goal':
              if dist_model != None:
                  model_dir = dist_model
              else:
                  print ('model is none')
                  model_dir = './eval_dist_cem/DistModel/reacher/128/'
              
              # model_dir = './eval_dist_cem/DistModel/reacher_goal_dist3/'
              self._distModel = DistModel('RAE_16', 5, 2, n_units=128)
              self._distModel.load_model(model_dir)
              self._cem = CEM_traj(self._distModel.pred, 2, v_min=[-1., -1.], v_max=[1., 1.], N=1000, maxits=50, sampleMethod='Uniform')
          else:
              print ('Reacher-trajectory')
              if dist_model != None:
                  print ('Path of dist func. ', dist_model)
                  #model_dir = "./eval_dist_cem/DistModel/NRT/noise_0/1/"
                  model_dir = dist_model
              else:
                  print ('model is none')
                  model_dir = './eval_dist_cem/DistModel/reacher_new/128/'
              # model_dir = './eval_dist_cem/DistModel/reacher_new/128/'
              self._distModel = DistModel('RAE_16', 7, 4, n_units=128)
              self._distModel.load_model(model_dir)
              self._cem = CEM_traj(self._distModel.pred, 4, v_min=[0, -1, -1, -1], v_max=[1, 1, 1, 1], maxits=1000, sampleMethod='Uniform')
      else:
        if env_id == 'Mass-point':
          if task == 'goal':
              model_dir = './eval_dist_cem/DistModel/mass_point/64/'
              self._distModel = DistModel_SGD('RAE_16', 5, 2, n_units=64)
              self._sgd = SGD(model_dir, self._distModel, 5, 2)
          else:
              #model_dir = './eval_traj/DistModel/MPT/128/'
              model_dir = './eval_dist_cem/DistModel/MPT/128/'
              self._distModel = DistModel_SGD('RAE_16', 7, 4, n_units=128)
              self._sgd = SGD(model_dir, self._distModel, 7, 4)
        else:
          if task == 'goal':
              model_dir = './eval_dist_cem/DistModel/reacher/128/'
              self._distModel = DistModel_SGD('RAE_16', 5, 2, n_units=128)
              self._sgd = SGD(model_dir, self._distModel, 5, 2)
          else:
              print ('Reacher-trajectory')
              model_dir = './eval_dist_cem/DistModel/reacher_new/128/'
              self._distModel = DistModel_SGD('RAE_16', 7, 4, n_units=128)
              self._sgd = SGD(model_dir, self._distModel, 7, 4)
            
    def map_shuffle(self, i):
        if self._task == 'goal':
            shuffle_idx = [1, 3, 0, 4, 2]
            return shuffle_idx[i]
        else:
            shuffle_matrix = np.array([[0, 0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1],
                                    [0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 0],
                                    [1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0]])
            return shuffle_matrix.dot(i)

    def eval_dist_cem(self, i):
        instr = np.zeros([5])
        # instr[self.map_shuffle(i)] = 1
        instr[i] = 1
        pred_coord = self._cem.eval(instr)
        # print('test: ', i)
        # print('eval: ', instr)
        return pred_coord

    def eval_dist_sgd(self, i):
        instr = np.zeros([5])
        instr[i] = 1
        pred_coord = self._sgd.eval(instr)[0]
        return pred_coord

    def eval_traj(self, i,j):
        instr = np.zeros([7])
        instr[i] = 1
        instr[j] = 1
        # pred_coord = self._cem.eval(instr)
        pred_coord = self._cem.eval(self.map_shuffle(instr))
        return pred_coord

if __name__ == '__main__':
    dist = Dist_CEM('goal','Mass-point')
    for i in range(5):
        print(dist.eval_dist_cem(i))
