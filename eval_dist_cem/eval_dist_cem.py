import numpy as np
import matplotlib.pyplot as plt
from eval_dist_cem.DistModel.distModel import DistModel
from eval_dist_cem.CEM.cem import CEM
#from utils import *

model_dir = './eval_dist_cem/DistModel/mass_point/64/'
#model_dir = './eval_dist_cem/DistModel/new_MP/64/'
distModel = DistModel('RAE_16', 5, 2, n_units=64)
distModel.load_model(model_dir)

plt_sz = 1
plt_range = [-plt_sz, plt_sz,-plt_sz, plt_sz]

cem = CEM(distModel.pred, 2, v_min=[-1, 0], v_max=[1, 1], maxits1=100)

def map_shuffle(i):
    shuffle_idx = [1, 3, 0, 4, 2]
    return shuffle_idx[i]

def eval_dist_cem(i):
    instr = np.zeros([5])
    #instr[map_shuffle(i)] = 1
    instr[i] = 1
    pred_coord = cem.eval(instr)
    #print('test: ', i)
    #print('eval: ', instr)
    return pred_coord

if __name__ == '__main__':
    for i in range(5):
        print(eval_dist_cem(i))
