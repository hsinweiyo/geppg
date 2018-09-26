import numpy as np
import matplotlib.pyplot as plt
from eval_traj.DistModel.distModel import DistModel
from eval_traj.CEM.cem import CEM
#from utils import *

model_dir = './eval_traj/DistModel/MPT/128/'
distModel = DistModel('RAE_16', 7, 4, n_units=128)
distModel.load_model(model_dir)

plt_sz = 1
plt_range = [-plt_sz, plt_sz,-plt_sz, plt_sz]

cem = CEM(distModel.pred, 4, v_min=[-0.5, -0.5, -1, -1], v_max=[0.5, 0.5, 1., 1.], maxits=1000, sampleMethod='Uniform')

def eval_traj(i,j):
    instr = np.zeros([7])
    instr[i] = 1
    instr[j] = 1
    pred_coord = cem.eval(instr)
    return pred_coord

if __name__ == '__main__':
    for i in range(5):
        for j in range(2):
            print(eval_dist_cem(i,j))
