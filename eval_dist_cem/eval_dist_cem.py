import numpy as np
import matplotlib.pyplot as plt
from eval_dist_cem.DistModel.distModel import DistModel
from eval_dist_cem.CEM.cem import CEM
#from utils import *

#model_dir = './eval_dist_cem/DistModel/instr_3d_mp/64/'
model_dir = './eval_dist_cem/DistModel/mass-point/64/'
distModel = DistModel('RAE_16', 5, 2, n_units=64)
distModel.load_model(model_dir)

plt_sz = 1
plt_range = [-plt_sz, plt_sz,-plt_sz, plt_sz]

cem = CEM(distModel.pred, 2, v_min=[-1, 0], v_max=[1, 1], maxits1=100)

def eval_dist_cem(i):
    instr = np.zeros([5])
    instr[i] = 1
    pred_coord = cem.eval(instr)
    return pred_coord

if __name__ == '__main__':
    for i in range(5):
        print(eval_dist_cem(i))
