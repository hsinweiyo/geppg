import numpy as np

from gep_utils import *


class CheetahRepresenter():

    def __init__(self):

        self._description = ['mean_vx', 'min_z']
        # define goal space
        self._initial_space = np.array([[-4, 7],[-3,2]])
        self._representation = None

    def represent(self, obs_seq, act_seq=None):
        obs = np.copy(obs_seq)
        mean_vx = np.array([obs[0, 8, :].mean()])
        min_z = np.array([obs[0, 0, :].min()])
        self._representation = np.concatenate([mean_vx, min_z], axis=0)

        # scale representation to [-1,1]^N
        self._representation = scale_vec(self._representation, self._initial_space)
        self._representation.reshape(1, -1)

        return self._representation

    @property
    def initial_space(self):
        return self._initial_space

    @property
    def dim(self):
        return self._initial_space.shape[0]

class CMCRepresenter():

    def __init__(self):
        self._description = ['max position', 'range position', 'spent energy']
        # define goal space
        self._initial_space= np.array([[-0.6, 0.6], [0., 1.8], [0, 100]])  # space in which goal are sampled
        self._representation = None

    def represent(self, obs_seq, act_seq):
        spent_energy = np.array([np.sum(act_seq[0, 0, np.argwhere(~np.isnan(act_seq))] ** 2 * 0.1)])
        diff = np.array([np.nanmax(obs_seq[0, 0, :]) - np.nanmin(obs_seq[0, 0, :])])
        max = np.array([np.nanmax(obs_seq[0, 0, :])])
        self._representation = np.concatenate([max, diff, spent_energy], axis=0)

        # scale representation to [-1,1]^N
        self._representation = scale_vec(self._representation, self._initial_space)
        #print('Representation')
        #print (self._representation)
        self._representation.reshape(1, -1)

        return self._representation

    @property
    def initial_space(self):
        return self._initial_space

    @property
    def dim(self):
        return self._initial_space.shape[0]

class KobukiRepresenter():
  
    def __init__(self):
        # self._description = ['red', 'green', 'blue']
        # define goal space
        # self._initial_space = np,array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) # space in which goal are sampled
        self._description = ['engineer_goal_r', 'engineer_goal_g', 'engineer_goal_b']
        # define goal space
        self._initial_space = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]) # space in which goal are sampled
        self._representation = None

    def represent(self, obs_seq, act_seq):
        #print('OBS SEQ')
        #print(obs_seq[0,:,-1])
        #print(obs_seq)
        #axis_x = np.nanobs_seq[]
         # self._representation = np.array(obs_seq[0, :, -1])
        idxs = [5, 9, 13, 17, 21]
        #self._representation = obs_seq[~np.isnan(np.array(obs_seq))].reshape((2, -1))[:, -1]
        print('OBS_seq')
        print(obs_seq)
        print(obs_seq.shape)
        obs_seq = obs_seq[~np.isnan(np.array(obs_seq))].reshape((22, -1))[:, -1]
        print('OBS_seq')
        print(obs_seq)
        print(obs_seq.shape)
        #objects = [ obs_seq[0,:,-1][x] for x in idxs ]
        objects = [ obs_seq[x] for x in idxs ]
        target = min(enumerate(objects), key=lambda x: x[1])[0]
        print('Target')
        print(target)
        self._representation = np.array([ obs_seq[x] for x in range(target*4+2, target*4+5) ])
        print('Representation')
        print(self._representation )
        self._representation = scale_vec(self._representation, self._initial_space)
        print('After scale Representation')
        print(self._representation )
        
        self._representation.reshape(1, -1)

        return self._representation, target
        
    @property
    def initial_space(self):
        return self._initial_space

    @property
    def dim(self):
        return self._initial_space.shape[0]
