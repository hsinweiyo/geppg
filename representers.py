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
        print(self._representation)

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
        self._description = ['engineer_goal_midx', 'engineer_goal_midy', 'engineer_goal_x', 'engineer_goal_y']
        self._initial_space = np.array([[-0.2, 0.2]]*4) # space in which goal are sampled
        self._representation = None

    def represent(self, obs_seq, act_seq):
        obs_seq_mid = obs_seq[~np.isnan(np.array(obs_seq))].reshape((6, -1))[:, obs_seq.shape[0]//2]
        obs_seq = obs_seq[~np.isnan(np.array(obs_seq))].reshape((6, -1))[:, -1]
        #print(obs_seq)

        midx, midy = obs_seq_mid[0]+obs_seq_mid[1], obs_seq_mid[2]+obs_seq_mid[3]
        x, y = obs_seq[0]+obs_seq[1], obs_seq[2]+obs_seq[3]
        self._representation = np.array([midx, midy, x, y])
        #print('Representation')
        #print(self._representation )
        
        # scale representation to [-1,1]^N
        
        self._representation = scale_vec(self._representation, self._initial_space)
        #print('After scale Representation')
        #print(self._representation)
        
        self._representation.reshape(1, -1)
        return self._representation

    @property
    def initial_space(self):
        return self._initial_space

    @property
    def dim(self):
        return self._initial_space.shape[0]
