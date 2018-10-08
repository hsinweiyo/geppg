import numpy as np
from gep_utils import *

class ReacherRepresenter():
    def __init__(self, nb_pt):
        self._description = ['engineer_goal_x', 'engineer_goal_y']
        self._initial_space = np.array([[-1., 1.]] * nb_pt)
        self._representation = None

    def represent(self, obs_seq, act_seq, task, nb_pt):
        nb_pair = nb_pt//2
        update_flag = True
        time_steps = max(1, int(obs_seq[~np.isnan(np.array(obs_seq))].reshape((4, -1)).shape[1]-1))
        obs_seq_mid = np.array(obs_seq[:,:time_steps])
        if task == 'traj':
            for i in range(1, time_steps):
                obs_seq_mid[:,i] = obs_seq[~np.isnan(np.array(obs_seq))].reshape((4, -1))[:, i]
        obs_seq = obs_seq[~np.isnan(np.array(obs_seq))].reshape((4, -1))[:, -1]
        obs_seq_mid = obs_seq_mid.reshape(4, -1)
        if task == 'traj': 
            testing = obs_seq_mid.T
            if (testing == testing[0]).all():
                update_flag = False

            self._representation = np.hstack([obs_seq_mid.T[:,:2], np.array([obs_seq[:2]]*time_steps)])
            for i in range(time_steps):
                self._representation[i,:] = scale_vec(self._representation[i,:], self._initial_space)
        else:
            self._representation = np.array(obs_seq[:2])
            self._representation = scale_vec(self._representation, self._initial_space)
            self._representation.reshape(1, -1)
        # scale representation to [-1,1]^N
        return self._representation, update_flag

    @property
    def initial_space(self):
        return self._initial_space

    @property
    def dim(self):
        return self._initial_space.shape[0]

class MassPointRepresenter():
  
    def __init__(self, nb_pt):
        self._description = ['engineer_goal_x', 'engineer_goal_y']
        self._initial_space = np.array([[-1.0, 1.0]]*nb_pt) # space in which goal are sampled
        self._representation = None

    def represent(self, obs_seq, act_seq, task, nb_pt):
        
        nb_pair = nb_pt//2
        update_flag = True
        #print('Obs seq in representer: ' + str(obs_seq))
        #print('Obs seq shape: ' + str(np.shape(obs_seq)))
        time_steps = max(1, int(obs_seq[~np.isnan(np.array(obs_seq))].reshape((7, -1)).shape[1] - 1))
        #print('real time step: ' + str(time_steps))
        obs_seq_mid = np.array(obs_seq[:,:time_steps])
        #print('Obs mid seq shape: ' + str(np.shape(obs_seq_mid)))
        if task == 'traj':
            for i in range(1, time_steps):
                #print(obs_seq[~np.isnan(np.array(obs_seq))].reshape((7, -1))[:, i])
                obs_seq_mid[:,i] = obs_seq[~np.isnan(np.array(obs_seq))].reshape((7, -1))[:, i]
                
            #obs_seq_mid = obs_seq[~np.isnan(np.array(obs_seq))].reshape((7, -1))[:, obs_seq.shape[1]//nb_pair]
        obs_seq = obs_seq[~np.isnan(np.array(obs_seq))].reshape((7, -1))[:, -1]
        obs_seq_mid = obs_seq_mid.reshape(7,-1)
        #print('Obs seq after reshape: ' + str(np.shape(obs_seq)))
        #print('Obs mid seq shape after process: ' + str(np.shape(obs_seq_mid)))
            #print('Obs mid seq after process: ' + str(obs_seq_mid.T[:,:2]))
        
        if task == 'traj':
            #print(np.concatenate([obs_seq_mid[:,:2], np.array([obs_seq[:2]]*time_steps)], axis=1))
            #print(np.shape(np.array([obs_seq[:2]]*time_steps)))
            #print(np.shape(obs_seq_mid.T[:,:2]))
            #print('Timesteps: ' + str(time_steps))
            #print('Shape of obs * time: ' + str(np.shape(np.array([obs_seq[:2]]*time_steps))))
            #print('Transposing obs_mid ' + str(obs_seq_mid[:2,:]))
            testing = obs_seq_mid.T
            if (testing == testing[0]).all():
                update_flag = False

            self._representation = np.hstack([obs_seq_mid.T[:,:2], np.array([obs_seq[:2]]*time_steps)])
            #print ('Representation in traj representer: ' + str(np.shape(self._representation)))
            #print ('Representation in traj representer: ' + str(self._representation))
            for i in range(time_steps):
                self._representation[i,:] = scale_vec(self._representation[i,:], self._initial_space)
                #print('representation', self._representation[i,:])
                #self._representation.reshape(i, -1)
        else:
            self._representation = np.array(obs_seq[:2])
            # scale representation to [-1,1]^N
            self._representation = scale_vec(self._representation, self._initial_space)
            self._representation.reshape(1, -1)
        
        
        return self._representation, update_flag

    @property
    def initial_space(self):
        return self._initial_space

    @property
    def dim(self):
        return self._initial_space.shape[0]
