import os
import sys
sys.path.append('./')
from eval_dist_cem.eval_dist_cem import eval_dist_cem
import gym
import custom_gym
from unity_env import UnityEnv
import matplotlib.pyplot as plt
import pickle
import argparse
from controllers import NNController
from representers import CheetahRepresenter
from inverse_models import KNNRegressor
from gep_utils import *
from configs import *
from gep import *

saving_folder = './outputs/'
trial_id = 701
env_id = 'Kobuki-v0'
def run_testing():
    print('testing')

if __name__ == '__main__':
    target = input('Enter an instruction: ')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nb_eps', type=int, default=1)
    args = vars(parser.parse_args())
    gep_memory = dict()
    data_path = (saving_folder + str(env_id) + '/' + str(trial_id) + '/')

    with open(data_path+'save_gep.pk', 'rb') as f:
        gep_memory = pickle.load(f)
    
    knn = KNNRegressor(n_neighbors=1)
    eval_perfs = np.array(gep_memory['eval_perfs']).tolist()

    _, _, _, nb_timesteps, _, controller, representer, _, _, _, _, knn, _, _ = kobuki_config()
    knn.update(gep_memory['representations'], gep_memory['policies'])
    #engineer_goal = np.random.uniform(-1.0, 1.0, (2,))
    engineer_goal = eval_dist_cem(int(target))
    print(engineer_goal)
    env = gym.make('MassPoint-v0')
    nb_rew = 1
    nb_eps = 1
    offline_evaluations(nb_eps, engineer_goal, knn, nb_rew, nb_timesteps, env, controller, eval_perfs)
    target_angel = range(18, 180, 36)
    target_rad = np.deg2rad(target_angel[int(target)])
    target_x, target_y = np.cos(target_rad), np.sin(target_rad)
    for key in traj_dict:
        fig = plt.figure()
        plt.axis([-1.0, 1.0, -1.0, 1.0])
        plt.plot(traj_dict[key][:,0], traj_dict[key][:,1], target_x, target_y, 'ro')
        fig.savefig('results/'+ key +'.png')
        plt.show()
        plt.close()
