import os
import sys
import csv
sys.path.append('./')
from eval_dist_cem.eval_dist_cem import Dist_CEM
import gym
import custom_gym
import numpy as np
import pickle
import argparse
from controllers import NNController
from representers import MassPointRepresenter
from inverse_models import KNNRegressor
from gep_utils import *
from configs import *
from gep import *

traj_dict = dict()
n_traj = 0
avg_error = []

def run_testing(target, mid_target, engineer_goal, knn, obs, nb_timesteps, env, controller):
    """
    Play the best policy found in memory to test it. Play it for nb_eps episodes and average the returns.
    """
    # use scaled engineer goal and find policy of nearest neighbor in representation space
    global n_traj
    task = args.task
    nb_pt = args.nb_pt
    # print('goal:', engineer_goal)
    best_policy = knn.predict(engineer_goal)[0, :]
    # print('best_policy', best_policy)
    # rew = np.zeros([nb_rew, nb_timesteps + 1])
    # rew.fill(np.nan)
    # rew[:, 0] = 0
    
    done = False
    plt_obs = [obs] # plot
        
    for t in range(nb_timesteps):
        if done: break
        act = controller.step(best_policy, obs)
        out = env.step(np.copy(act))
        obs = out[0].squeeze().astype(np.float)
        # rew[:, t + 1] = out[1]
        done = out[2]
        #env.render()
        plt_obs.append(obs) # plot
    #env.close()
    
    if task == 'goal':
        key = "_".join([str(n_traj), str(target), str(engineer_goal[0]), str(engineer_goal[1])])
    else:
        key = "_".join([str(n_traj), str(target), str(mid_target), str(engineer_goal[0]), str(engineer_goal[1]), str(engineer_goal[2]), str(engineer_goal[3])])
        
    #key = "_".join([str(engineer_goal[0]), str(engineer_goal[1]), str(engineer_goal[2]), str(engineer_goal[3]), str(n_traj), str(info['hit'])])
    traj_dict[key] = np.array(plt_obs)
    n_traj += 1
    
    real_dis = find_closest(target, target_mid, plt_obs, env_id)

    if task == 'goal':
        return obs[:2]
    else:
        return real_dis

def find_closest(target, target_mid, real_traj, env_id):
    #print('target in find closest: ', target)
    #print('target_mid in find closest: ', target_mid)
    mid_pos = np.zeros(2)
    pos = np.zeros(2)
    real_traj = np.array(real_traj)[:,:2]
    if env_id == 'Mass-point':
        mid_pos[0], mid_pos[1], pos[0], pos[1] = mass_target_position(target, target_mid, 'traj')
    else:
        mid_pos[0], mid_pos[1] = reacher_target_position(target_mid, 'traj')
        pos[0], pos[1] = reacher_target_position(target, 'traj')
    ctcp = np.argmin(np.linalg.norm(real_traj-mid_pos, axis=1))
    ctft = np.argmin(np.linalg.norm(real_traj[ctcp:]-pos, axis=1))
    # min_dist_cp = np.linalg.norm(real_traj[ctcp]-mid_pos)
    # min_dist_ft = np.linalg.norm(real_traj[ctft]-pos)
    ctft = ctft + ctcp

    # return min_dist_cp + min_dist_ft
    return np.concatenate((real_traj[ctcp], real_traj[ctft]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trial_id', type=str, help='Iteration ID, ends with 1', default='0')
    parser.add_argument('--env_id', type=str, help='Environment ID', default='Mass-point')
    parser.add_argument('--data_folder', type=str, help='Folder name', default='1')
    parser.add_argument('--task', type=str, help='Goal or traj oriented', default='goal')
    parser.add_argument('--noise', type=str, help='Value of noise in training', default='0.05')
    parser.add_argument('--n_neighbors', type=int, help='The number of k in nearest neighbor', default=1)
    parser.add_argument('--nb_eps', type=int, help='The number of episodes to evalutate', default=100)
    parser.add_argument('--nb_pt', type=int, help='Number of points', default=2)
    parser.add_argument('--saving_folder', type=str, help='Path of .pk file save', default='./outputs/')
    parser.add_argument('--save_plot', type=bool, help='To save figure or not', default=False)
    parser.add_argument('--output', type=str, help='Output filename', default='testing.csv')
    parser.add_argument('--dist', type=str, help='Dist mode name', default=None)
    args = parser.parse_args()
    
    nb_rew      = 1
    n_neighbors = args.n_neighbors
    nb_eps      = args.nb_eps
    env_id      = args.env_id
    trial_id    = args.trial_id
    task        = args.task
    saving_folder = args.saving_folder
    noise       = args.noise
    data_folder = args.data_folder 
    data_path   = (saving_folder + env_id + '/' + data_folder + '/' + noise + '_' + trial_id + '_itr.pk')
    save_plot = args.save_plot
    dist_model = args.dist
    # print(data_path)

    gep_memory = dict()
    with open(data_path, 'rb') as f:
        gep_memory = pickle.load(f)
    
    if env_id == 'Mass-point':
        env = gym.make('FiveTargetEnv-v1')
    else:
        # print('Error before make gym')
        env = gym.make('ReacherGEP-v0')
        # print('Error after make gym')

    nb_act = env.action_space.shape[0]

    if env_id == 'Mass-point':
        nb_timesteps, controller, representer = mass_test_config(args.nb_pt, env_id, nb_act)
    else:
        nb_timesteps, controller, representer = reacher_test_config(args.nb_pt, env_id, nb_act)

    knn = KNNRegressor(n_neighbors)
    knn.init_update(gep_memory['representations'], gep_memory['policies'])

    
    if task == 'goal':
        task_id = 2
        #model_dir = './eval_dist_cem/DistModel/mass_point/64/'
    else: 
        task_id = 3
        #model_dir = './eval_traj/DistModel/MPT/128/'
    dist_cem = Dist_CEM(task, env_id, dist_model)
    shuffle_idx = [1, 3, 0, 4, 2]

    for i in range(nb_eps):
        target = i % 5
        target_mid = i % 2 + 5
        env.reset()
        if env_id == 'Mass-point':
            obs = env.unwrapped.reset(np.array([task_id, target_mid, target, 0., 0.]))
        else:
            obs = env.unwrapped.reset_model(np.array([task_id, target_mid, target, 0., 0.]))
        if task == 'goal':
            # print ('Traget: ', np.shape(target))
            # TODO: make change of cem and sgd
            goal = dist_cem.eval_dist_cem(target)
            if env_id == 'Mass-point':
                # Change for instruction shuffle
                # x, y = mass_target_position(target, target_mid, task)
                x, y = mass_target_position(target, target_mid, task)
                ideal_pos = [x,y]
            else:
                ideal_pos = [reacher_target_position(target, task)]
        else:
            if env_id == 'Mass-point':
                goal = dist_cem.eval_traj(target+2, target_mid-5)
                mid_x, mid_y, x, y = mass_target_position(target, target_mid, task)
                ideal_pos = [mid_x, mid_y, x, y]
            else:
                goal = dist_cem.eval_traj(target_mid-5, target+2)
                ideal_pos = np.concatenate((reacher_target_position(target_mid, task), reacher_target_position(target, task))) 

        last_pos = run_testing(target, target_mid, goal, knn, obs, nb_timesteps, env, controller)
        #print ('Goal evaluated by dist_cem: ', goal)
        #print ('Real target position: ', ideal_pos)
        #print ('Last agent position: x: ' + str(last_pos[0]) + ' y: ' + str(last_pos[1]))
        #print('l2norm: ', last_pos)
        if task == 'goal':
            error = np.linalg.norm(last_pos - ideal_pos)
            #print ('Error: ', error)
            avg_error.append(error)
        else:
            vec = np.linalg.norm(last_pos[0:2] - ideal_pos[0:2])
            vec2 = np.linalg.norm(last_pos[2:4] - ideal_pos[2:4])
            avg_error.append(vec+vec2)

    print('Average error: ' + str (np.array(avg_error).mean()))
    total_timesteps = int(trial_id) * nb_timesteps
    with open(args.output, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow([str(total_timesteps), noise, str(n_neighbors), str(np.array(avg_error).mean())])
    
    mass_test_plot(save_plot, traj_dict, task, env_id)
