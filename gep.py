import os
import sys
sys.path.append('./')
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import gym
import custom_gym
from unity_env import UnityEnv
import matplotlib.pyplot as plt
import pickle
import argparse
import random
from DDPG_baseline_v2.baselines.ddpg.main_config import run_ddpg
from DDPG_baseline_v2.baselines.ddpg.configs.config import ddpg_config
from controllers import NNController
from representers import CheetahRepresenter
from inverse_models import KNNRegressor
from gep_utils import *
from configs import *


saving_folder = './results/'
traj_folder = './trajectory/'
trial_id = 1
nb_runs = 1
env_id = 'Kobuki-v0' #'HalfCheetah-v2'# #'MountainCarContinuous-v0' #
study = 'DDPG' # 'GEP' # 'GEPPG' #
ddpg_noise = 'ou_0.3'# 'adaptive_param_0.2' #
nb_exploration = 500 # nb of episodes for gep exploration
traj_dict = dict()
obj_dict = dict()
n_traj = 0
sample_obs = []

def run_experiment(env_id, trial, noise_type, study, nb_exploration, saving_folder, traj_folder):

    # create data path
    data_path = create_data_path(saving_folder, env_id, trial_id)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # GEP
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    if 'GEP' in study:
        # get GEP config
        if env_id=='HalfCheetah-v2':
            nb_bootstrap, nb_explorations, nb_tests, nb_timesteps, offline_eval, controller, representer,\
            nb_rep, engineer_goal, goal_space, initial_space, knn, noise, nb_weights = cheetah_config()
        elif env_id=='MountainCarContinuous-v0':
            nb_bootstrap, nb_explorations, nb_tests, nb_timesteps, offline_eval, controller, representer, \
            nb_rep, engineer_goal, goal_space, initial_space, knn, noise, nb_weights = cmc_config()
        elif env_id=='Kobuki-v0':
            nb_bootstrap, nb_explorations, nb_tests, nb_timesteps, offline_eval, controller, representer, \
            nb_rep, engineer_goal, goal_space, initial_space, knn, noise, nb_weights = kobuki_config()

        # overun some settings
        nb_explorations = nb_exploration
        nb_bootstrap = nb_explorations/4
        #nb_tests = 100
        offline_eval = (1e6, 10) #(x,y): y evaluation episodes every x (done offline)

        train_perfs = []
        eval_perfs = []
        final_eval_perfs = []

        # compute test indices:
        test_ind = range(int(offline_eval[0])-1, nb_explorations, int(offline_eval[0]))

        # define environment
        env = gym.make('FiveTarget-v1')
        nb_act = env.action_space.shape[0]
        nb_obs = env.observation_space.shape[0]
        nb_rew = 1
        action_seqs = np.array([]).reshape(0, nb_act, nb_timesteps)
        observation_seqs = np.array([]).reshape(0, nb_obs, nb_timesteps+1)
        reward_seqs = np.array([]).reshape(0, nb_rew, nb_timesteps+1)

        # bootstrap phase
        # # # # # # # # # # #
        for ep in range(nb_bootstrap):
            print('Bootstrap episode #', ep+1)
            # sample policy at random
            policy = np.random.random(nb_weights) * 2 - 1
            #print('policy' + str(policy))

            # play policy and update knn
            obs, act, rew = play_policy(policy, nb_obs, nb_timesteps, nb_act, nb_rew, env, controller,
                                        representer, knn)

            # save
            action_seqs = np.concatenate([action_seqs, act], axis=0)
            observation_seqs = np.concatenate([observation_seqs, obs], axis=0)
            reward_seqs = np.concatenate([reward_seqs, rew], axis=0)
            train_perfs.append(np.nansum(rew))

            # offline tests
            if ep in test_ind:
                print('Engineer Goal:')
                print(engineer_goal)
                offline_evaluations(offline_eval[1], engineer_goal, knn, nb_rew, nb_timesteps, env, controller, eval_perfs)

        # exploration phase
        # # # # # # # # # # # #
        for ep in range(nb_bootstrap, nb_explorations):
            print('Random Goal episode #', ep+1)

            # random goal strategy
            policy = random_goal(nb_rep, knn, goal_space, initial_space, noise, nb_weights)

            # play policy and update knn
            obs, act, rew = play_policy(policy, nb_obs, nb_timesteps, nb_act, nb_rew, env, controller,
                                        representer, knn)
                                        
            #save
            action_seqs = np.concatenate([action_seqs, act], axis=0)
            observation_seqs = np.concatenate([observation_seqs, obs], axis=0)
            reward_seqs = np.concatenate([reward_seqs, rew], axis=0)
            train_perfs.append(np.nansum(rew))
           
            # offline tests
            if ep in test_ind:
                engineer_goal = np.random.uniform(-1.0, 1.0, (2,))
                print('Engineer Goal:')
                print(engineer_goal)
                offline_evaluations(offline_eval[1], engineer_goal, knn, nb_rew, nb_timesteps, env,
                                    controller, eval_perfs)

        # final evaluation phase
        # # # # # # # # # # # # # # #
        for ep in range(nb_tests):
            #engineer_goal[0] = np.random.uniform(-.5, .5)
            #engineer_goal[1] = np.random.uniform(-.5, 0)
            engineer_goal = np.random.uniform(-1.0, 1.0, (2,))
            #print('Test episode #', ep+1)
            #print('Engineer Goal:')
            #print(engineer_goal)
            #print('nb_test:', nb_tests)
            best_policy = offline_evaluations(1, engineer_goal, knn, nb_rew, nb_timesteps, env, controller, final_eval_perfs)

        #print('Observation:')
        #print(obs)
        print('Run performance: ', np.nansum(rew))

        print('Final performance for the run: ', np.array(final_eval_perfs).mean())

        # wrap up and save
        # # # # # # # # # # #
        gep_memory = dict()
        gep_memory['actions'] = action_seqs.swapaxes(1, 2)
        gep_memory['observations'] = observation_seqs.swapaxes(1, 2)
        gep_memory['rewards'] = reward_seqs.swapaxes(1, 2)
        gep_memory['best_policy'] = best_policy
        gep_memory['train_perfs'] = np.array(train_perfs)
        gep_memory['eval_perfs'] = np.array(eval_perfs)
        gep_memory['final_eval_perfs'] = np.array(final_eval_perfs)
        gep_memory['representations'] = knn._X
        gep_memory['policies'] = knn._Y
        gep_memory['metrics'] = compute_metrics(gep_memory) # compute metrics for buffer analysis
        #print(gep_memory)

        with open(data_path+'save_gep.pk', 'wb') as f:
            pickle.dump(gep_memory, f)
    
    # print(pickle.load(open(data_path+'save_gep.pk', 'r')))

    return np.array(final_eval_perfs).mean(), knn._Y

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # DDPG
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



def play_policy(policy, nb_obs, nb_timesteps, nb_act, nb_rew, env, controller, representer, knn):
    """
    Play a policy in the environment for a given number of timesteps, usin a NN controller.
    Then represent the trajectory and update the inverse model.
    """
    obs = np.zeros([1, nb_obs, nb_timesteps + 1])
    act = np.zeros([1, nb_act, nb_timesteps])
    rew = np.zeros([1, nb_rew, nb_timesteps + 1])
    obs.fill(np.nan)
    act.fill(np.nan)
    rew.fill(np.nan)
    obs[0, :, 0] = env.reset()
    rew[0, :, 0] = 0
    done = False  # termination signal
    max_timestep = False
    #env_timestep = 0

    # Check Observation Range
    for t in range(nb_timesteps):
        #print('policy:' + str(policy))
        #print('obs:' + str(obs[0, :, t]))
        act[0, :, t] = controller.step(policy, obs[0, :, t]).reshape(1, -1)
        #print('act:' + str(act[0, :, t]))
        out = env.step(np.copy(act[0, :, t]))
        

        # env.render()
        obs[0, :, t + 1] = out[0]
        rew[0, :, t + 1] = out[1]
        done = out[2]
        #info = out[3]
        #env_timestep = info['t']

        if done:
            #print('(done)t & env_t:' + str(t) + ' ' + str(env_timestep))
            #if env_timestep == 200:
            #    max_timestep = True
            break

    # convert the trajectory into a representation (=behavioral descriptor)
    rep = representer.represent(obs, act)
    plt_obs = np.reshape(np.array(obs), (7,nb_timesteps+1))
    plt_obs = plt_obs.transpose()
    #print(plt_obs)
    #key = "_".join([str(plt_obs[env_timestep//2,0]), str(plt_obs[env_timestep//2,1]), str(plt_obs[env_timestep,0]), str(plt_obs[env_timestep,1])])
    #traj_dict[key] = np.array(plt_obs)
    #print('Representatio: ' + str(rep))
    #print('obs: ' + str(obs[0, :, env_timestep]))

    '''if max_timestep == True:
        rep = np.array([0.0, 0.0])
        #print('Representation w/ Max:' + str(rep))
        #print('obs: ' + str(obs[0, :, -1]))
    else:
        rep = representer.represent(obs, act)
        print('Representation w/o Max:' + str(rep))
        print('obs: ' + str(obs[0, :, env_timestep]))'''

    # update inverse model
    #print ('knn._X: '+str(knn._X))
    knn.update(X=rep, Y=policy)

    return obs, act, rew

def offline_evaluations(nb_eps, engineer_goal, knn, nb_rew, nb_timesteps, env, controller, eval_perfs):
    """
    Play the best policy found in memory to test it. Play it for nb_eps episodes and average the returns.
    """
    # use scaled engineer goal and find policy of nearest neighbor in representation space
    global n_traj
    #print('engl: ' + str(engineer_goal))
    #coord = [18, 54, 90, 126, 162]
    #np.array([np.cos(np.deg2rad(coord[n_traj])), np.sin(np.deg2rad(coord[n_traj]))])
    best_policy = knn.predict(engineer_goal)[0, :]

    returns = []
    for i in range(nb_eps):
        rew = np.zeros([nb_rew, nb_timesteps + 1])
        rew.fill(np.nan)

        env.reset()
        #obs = env.unwrapped.reset(task=engineer_goal) # TODO: need pass config to environment
        obs = env.unwrapped.reset() # TODO: need pass config to environment
        rew[:, 0] = 0
        done = False
        info = {}
        plt_obs = [obs] # plot
        
        for t in range(nb_timesteps):
            if done: break
            
            act = controller.step(best_policy, obs)
            out = env.step(np.copy(act))
            obs = out[0].squeeze().astype(np.float)
            rew[:, t + 1] = out[1]
            done = out[2]
            info = out[3]
            
            plt_obs.append(obs) # plot

        returns.append(np.nansum(rew))
        target = np.where(np.array(obs[2:] == engineer_goal))
        
        key = "_".join([str(engineer_goal[0]), str(engineer_goal[1]), str(n_traj)])
        #key = "_".join([str(engineer_goal[0]), str(engineer_goal[1]), str(engineer_goal[2]), str(engineer_goal[3]), str(n_traj), str(info['hit'])])
        #traj_dict[key] = np.array(plt_obs)
        # write the observation to text file
        #with open(traj_folder + "agent_" + str(engineer_goal[0]) + str(engineer_goal[1]) + str(engineer_goal[2]) + str(engineer_goal[3]), "wb") as text_file:
        with open(traj_folder + "agent_" + str(engineer_goal[0]) + str(engineer_goal[1]), "wb") as text_file:
            pickle.dump(plt_obs, text_file)
            
        n_traj += 1

    eval_perfs.append(np.array(returns).mean())

    return best_policy

def random_goal(nb_rep, knn, goal_space, initial_space, noise, nb_weights):
    """
    Draw a goal, find policy associated to its nearest neighbor in the representation space, add noise to it.
    """
    # draw goal in goal space
    goal = np.copy(sample(goal_space))
    # scale goal to [-1,1]^N
    goal = scale_vec(goal, initial_space)

    # find policy of nearest neighbor
    policy = knn.predict(goal)[0]

    # add exploration noise
    policy += np.random.normal(0, noise*2, nb_weights) # noise is scaled by space measure
    policy_out = np.clip(policy, -1, 1)

    return policy_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trial', type=int, default=trial_id)
    parser.add_argument('--env_id', type=str, default=env_id)
    parser.add_argument('--noise_type', type=str, default=ddpg_noise)  # choices are adaptive-param_xx, ou_xx, normal_xx, decreasing-ou_xx, none
    parser.add_argument('--study', type=str, default=study) #'DDPG'  #'GEP_PG'
    parser.add_argument('--nb_exploration', type=int, default=nb_exploration)
    parser.add_argument('--saving_folder', type=str, default=saving_folder)
    parser.add_argument('--traj_folder', type=str, default=traj_folder)
    #parser.add_argument('--video_folder', type=str, default=video_folder)
    args = vars(parser.parse_args())

    gep_perf = np.zeros([nb_runs])
    
    #pos_dict[0.,0.,0.]
    for i in range(nb_runs):
        gep_perf[i], policies = run_experiment(**args)
        print(gep_perf)
        print('Average performance: ', gep_perf.mean())
        #replay_save_video(env_id, policies, video_folder)
   
    
    #for key in traj_dict:
     #   fig = plt.figure()
       # plt.axis([-1.0, 1.0, -1.0, 1.0])
        
       # x_z = key.split('_')

        #plt.plot(traj_dict[key][:,0], traj_dict[key][:,1], float(x_z[0]), float(x_z[1]), 'ro')
        #plt.plot(traj_dict[key][:,0], traj_dict[key][:,1])
        #plt.plot(float(x_z[0]), float(x_z[1]), 'bo')
       # plt.plot(float(x_z[2]), float(x_z[3]), 'ro')
        #fig.savefig('figures/'+ key +'.png')
        #plt.close()'''
