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

def run_experiment(env_id, trial, noise_type, study, nb_exploration, saving_folder):

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
        nb_tests = 100
        offline_eval = (1e6, 10) #(x,y): y evaluation episodes every x (done offline)

        train_perfs = []
        eval_perfs = []
        final_eval_perfs = []

        # compute test indices:
        test_ind = range(int(offline_eval[0])-1, nb_explorations, int(offline_eval[0]))

        # define environment
        env = gym.make('FiveTargetColor-v0')
        #env = UnityEnv('UnityGEP_2+20/kobuki.x86_64', 1)
        nb_act = env.action_space.shape[0]
        nb_obs = env.observation_space.shape[0]
        nb_rew = 1
        action_seqs = np.array([]).reshape(0, nb_act, nb_timesteps)
        observation_seqs = np.array([]).reshape(0, nb_obs, nb_timesteps+1)
        reward_seqs = np.array([]).reshape(0, nb_rew, nb_timesteps+1)
        n_traj = 1

        # bootstrap phase
        # # # # # # # # # # #
        for ep in range(nb_bootstrap):
            print('Bootstrap episode #', ep+1)
            # sample policy at random
            policy = np.random.random(nb_weights) * 2 - 1

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
                offline_evaluations(offline_eval[1], engineer_goal, knn, nb_rew, nb_timesteps, env,
                                    controller, eval_perfs)

        # exploration phase
        # # # # # # # # # # # #
        for ep in range(nb_bootstrap, nb_explorations):
            print('Random Goal episode #', ep+1)

            # random goal strategy
            policy = random_goal(nb_rep, knn, goal_space, initial_space, noise, nb_weights)

            # play policy and update knn
            obs, act, rew = play_policy(policy, nb_obs, nb_timesteps, nb_act, nb_rew, env, controller,
                                        representer, knn)
            #print('Reward')
            #print(rew)
            # save
            action_seqs = np.concatenate([action_seqs, act], axis=0)
            observation_seqs = np.concatenate([observation_seqs, obs], axis=0)
            reward_seqs = np.concatenate([reward_seqs, rew], axis=0)
            train_perfs.append(np.nansum(rew))
            #TODO dirty bug
            if (ep+1) % 20 == 0.0:
                print('Engineer Goal:')
                #engineer_goal = np.random.random_sample((3))
                ran_key, engineer_goal = random.choice(list(obj_dict.items()))
                print(engineer_goal)
                offline_evaluations(1, engineer_goal, knn, nb_rew, nb_timesteps, env, controller, eval_perfs) 
            # offline tests
            if ep in test_ind:
                offline_evaluations(offline_eval[1], engineer_goal, knn, nb_rew, nb_timesteps, env, controller, eval_perfs)

        # final evaluation phase
        # # # # # # # # # # # # # # #
        for ep in range(nb_tests):
            print('Test episode #', ep+1)
            print('Engineer Goal:')
            #engineer_goal = np.random.random_sample((3,)) 
            ran_key, engineer_goal = random.choice(list(obj_dict.items()))
            print(engineer_goal)
            best_policy = offline_evaluations(1, engineer_goal, knn, nb_rew, nb_timesteps, env, controller, final_eval_perfs)

        print('Observation:')
        print(obs)
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

    # Check Observation Range
    #print ('Observation Range')
    #print(env.observation_space.low)
    #print(env.observation_space.high)
    for t in range(nb_timesteps):
        if done:
            break
        elif t == 210:
            max_timestep = True
            break
        #print('policy')
        #print(policy)
        #print('observation:')
        #print(obs[0,:,t])
        act[0, :, t] = controller.step(policy, obs[0, :, t]).reshape(1, -1)
        #print('actions:')
        #print(act[0,:,t])
        out = env.step(np.copy(act[0, :, t]))
        # env.render()
        obs[0, :, t + 1] = out[0]
        rew[0, :, t + 1] = out[1]
        done = out[2]

    # convert the trajectory into a representation (=behavioral descriptor)
    if max_timestep == True:
        rep = np.array([ 0.5, 0.5, 0.5])    
        print('Representation w/ Max:' + str(rep))
    else:
        rep = representer.represent(obs, act)
        print('Representation w/o Max:' + str(rep))

    # update inverse model
    print(rep)
    knn.update(X=rep, Y=policy)

    return obs, act, rew

def offline_evaluations(nb_eps, engineer_goal, knn, nb_rew, nb_timesteps, env, controller, eval_perfs):
    """
    Play the best policy found in memory to test it. Play it for nb_eps episodes and average the returns.
    """
    # use scaled engineer goal and find policy of nearest neighbor in representation space
    global n_traj
    best_policy = knn.predict(engineer_goal)[0, :]

    returns = []
    for i in range(nb_eps):
        rew = np.zeros([nb_rew, nb_timesteps + 1])
        rew.fill(np.nan)
        goal_dict = dict()
        goal_dict['engineer_goal_r'] = engineer_goal[0]
        goal_dict['engineer_goal_g'] = engineer_goal[1]
        goal_dict['engineer_goal_b'] = engineer_goal[2]

        obs = env.reset(config=goal_dict)
        plt_obs = np.array(obs)
        rew[:, 0] = 0
        done = False
        for t in range(nb_timesteps):
            if done:
                break
            act = controller.step(best_policy, obs).reshape(1, -1)
            out = env.step(np.copy(act))
            obs = out[0].squeeze().astype(np.float)
            rew[:, t + 1] = out[1]
            done = out[2]
            plt_obs = np.concatenate((plt_obs,np.array([obs])[0:2]),axis=0)
        returns.append(np.nansum(rew))
        plt_obs = np.transpose(plt_obs)
        
        target = np.where(np.array(obs[2:] == engineer_goal))[0]
        
        key = "_".join([str(i) for i in engineer_goal])
        key = "_".join([key,str(i)])
        key = "_".join([key,str(n_traj)])
        print(n_traj)
        n_traj += 1
        traj_dict[key] = plt_obs
        print('Plt_obs')
        print(plt_obs)
    print(obs)
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
    #parser.add_argument('--video_folder', type=str, default=video_folder)
    args = vars(parser.parse_args())

    gep_perf = np.zeros([nb_runs])
    obj_dict['1'] = np.array([1.,0.,0.])
    obj_dict['2'] = np.array([0.,1.,0.])
    obj_dict['3'] = np.array([0.,0.,1.])
    obj_dict['4'] = np.array([1.,0.,1.])
    obj_dict['5'] = np.array([0.,1.,1.])
    
    #pos_dict[0.,0.,0.]
    for i in range(nb_runs):
        gep_perf[i], policies = run_experiment(**args)
        print(gep_perf)
        print('Average performance: ', gep_perf.mean())
        #replay_save_video(env_id, policies, video_folder)
    for key in traj_dict:
        fig = plt.figure()
        plt.axis([-1.0,1.0,-1.0,1.0])
        
        x_z = key.split('_')
        x_z = [float(i) for i in x_z]
        
        plt.plot(traj_dict[key][0], traj_dict[key][1], x_z[0], x_z[1], 'ro')
        fig.savefig('figures/'+key+'.png')
        plt.close()

