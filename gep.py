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


saving_folder = './outputs/'
traj_folder = './trajectory/'
trial_id = 1
nb_runs = 1
env_id = 'Kobuki-v0' #'HalfCheetah-v2'# #'MountainCarContinuous-v0' #
study = 'GEP' # 'GEP' # 'GEPPG' # 'DDPG'
ddpg_noise = 'ou_0.3'# 'adaptive_param_0.2' #
nb_exploration = 500 # nb of episodes for gep exploration
traj_dict = dict()
obj_dict = dict()
n_traj = 0
gep_error = []

def run_experiment(env_id, trial, noise_type, study, nb_exploration, saving_folder, traj_folder):

    # create data path
    data_path = create_data_path(saving_folder, env_id, trial_id)
    task = args.task_type
    nb_pt = args.nb_pt

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
            nb_rep, engineer_goal, goal_space, initial_space, knn, noise, nb_weights = kobuki_config(task, nb_pt)

        
        # overun some settings
        nb_explorations = nb_exploration
        nb_bootstrap = nb_explorations//4
        nb_tests = args.nb_tests
        offline_eval = (1e6, 10) #(x,y): y evaluation episodes every x (done offline)

        train_perfs = []
        eval_perfs = []
        final_eval_perfs = []


        # compute test indices:
        test_ind = range(int(offline_eval[0])-1, nb_explorations, int(offline_eval[0]))

        # define environment
        if task == 'goal':
            env = gym.make('ReacherGEP-v0')
        else: 
            env = gym.make('ReacherGEPTraj-v0')
        
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
            #policy = policy * 0.1
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
                                        
            #save
            action_seqs = np.concatenate([action_seqs, act], axis=0)
            observation_seqs = np.concatenate([observation_seqs, obs], axis=0)
            reward_seqs = np.concatenate([reward_seqs, rew], axis=0)
            train_perfs.append(np.nansum(rew))
           
            # offline tests
            if ep in test_ind:
                '''while True:
                    xpos = np.random.sample() * 2 - 1
                    ypos = np.random.sample() * 2 - 1
                    if np.linalg.norm([xpos*0.21, ypos*0.21]) <= 0.21:
                        break
                while True:
                    mxpos = np.random.sample() * 2 - 1
                    mypos = np.random.sample() * 2 - 1
                    if np.linalg.norm([mxpos*0.21, mypos*0.21]) <= 0.21:
                        break
                if task == 'goal':
                else:
                    engineer_goal = np.array([mxpos, mypos, xpos, ypos])
                '''
                offline_evaluations(offline_eval[1], engineer_goal, knn, nb_rew, nb_timesteps, env, controller, eval_perfs)

        # final evaluation phase
        # # # # # # # # # # # # # # #
        for ep in range(nb_tests):
            theta = np.random.sample() * 360
            rad = np.random.sample()
            if task == 'goal':
                engineer_goal = np.array([np.cos(np.deg2rad(theta)) * rad, np.sin(np.deg2rad(theta)) * rad])
            else:
                m_theta = np.random.sample() * 360
                m_rad = np.random.sample()
                engineer_goal = np.array([np.cos(np.deg2rad(m_theta)) * m_rad, np.sin(np.deg2rad(m_theta)) * m_rad, np.cos(np.deg2rad(theta)) * rad, np.sin(np.deg2rad(theta)) * rad])
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
    #print('obs reset: ' + str(obs))
    rew[0, :, 0] = 0
    done = False  # termination signal
    env_timestep = 0
    task = args.task_type
    nb_pt = args.nb_pt

    global n_traj
    #env.render()
    # Check Observation Range
    for t in range(nb_timesteps):
        act[0, :, t] = controller.step(policy, obs[0, :, t]).reshape(1, -1)
        # check action as obs close to zero
        #x, y = (obs[:,0,t]*(1+obs[:,1,t])+obs[:,2,t]*obs[:,3,t])/10, (obs[:,2,t]*(1+obs[:,1,t])+obs[:,0,t]*obs[:,3,t])/10
        #if abs(x) <= 0.02 and abs(y) <= 0.02:
        #    print('x, y:', x, ' ', y)
        #    print('action:', act[0,:,t])
        out = env.step(np.copy(act[0, :, t]))
        
        obs[0, :, t + 1] = out[0]
        rew[0, :, t + 1] = out[1]
        done = out[2]
        info = out[3]
        env_timestep = info['t']

        if done:
            break
    # convert the trajectory into a representation (=behavioral descriptor)
    #env.close()
    rep = representer.represent(obs[0,:,0:env_timestep], act, task, nb_pt)
    #x, y = obs[:,0,:] + obs[:,1,:], obs[:,2,:] + obs[:,3,:]
    x, y = obs[:,0,:], obs[:,1,:]
    plt_obs = np.reshape(np.array([x,y]), (2, nb_timesteps+1))
    plt_obs = plt_obs.transpose()
    if task == 'goal':
        key = "_".join([str(plt_obs[env_timestep,0]), str(plt_obs[env_timestep,1]), str(n_traj)])
    else:
        key = "_".join([str(plt_obs[env_timestep//2,0]), str(plt_obs[env_timestep//2,1]), str(plt_obs[env_timestep,0]), str(plt_obs[env_timestep,1]), str(n_traj)])
    #print('env_timestep-1:' + str(plt_obs[env_timestep-1]))
    #print('env_timestep:' + str(plt_obs[env_timestep]))
    #traj_dict[key] = np.array(plt_obs)
    #n _traj += 1

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
    task = args.task_type
    nb_pt = args.nb_pt
    #print('best_policy', best_policy)

    returns = []
    for i in range(nb_eps):
        rew = np.zeros([nb_rew, nb_timesteps + 1])
        rew.fill(np.nan)

        env.reset()
        obs = env.unwrapped.reset_model(engineer_goal) # TODO: need pass config to environment
        rew[:, 0] = 0
        done = False
        info = {}
        env_timestep = 0

        x, y = obs[0], obs[1]
        plt_obs = [[x,y]] # plot

        for t in range(nb_timesteps):
            act = controller.step(best_policy, obs)
            out = env.step(np.copy(act))
            obs = out[0].squeeze().astype(np.float)
            rew[:, t + 1] = out[1]
            done = out[2]
            info = out[3]
            env_timestep = info['t']
            #env.render()

            x, y = obs[0], obs[1]
            plt_obs.append([x,y]) # plot

            if done: 
                break
        
        #env.close()
        returns.append(np.nansum(rew))

        if task == 'goal':
            key = "_".join([str(engineer_goal[0]), str(engineer_goal[1]), str(n_traj)])
        else:
            key = "_".join([str(engineer_goal[0]), str(engineer_goal[1]), str(engineer_goal[2]), str(engineer_goal[3]), str(n_traj)])
        #key = "_".join([str(plt_obs[env_timestep//2,0]), str(plt_obs[env_timestep//2,1]), str(plt_obs[env_timestep,0]), str(plt_obs[env_timestep,1]), str(n_traj)])
        traj_dict[key] = np.array(plt_obs)
        # write the observation to text file
        #with open(traj_folder + "agent_" + str(engineer_goal[0]) + str(engineer_goal[1]), "wb") as text_file:
            #pickle.dump(plt_obs, text_file)
            
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
    parser.add_argument('--env_id', type=str, default='Kobuki-v0')
    parser.add_argument('--noise_type', type=str, default=ddpg_noise)  # choices are adaptive-param_xx, ou_xx, normal_xx, decreasing-ou_xx, none
    parser.add_argument('--study', type=str, default=study) #'DDPG'  #'GEP_PG'
    parser.add_argument('--nb_exploration', type=int, default=nb_exploration)
    parser.add_argument('--nb_tests', type=int, default=100)
    parser.add_argument('--saving_folder', type=str, default=saving_folder)
    parser.add_argument('--traj_folder', type=str, default=traj_folder)
    parser.add_argument('--task_type', type=str, choices=['goal', 'traj'], default='goal')
    parser.add_argument('--nb_pt', type=int, default=2)
    args = parser.parse_args()

    gep_perf = np.zeros([nb_runs])

    trial_id = args.trial
    env_id = args.env_id
    noise_type = args.noise_type
    study = args.study
    nb_exploration = args.nb_exploration
    saving_folder = args.saving_folder
    traj_folder = args.traj_folder
    task = args.task_type
    nb_pt = args.nb_pt

    #pos_dict[0.,0.,0.]
    for i in range(nb_runs):
        gep_perf[i], policies = run_experiment(env_id, trial_id, noise_type, study, nb_exploration, saving_folder, traj_folder)
        print(gep_perf)
        print('Average performance: ', gep_perf.mean())
        #replay_save_video(env_id, policies, video_folder)
        #print('Average error: ' + str(gep_error[0].mean()) + " " + str(gep_error[1].mean()))
   
    #fig = plt.figure()
    #plt.axis([-2.0, 2.0, -2.0, 2.0])

    for key in traj_dict:
        fig = plt.figure()
        plt.axis([-1, 1, -1, 1])
        x_z = key.split('_')
        if task == 'goal':
            plt.plot(traj_dict[key][:,0], traj_dict[key][:,1], float(x_z[0]), float(x_z[1]), 'ro')
        else:
            plt.plot(traj_dict[key][:,0], traj_dict[key][:,1], float(x_z[0]), float(x_z[1]), 'bo')
            plt.plot(float(x_z[2]), float(x_z[3]), 'ro')
        #print(x_z[2])
        #plt.plot(float(x_z[0]), float(x_z[1]), 'ro')
        fig.savefig('figures/'+key + '.png')
        plt.close()

    #fig.savefig('figures/'+key + '.png')
    #plt.close()
