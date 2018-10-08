import os
import sys
sys.path.append('./')
import numpy as np
import gym
import custom_gym
import matplotlib.pyplot as plt
import pickle
import argparse
import random
from controllers import NNController
from representers import MassPointRepresenter
from inverse_models import KNNRegressor
from gep_utils import *
from configs import *

train_obs = dict()
eval_obs  = dict()
obj_dict = dict()
n_traj = 0

def run_experiment(env_id, trial, nb_exploration, saving_folder, eval_ins):

    # create data path
    print('ENV_ID' + env_id)
    data_path = create_data_path(saving_folder, env_id, trial)
    task = args.task_type
    nb_pt = args.nb_pt
    cus_noise = args.cus_noise

    # define environment
    if env_id == 'Mass-point':
        env = gym.make('FiveTargetEnv-v1')
    else:
        env = gym.make('ReacherGEP-v0')
    nb_act = env.action_space.shape[0]
    nb_obs = env.observation_space.shape[0]
    nb_rew = 1

    # get GEP config
    if env_id=='Reacher':
        nb_bootstrap, nb_explorations, nb_tests, nb_timesteps, offline_eval, controller, representer,\
        nb_rep, engineer_goal, goal_space, initial_space, knn, noise, nb_weights = reacher_train_config(env_id, nb_pt, cus_noise, nb_act)
    elif env_id=='Mass-point':
        nb_bootstrap, nb_explorations, nb_tests, nb_timesteps, offline_eval, controller, representer, \
        nb_rep, engineer_goal, goal_space, initial_space, knn, noise, nb_weights = mass_train_config(env_id, nb_pt, cus_noise, nb_act)

    # overun some settings
    nb_explorations = nb_exploration
    nb_bootstrap = int(nb_explorations/4)
    nb_tests = args.nb_tests
    if args.save_pickle:
        offline_eval = (1, 20) #(x,y): y evaluation episodes every x (done offline)
    else:
        offline_eval = (1e6, 20)

    # train_perfs = []
    # eval_perfs = []
    # final_eval_perfs = []

    # compute test indices:
    test_ind = range(int(offline_eval[0])-1, nb_explorations+int(offline_eval[1]), int(offline_eval[1]))

    action_seqs = np.array([]).reshape(0, nb_act, nb_timesteps)
    observation_seqs = np.array([]).reshape(0, nb_obs, nb_timesteps+1)
    reward_seqs = np.array([]).reshape(0, nb_rew, nb_timesteps+1)

    # bootstrap phase
    # # # # # # # # # # #
    for ep in range(nb_bootstrap):
        # for random max timepsteps
        if env_id == 'Mass-point':
            nb_timesteps = np.random.randint(10, 51)
        else:
            nb_timesteps = np.random.randint(5, 21)
          
        action_seqs = np.array([]).reshape(0, nb_act, nb_timesteps)
        observation_seqs = np.array([]).reshape(0, nb_obs, nb_timesteps+1)
        reward_seqs = np.array([]).reshape(0, nb_rew, nb_timesteps+1)

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
        # train_perfs.append(np.nansum(rew))

        # offline tests
        if ep in test_ind:
            file_path = saving_folder + env_id + '/' + trial_id + '/' + str(noise) + '_' + str(int(ep)) + '_itr.pk'
            write_file(knn, file_path)

    # exploration phase
    # # # # # # # # # # # #
    for ep in range(nb_bootstrap, nb_explorations):
        # for random max timepsteps
        if env_id == 'Mass-point':
            nb_timesteps = np.random.randint(10, 51)
        else:
            nb_timesteps = np.random.randint(5, 21)
        action_seqs = np.array([]).reshape(0, nb_act, nb_timesteps)
        observation_seqs = np.array([]).reshape(0, nb_obs, nb_timesteps+1)
        reward_seqs = np.array([]).reshape(0, nb_rew, nb_timesteps+1)

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
        # train_perfs.append(np.nansum(rew))
            
        if ep in test_ind:
            file_path = saving_folder + env_id + '/' + trial_id + '/' + str(noise) + '_' + str(int(ep)) + '_itr.pk'
            # file_path = './outputs/Kobuki-v0/mass-point-act/' + str(noise) + '_' + str(int(ep)) + '_itr.pk'
            write_file(knn, file_path)

    # for random max timepsteps
    if env_id == 'Mass-point':
        nb_timesteps = 50
    else:
        nb_timesteps = 20
    action_seqs = np.array([]).reshape(0, nb_act, nb_timesteps)
    observation_seqs = np.array([]).reshape(0, nb_obs, nb_timesteps+1)
    reward_seqs = np.array([]).reshape(0, nb_rew, nb_timesteps+1)
    # final evaluation phase
    # # # # # # # # # # # # # # #
    if env_id == 'Reacher' and task == 'goal':
        mid_goal = np.array([[.1, .1], [.1, -.1]]) / 0.21
        f_goal = np.array([[0, .15], [-.1, .1], [-.2, 0], [-.1, -.1], [0, -.15]]) / 0.21
    elif env_id == 'Reacher' and task == 'traj':
        mid_goal = np.array([[np.cos(np.deg2rad(45)), np.sin(np.deg2rad(45))], [np.cos(np.deg2rad(315)), np.sin(np.deg2rad(315))]])
        f_goal = []
        for x in range(5):
            f_goal.append([np.cos(np.deg2rad(180)) * x * 0.2 - 0.1, np.sin(np.deg2rad(180)) * x * 0.2])
            
    for ep in range(nb_tests):
        engineer_goal = engineer_goal_sample(task, env_id, eval_ins, engineer_goal)
        # if task == 'goal':
        #     engineer_goal[:2] = np.random.uniform(-1.0, 1.0, (2,))
        # elif task == 'traj':
        #     engineer_goal[:2] = np.random.uniform(-1., 1., (2,))
        #     engineer_goal[2:4] = np.random.uniform(-1.0, 1.0, (2,))
        # else:
        #     print('Error of task type')
        print('Test episode #', ep+1)
        print('nb_test:', nb_tests)
        best_policy = offline_evaluations(1, engineer_goal, knn, nb_rew, nb_timesteps, env, controller)
    # print('Final performance for the run: ', np.array(final_eval_perfs).mean())

    # wrap up and save
    # # # # # # # # # # #
    file_path = saving_folder + env_id + '/' + str(trial_id) + '/' + str(noise) + '_final.pk'
    # file_path = './outputs/Kobuki-v0/mass-point-act/' + str(noise) + '_final.pk'
    write_file(knn, file_path)

    # return np.array(final_eval_perfs).mean(), knn._Y
    return knn._Y

def engineer_goal_sample(task, env_id, eval_ins, engineer_goal):
    if env_id == 'Mass-point':
        if task == 'goal':
            engineer_goal[:2] = np.random.uniform(-1.0, 1.0, (2,))
        elif task == 'traj':
            engineer_goal[:2] = np.random.uniform(-1., 1., (2,))
            engineer_goal[2:4] = np.random.uniform(-1.0, 1.0, (2,))
        else:
            print('Error Task Type')
    else:
        theta = np.random.sample() * 360
        rad = np.random.sample()
        if task == 'goal':
            engineer_goal = np.array([np.cos(np.deg2rad(theta)) * rad, np.sin(np.deg2rad(theta)) * rad])
        elif task == 'traj':
            if eval_ins == True:
                engineer_goal = np.concatenate((mid_goal[ep%2], f_goal[ep%5]), axis=0)
            else:
                m_theta = np.random.sample() * 360
                m_rad = np.random.sample()
                engineer_goal = np.array([np.cos(np.deg2rad(m_theta)) * m_rad, np.sin(np.deg2rad(m_theta)) * m_rad, np.cos(np.deg2rad(theta)) * rad, np.sin(np.deg2rad(theta)) * rad])
        else:
            print('Error Task Type')
    return engineer_goal

def play_policy(policy, nb_obs, nb_timesteps, nb_act, nb_rew, env, controller, representer, knn):
    """
    Play a policy in the environment for a given number of timesteps, usin a NN controller.
    Then represent the trajectory and update the inverse model.
    """
    global n_traj
    task = args.task_type
    nb_pt = args.nb_pt
    env_id = args.env_id
    obs = np.zeros([1, nb_obs, nb_timesteps + 1])
    act = np.zeros([1, nb_act, nb_timesteps])
    rew = np.zeros([1, nb_rew, nb_timesteps + 1])
    obs.fill(np.nan)
    act.fill(np.nan)
    rew.fill(np.nan)
    env.reset()
    if env_id == 'Mass-point':
        if task == 'goal':
            obs[0, :, 0] = env.unwrapped.reset(np.concatenate(([0], np.zeros(4))), nb_timesteps)
        else:
            obs[0, :, 0] = env.unwrapped.reset(np.concatenate(([1], np.zeros(4))), nb_timesteps)
    else:
        if task == 'goal':
            obs[0, :, 0] = env.unwrapped.reset_model(np.concatenate(([0], np.zeros(4))), maxtimestep=nb_timesteps)
        else:
            obs[0, :, 0] = env.unwrapped.reset_model(np.concatenate(([1], np.zeros(4))), maxtimestep=nb_timesteps)
        
    rew[0, :, 0] = 0
    done = False  # termination signal
    max_timestep = False
    #env_timestep = 0
    plt_timestep = 0
    # Check Observation Range
    for t in range(nb_timesteps):
        
        #print('policy:' + str(policy))
        act[0, :, t] = controller.step(policy, obs[0, :, t]).reshape(1, -1)
        out = env.step(np.copy(act[0, :, t]))

        # env.render()
        obs[0, :, t + 1] = out[0]
        rew[0, :, t + 1] = out[1]
        done = out[2]
        if env_id == 'Mass-point': 
            plt_timestep = t+1
        else:
            info = out[3]
            plt_timestep = info['t']

        if done:
            break

    # convert the trajectory into a representation (=behavioral descriptor)
    rep, flag = representer.represent(obs[0,:,0:plt_timestep], act, task, nb_pt)
    if env_id == 'Mass-point':
        plt_obs = np.reshape(np.array(obs), (7,nb_timesteps+1))
    else:
        x, y = obs[:,0,:], obs[:,1,:]
        plt_obs = np.reshape(np.array([x,y]), (2, nb_timesteps+1))
    plt_obs = plt_obs.transpose()
        
    if task == 'goal':
        key = "_".join([str(plt_obs[plt_timestep,0]), str(plt_obs[plt_timestep,1])])
        train_obs[key] = np.array(plt_obs)
    else:
        for i in range(plt_timestep):
            key = "_".join([str(plt_obs[i,0]), str(plt_obs[i,1]), str(plt_obs[plt_timestep,0]), str(plt_obs[plt_timestep,1])])
            train_obs[key] = np.array(plt_obs)

    # update inverse model
    if task == 'traj':
        if flag:
            policy = np.array([policy] * rep.shape[0])
            knn.update(X=rep, Y=policy)
    else:
        knn.goal_update(X=rep, Y=policy)

    return obs, act, rew

def offline_evaluations(nb_eps, engineer_goal, knn, nb_rew, nb_timesteps, env, controller):
    """
    Play the best policy found in memory to test it. Play it for nb_eps episodes and average the returns.
    """
    # use scaled engineer goal and find policy of nearest neighbor in representation space
    global n_traj
    task = args.task_type
    nb_pt = args.nb_pt

    best_policy = knn.predict(engineer_goal)[0, :]

    # returns = []
    for i in range(nb_eps):
        rew = np.zeros([nb_rew, nb_timesteps + 1])
        rew.fill(np.nan)

        env.reset()
        if task == 'goal':
            configs = np.concatenate(([4], np.zeros(2), np.array(engineer_goal)),axis=0)
        else:
            configs = np.concatenate(([5], np.array(engineer_goal)),axis=0)
        if env_id == 'Mass-point':
            obs = env.unwrapped.reset(configs, nb_timesteps) 
        else:
            obs = env.unwrapped.reset_model(configs, nb_timesteps, isEval=True) 
        rew[:, 0] = 0
        done = False
        info = {}
        if env_id == 'Mass-point':
            plt_obs = [obs] # plot
        else:
            plt_obs = [[obs[0], obs[1]]]

        for t in range(nb_timesteps):
            if done: break
            
            act = controller.step(best_policy, obs)
            out = env.step(np.copy(act))
            obs = out[0].squeeze().astype(np.float)
            rew[:, t + 1] = out[1]
            done = out[2]
            info = out[3]
            
            if env_id == 'Mass-point':
                plt_obs.append(obs) # plot
            else:
                plt_obs.append([obs[0], obs[1]])

        plt_obs = np.array(plt_obs)
        # returns.append(np.nansum(rew))

        if task == 'goal':
            key = "_".join([str(engineer_goal[0]), str(engineer_goal[1]), str(n_traj)])
        else:
            key = "_".join([str(engineer_goal[0]), str(engineer_goal[1]), str(engineer_goal[2]), str(engineer_goal[3])])
        
        eval_obs[key] = np.array(plt_obs)
            
        n_traj += 1

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
    parser.add_argument('--trial', type=str, help='Folder name')
    parser.add_argument('--env_id', type=str, help='Environment ID', choices=['Mass-point', 'Reacher'], default='Mass-point')
    parser.add_argument('--task_type', type=str, help='Goal or Traj oriented task', choices=['goal', 'traj',] ,default='goal')
    parser.add_argument('--cus_noise', type=str, help='Value of noise in training', default='0.05')
    parser.add_argument('--nb_exploration', type=int, help='The number of explorations', default=1000)
    parser.add_argument('--nb_tests', type=int, help='Number of evaluation episodes', default=100)
    parser.add_argument('--nb_pt', type=int, help='Number of points', default=2)
    parser.add_argument('--saving_folder', type=str, help='Path of .pk file save', default='./outputs/')
    parser.add_argument('--save_plot', type=bool, help='To save figure or not', default=False)
    parser.add_argument('--save_pickle', type=bool, help='To save pickle or not', default=False)
    parser.add_argument('--eval_ins', type=bool, default=False)

    args = parser.parse_args()

    trial_id       = args.trial
    env_id         = args.env_id
    task           = args.task_type
    nb_pt          = args.nb_pt
    nb_exploration = args.nb_exploration
    saving_folder  = args.saving_folder
    save_plot      = args.save_plot
    eval_ins       = args.eval_ins

    policies = run_experiment(env_id, trial_id, nb_exploration, saving_folder, eval_ins)
   
    for key in train_obs:
        save_traj = train_obs[key][:,:2]
        obj_dict[key] = save_traj[~np.isnan(np.array(save_traj))].reshape(-1,2)

    eval_plot(save_plot, eval_obs, task)
