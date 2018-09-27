import os
import sys
import csv
sys.path.append('./')
from eval_ablative.eval_ablative import eval_ablative
import gym
import custom_gym
import matplotlib.pyplot as plt
import pickle
import argparse
from controllers import NNController
from representers import KobukiRepresenter
from inverse_models import KNNRegressor
from gep_utils import *

traj_dict = dict()
avg_error = []
n_traj = 0
def run_testing(target, target_mid, engineer_goal, knn, obs, nb_rew, nb_timesteps, env, controller, n_neighbors):
    """
    Play the best policy found in memory to test it. Play it for nb_eps episodes and average the returns.
    """
    # use scaled engineer goal and find policy of nearest neighbor in representation space
    global n_traj
    #coord = [18, 54, 90, 126, 162]
    #np.array([np.cos(np.deg2rad(coord[n_traj])), np.sin(np.deg2rad(coord[n_traj]))])

    engineer_goal = np.array([engineer_goal] * n_neighbors)
    best_policy = knn.predict(engineer_goal)[0, :]
    returns = []

    rew = np.zeros([nb_rew, nb_timesteps + 1])
    rew.fill(np.nan)

    rew[:, 0] = 0
    done = False
    plt_obs = [obs] # plot
    nb_pt = args.nb_pt
        
    for t in range(nb_timesteps):
        if done: break
        
        act = controller.step(best_policy, obs)
        out = env.step(np.copy(act))
        obs = out[0].squeeze().astype(np.float)
        rew[:, t + 1] = out[1]
        done = out[2]
            
        plt_obs.append(obs) # plot

    returns.append(np.nansum(rew))

    if task == 'goal': 
        key = "_".join([str(n_traj), str(target)])
    else:
        key = "_".join([str(n_traj), str(target), str(target_mid)])
    
    #key = "_".join([str(engineer_goal[0]), str(engineer_goal[1]), str(engineer_goal[2]), str(engineer_goal[3]), str(n_traj), str(info['hit'])])
    traj_dict[key] = np.array(plt_obs)
    n_traj += 1

    #eval_perfs.append(np.array(returns).mean())
    # return obs[:2]
    nearest_pos = find_closest(target, target_mid, plt_obs)
    return nearest_pos

def testing_config():
    # run parameters
    nb_timesteps = 20
    nb_pt = args.nb_pt
    # controller parameters
    hidden_sizes = []
    controller_tmp = 1.
    activation = 'relu'
    subset_obs = range(4)
    norm_values = None

    scale = np.array([[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    controller = NNController(hidden_sizes, controller_tmp, subset_obs, 2, norm_values, scale, activation)

    # representer
    representer = KobukiRepresenter(nb_pt)
    
    # inverse model
    # knn = KNNRegressor(n_neighbors=1)

    return nb_timesteps, controller, representer

def target_position(target):
    target_pos = np.array([[0, .15], [-.1, .1], [-.2, 0], [-.1, -.1], [0, -.15], [.1, .1], [.1, -.1]]) / .21
    x, y = target_pos[int(target)]

    return x, y

def find_closest(target, target_mid, real_traj):
    #print('target in find closest: ', target)
    #print('target_mid in find closest: ', target_mid)
    if task == 'goal':
        pos = np.zeros(2)
        real_traj = np.array(real_traj)[:,:2]
        pos[0], pos[1] = target_position(target)
        ctft = np.argmin(np.linalg.norm(real_traj-pos, axis=1))
        #print('mid_pos, rpos:', real_traj[ctcp], ' ', mid_pos)
        #print('fpos, rpos:', real_traj[ctft], ' ', pos)
        #print('mdc, mdf: ', min_dist_cp, ' ', min_dist_ft)
        return real_traj[ctft]
    else:
        mid_pos = np.zeros(2)
        pos = np.zeros(2)
        real_traj = np.array(real_traj)[:,:2]
        mid_pos[0], mid_pos[1] = target_position(target_mid)
        pos[0], pos[1] = target_position(target)
        ctcp = np.argmin(np.linalg.norm(real_traj-mid_pos, axis=1))
        ctft = np.argmin(np.linalg.norm(real_traj[ctcp:]-pos, axis=1))
        #print('mid_pos, rpos:', real_traj[ctcp], ' ', mid_pos)
        #print('fpos, rpos:', real_traj[ctft], ' ', pos)
        min_dist_cp = np.linalg.norm(real_traj[ctcp]-mid_pos)
        min_dist_ft = np.linalg.norm(real_traj[ctft]-pos)
        #print('mdc, mdf: ', min_dist_cp, ' ', min_dist_ft)
        return np.concatenate((real_traj[ctcp], real_traj[ctft])) 


def plot_fig(traj_dict):
    for key in traj_dict:
        fig = plt.figure()
        plt.axis([-1.0, 1.0, -1.0, 1.0])
        names = key.split('_')
        target_x, target_y = target_position(names[1])
        plt.plot(traj_dict[key][:,0], traj_dict[key][:,1], target_x, target_y, 'ro')
        if task == 'traj':
            mid_x, mid_y = target_position(names[2])
            plt.plot(mid_x, mid_y, 'bo')
            
        fig.savefig('results/'+ key +'.png')
        plt.show()
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_neighbors', type=int, help='The number of k in nearest neighbor', default=1)
    parser.add_argument('--nb_eps', type=int, help='The number of episodes to evalutate', default=500)
    parser.add_argument('--nb_rew', type=int, help='The number of reward', default=1)
    parser.add_argument('--env_id', type=str, help='Environment ID', default='Kobuki-v0/')
    parser.add_argument('--trial_id', type=str, help='Trial ID, ends with 1', default='1/')
    parser.add_argument('--saving_folder', type=str, help='Path of .pk file save', default='./outputs/')
    parser.add_argument('--task', type=str, help='Goal or traj oriented', default='goal')
    parser.add_argument('--nb_pt', type=int, help='Number of points', default=2)
    parser.add_argument('--save_plot', type=bool, help='To save figure or not', default=False)
    parser.add_argument('--noise', type=str, help='Value of noise in training', default='0.01')
    parser.add_argument('--output', type=str, help='Output filename', default='reacher_goal.csv')
    parser.add_argument('--exp_folder', type=str, help='Folder name', default='reacher/')
    args = parser.parse_args()
    
    n_neighbors = args.n_neighbors
    nb_eps      = args.nb_eps
    nb_rew      = args.nb_rew
    env_id      = args.env_id
    trial_id    = args.trial_id
    saving_folder = args.saving_folder
    task        = args.task
    noise       = args.noise
    output      = args.output
    exp_folder  = args.exp_folder
    data_path   = (saving_folder + env_id + exp_folder + noise + '_' + trial_id + '_itr.pk')
    plot = args.save_plot
    print(data_path)

    if task == 'goal':
        env = gym.make('ReacherGEPTest-v0')
    else:
        env = gym.make('ReacherGEPTrajTest-v0')

    gep_memory = dict()
    with open(data_path, 'rb') as f:
        gep_memory = pickle.load(f)
    
    nb_timesteps, controller, representer = testing_config()
    
    knn = KNNRegressor(n_neighbors=int(args.n_neighbors))
    #eval_perfs = np.array(gep_memory['eval_perfs']).tolist()

    knn.init_update(gep_memory['representations'], gep_memory['policies'])

    for i in range(nb_eps):
        # target = np.random.randint(5)
        target = i % 5
        if task == 'goal':
            target_mid = -1
        else:
            target_mid = i%2
        env.reset()
        obs = env.unwrapped.reset_model(np.array([target_mid, target]))
        goal = eval_ablative(target)
        if task == 'goal':
            ideal_pos = [target_position(target)]
        else:
            ideal_pos = np.concatenate(([target_position(target)], [target_position(target_mid)])) 
        #last_pos = run_testing(target, target_mid, goal, knn, obs, nb_rew, nb_timesteps, env, controller)
        nearest_pos = run_testing(target, target_mid, goal, knn, obs, nb_rew, nb_timesteps, env, controller, n_neighbors)
        print ('Goal evaluated by dist_cem: ' + str(goal))
        print ('Real target position: ' + str(ideal_pos))
        #print ('Loss between dist_cem & real_target === ' + str(np.linalg.norm(goal - ideal_pos)))
        print ('Last agent position: x: ' + str(nearest_pos[0]) + ' y: ' + str(nearest_pos[1]))
        print ('L2norm === ' + str(np.linalg.norm(nearest_pos - ideal_pos)))
        if task == 'goal':
            avg_error.append(np.linalg.norm(nearest_pos - ideal_pos))
        else:
            avg_error.append(np.linalg.norm(nearest_pos[:2] - ideal_pos[:2]))
            avg_error.append(np.linalg.norm(nearest_pos[2:4] - ideal_pos[2:4]))
    print('Average error: ' + str (np.array(avg_error).mean()))
    
    total_timesteps = int(trial_id) * 20
    with open(output, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow([str(total_timesteps), args.noise, str(n_neighbors), str(np.array(avg_error).mean())])

    if plot:
        plot_fig(traj_dict)
