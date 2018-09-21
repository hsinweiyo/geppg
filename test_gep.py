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

traj_dict = dict()
avg_error = []
def run_testing(target, mid_target, engineer_goal, knn, obs, nb_rew, nb_timesteps, env, controller, eval_perfs):
    """
    Play the best policy found in memory to test it. Play it for nb_eps episodes and average the returns.
    """
    # use scaled engineer goal and find policy of nearest neighbor in representation space
    global n_traj
    #coord = [18, 54, 90, 126, 162]
    #np.array([np.cos(np.deg2rad(coord[n_traj])), np.sin(np.deg2rad(coord[n_traj]))])
    best_policy = knn.predict(engineer_goal)[0, :]

    returns = []

    rew = np.zeros([nb_rew, nb_timesteps + 1])
    rew.fill(np.nan)

    task = args.task
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
        key = "_".join([str(n_traj), str(target), str(mid_target)])
        
    #key = "_".join([str(engineer_goal[0]), str(engineer_goal[1]), str(engineer_goal[2]), str(engineer_goal[3]), str(n_traj), str(info['hit'])])
    traj_dict[key] = np.array(plt_obs)
    n_traj += 1

    eval_perfs.append(np.array(returns).mean())
    return obs[:2]

def testing_config():
    # run parameters
    nb_timesteps = 200
    nb_pt = args.nb_pt
    # controller parameters
    hidden_sizes = []
    controller_tmp = 1.
    activation = 'relu'
    subset_obs = range(7)
    norm_values = None

    scale = np.array([[-1.,1.], [-1.,1.], [0.,1.], [0.,1.], [0.,1.], [0.,1.], [0.,1.]])
    controller = NNController(hidden_sizes, controller_tmp, subset_obs, 2, norm_values, scale, activation)

    # representer
    representer = KobukiRepresenter(nb_pt)
    
    # inverse model
    knn = KNNRegressor(n_neighbors=1)

    return nb_timesteps, controller, representer, knn

def target_position(target, mid_target, task):
    target_angel = range(18, 180, 36)
    target_rad = np.deg2rad(target_angel[int(target)])
    x, y = np.cos(target_rad), np.sin(target_rad)

    if task == 'goal':
        return x, y
    else:
        mid_angle = [0, 180]
        mid_rad = np.deg2rad(mid_angle[int(mid_target)-5])
        mid_x, mid_y = 0.25 * np.cos(mid_rad), 0.25 * np.sin(mid_rad) 
        return mid_x, mid_y, x, y

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
    args = parser.parse_args()
    
    n_neighbors = args.n_neighbors
    nb_eps      = args.nb_eps
    nb_rew      = args.nb_rew
    env_id      = args.env_id
    trial_id    = args.trial_id
    task        = args.task
    saving_folder = args.saving_folder
    data_path   = (saving_folder + env_id + trial_id + 'save_gep.pk')
    print(data_path)

    gep_memory = dict()
    with open(data_path, 'rb') as f:
        gep_memory = pickle.load(f)
    
    knn = KNNRegressor(n_neighbors)
    eval_perfs = np.array(gep_memory['eval_perfs']).tolist()

    nb_timesteps, controller, representer, knn, = testing_config()

    knn.init_update(gep_memory['representations'], gep_memory['policies'])

    env = gym.make('FiveTargetEnv-v1')
    
    if task == 'goal': task_id = 2
    else: task_id = 3

    for i in range(nb_eps):
        target = np.random.randint(5)
        target_mid = np.random.randint(1) + 5
        env.reset()
        obs = env.unwrapped.reset(np.array([task_id, target_mid, target, 0., 0.]))
        # target = np.where(obs[2:] == 1)[0][0]
        goal = eval_dist_cem(target)
        if task == 'goal':
            x, y = target_position(target, target_mid, task)
            ideal_pos = [x,y]
        else:
            mid_x, mid_y, x, y = target_position(target, target_mid, task)
            ideal_pos = [mid_x, mid_y, x, y]

        last_pos = run_testing(target, target_mid, goal, knn, obs, nb_rew, nb_timesteps, env, controller, eval_perfs)
        #print ('Goal evaluated by dist_cem: ' + str(goal))
        #print ('Real target position: x: ' + str(x) + ' y: ' + str(y))
        #print ('Last agent position: x: ' + str(last_pos[0]) + ' y: ' + str(last_pos[1]))
        avg_error.append(np.linalg.norm(last_pos - ideal_pos))

    print('Average error: ' + str (np.array(avg_error).mean()))
    
    for key in traj_dict:
        fig = plt.figure()
        plt.axis([-1.0, 1.0, -1.0, 1.0])
        names = key.split('_')
        if task == 'goal':
            target_x, target_y = target_position(names[1], 0, task)
            plt.plot(traj_dict[key][:,0], traj_dict[key][:,1], target_x, target_y, 'ro')
        else:
            mid_x, mid_y, target_x, target_y = target_position(names[1], names[2], task)
            plt.plot(traj_dict[key][:,0], traj_dict[key][:,1], target_x, target_y, 'ro')
            plt.plot(mid_x, mid_y, 'bo')
        fig.savefig('results/'+ key +'.png')
        plt.show()
        plt.close()
