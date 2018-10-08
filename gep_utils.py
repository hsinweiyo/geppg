import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import gym
# from configs import cheetah_config

def create_data_path(saving_folder, env_id, trial_id):
    data_path = saving_folder + env_id + '/' + str(trial_id) + '/'
    if os.path.exists(data_path):
        # i = 1
        # while os.path.exists(saving_folder + env_id + '/' + str(trial_id + 100 * i) + '/'):
        #     i += 1
        # trial_id += i * 100
        print('result_path already exist, trial_id changed to: ', trial_id)
    data_path = saving_folder + env_id + '/' + str(trial_id) + '/'
    #os.mkdir(data_path)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    return data_path

def scale_vec(vector, initial_space):
    """
    Scale vector from initial space to [-1,1]^N
    """
    vec_in = np.copy(vector)
    vec_out = (vec_in - initial_space[:, 0]) * 2 / np.diff(initial_space).squeeze() - 1 

    return vec_out

def sample(space):
    """
    Uniform sampling of a vector from the input space.
    Space must be a 2d-array such that 1st column are the mins and 2nd the max for each dimension
    """
    vec_out = np.random.random(space.shape[0]) * np.diff(space).squeeze() + space[:, 0]

    return vec_out

def compute_metrics(gep_memory):
    metrics = []
    #
    obs = np.copy(gep_memory['observations'])
    act = np.copy(gep_memory['actions'])
    rew = np.copy(gep_memory['rewards'])
    rep = np.copy(gep_memory['representations'])
    final_perfs = np.copy(gep_memory['final_eval_perfs'])
    train_perfs = np.copy(gep_memory['train_perfs'])
    all_obs = np.zeros([obs.shape[0] * obs.shape[1], obs.shape[2]])
    all_act = np.zeros([act.shape[0] * act.shape[1], act.shape[2]])
    all_rew = np.zeros([rew.shape[0] * rew.shape[1], rew.shape[2]])
    all_rep = rep
    for i in range(obs.shape[2]):
        all_obs[:, i] = obs[:, :, i].ravel()
    for i in range(act.shape[2]):
        all_act[:, i] = act[:, :, i].ravel()
    for i in range(rew.shape[2]):
        all_rew[:, i] = rew[:, :, i].ravel()

    # size of the buffer
    size_buffer = obs.shape[0]
    metrics.append(size_buffer)
    # mean eval performance
    mean_eval_performance = final_perfs.mean()
    metrics.append(mean_eval_performance)
    # mean training performances
    mean_train_performance = train_perfs.mean()
    metrics.append(mean_train_performance)
    # std training performances
    std_train_performance = train_perfs.std()
    metrics.append(std_train_performance)
    # std obs
    normalized_std_obs = all_obs.std(axis=0) / (all_obs.max(axis=0) - all_obs.min(axis=0))
    normalized_std_obs = normalized_std_obs.sum()
    metrics.append(normalized_std_obs)
    # entropy per dimension (observations)
    entropies = []
    n_bins = 50
    for i in range(all_obs.shape[1]):
        non_nan = np.argwhere(~np.isnan(all_obs[:, i]))
        obs = all_obs[non_nan, i]
        assert np.isnan(obs).sum() == 0
        c_normalized = np.histogram(obs, n_bins)[0]
        c_normalized = c_normalized / np.float(c_normalized.sum())
        c_normalized = c_normalized[np.nonzero(c_normalized)]
        entropies.append(-sum(c_normalized * np.log(c_normalized)))
    metrics.append(np.array(entropies).mean())
    # coverage goal space
    all_rep = all_rep[np.argwhere(~np.isnan(all_rep[:, 0]))[:, 0], :]
    uni = 1 / (n_bins ** all_rep.shape[1])
    c = np.histogramdd(all_rep, n_bins)[0].ravel()
    c = c / np.float(c.sum())
    for i in range(c.shape[0]):
        if c[i] < uni:
            c[i] = 2 * c[i] - uni
    cov_metric = c.mean()
    metrics.append(cov_metric)
    # diversity score
    nn = NearestNeighbors(n_neighbors=4, algorithm='ball_tree', metric='euclidean')
    nn.fit(all_rep)
    dists = np.zeros([all_rep.shape[0]])
    for i in range(all_rep.shape[0]):
        dists_nn, tmp = nn.kneighbors(all_rep[i, :].reshape(1, -1))
        dists[i] = dists_nn[0, 1:].mean()
    metrics.append(dists.mean())

    return metrics

def replay_save_video(env_id, policy, path_vids):


    if env_id == 'HalfCheetah-v2':
        from configs import cheetah_config
        nb_bootstrap, nb_explorations, nb_tests, nb_timesteps, offline_eval, controller, representer, \
        nb_rep, engineer_goal, goal_space, initial_space, knn, noise, nb_weights = cheetah_config()
    elif env_id == 'MountainCarContinuous-v0':
        nb_bootstrap, nb_explorations, nb_tests, nb_timesteps, offline_eval, controller, representer, \
        nb_rep, engineer_goal, goal_space, initial_space, knn, noise, nb_weights = cmc_config()

    nb_timesteps = 100
    print('Timesteps: ' + str(nb_timesteps))
    env = gym.make(env_id)
    vid_env = VideoRecorder(env=env, path=path_vids)
    obs = env.reset()

    rew = np.zeros([nb_timesteps+1])
    done = False
    
    for t in range(nb_timesteps):
        if done:
            break
        print ('Len of controller.step in replay fun\n')
        print (len(controller.step(policy, obs)))
        act = controller.step(policy, obs).reshape(1, -1)
        out = env.step(np.copy(act))
        env.render()
        # vid_env.capture_frame()
        obs = out[0]
        rew[t + 1] = out[1]
        done = out[2]
    print('Run performance: ', np.nansum(rew))
    vid_env.close()

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

def mass_test_plot(flag, traj_dict, task):
    if flag:
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
                plt.plot(names[3], names[4], 'go')
                plt.plot(names[5], names[6], 'yo')
            # Paulolbear
            #fig.savefig('results/'+ key +'.png')
            fig.savefig('results02/'+ key +'.png')
            #plt.show()
            plt.close()
            
def eval_plot(flag, traj_dict, task):
    if flag:
        for key in traj_dict:
            fig = plt.figure()
            plt.axis([-1.0, 1.0, -1.0, 1.0])
        
            x_z = key.split('_')

            if task == 'goal':
                plt.plot(traj_dict[key][:,0], traj_dict[key][:,1], float(x_z[0]), float(x_z[1]), 'ro')
            else:
                plt.plot(traj_dict[key][:,0], traj_dict[key][:,1])
                plt.plot(float(x_z[0]), float(x_z[1]), 'bo')
                plt.plot(float(x_z[2]), float(x_z[3]), 'ro')
            fig.savefig('figures/'+ key +'.png')
            plt.close()

def write_file(knn, data_path):
    gep_memory = dict()
    gep_memory['representations'] = knn._X
    gep_memory['policies'] = knn._Y

    with open(data_path, 'wb') as f:
        pickle.dump(gep_memory, f)
