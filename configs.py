import numpy as np
import random

from controllers import NNController
from representers import ReacherRepresenter, MassPointRepresenter, KobukiRepresenter
from inverse_models import KNNRegressor

from gep_utils import *

def reacher_train_config(env_id, nb_pt, cus_noise, nb_act):
    # run parameters
    nb_bootstrap = 100
    nb_explorations = 100
    nb_tests = 100
    nb_timesteps = 50
    offline_eval = (1e6, 20)  # (x,y): y evaluation episodes every x (done offline)

    # controller parameters
    hidden_sizes = []
    controller_tmp = 1.
    activation = 'relu'
    subset_obs = range(4)
    norm_values = None
    scale = np.array([[-1.0, 1.0],[-1.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    controller = NNController(hidden_sizes, controller_tmp, subset_obs, nb_act, norm_values, scale, activation, env_id)
    nb_weights = controller.nb_weights

    # representer
    representer = ReacherRepresenter(nb_pt)
    initial_space = representer.initial_space
    goal_space = representer.initial_space  # space in which goal are sampled
    nb_rep = representer.dim
    engineer_goal = np.random.uniform(-1., 1., (nb_pt,))  # engineer goal
    # scale engineer goal to [-1,1]^N
    engineer_goal = scale_vec(engineer_goal, initial_space)
    # inverse model
    knn = KNNRegressor(n_neighbors=1)

    # exploration_noise
    noise = float(cus_noise)

    return nb_bootstrap, nb_explorations, nb_tests, nb_timesteps, offline_eval,  \
           controller, representer, nb_rep, engineer_goal, goal_space, initial_space, knn, noise, nb_weights

def mass_train_config(env_id, nb_pt, cus_noise, nb_act):
    # run parameters
    nb_bootstrap = 300
    nb_explorations = 100
    nb_tests = 100
    nb_timesteps = 50
    offline_eval = (1e6, 20)

    # controller parameters
    hidden_sizes = []
    controller_tmp = 1.
    activation = 'relu'
    #if task == 'goal':
    #    subset_obs = range(2)
    #    scale = np.array([[-1.0,1.0],[-1.0, 1.0]])
    #    engineer_goal = np.random.uniform(-1.0, 1.0, (2,))
    #else:
    subset_obs = range(7)
    scale = np.array([[-1.0,1.0], [-1.,1.], [0.,1.], [0.,1.], [0.,1.], [0.,1.], [0.,1.]])
    engineer_goal = np.random.uniform(-1.0, 1.0, (nb_pt,))
    norm_values = None
    #scale = np.vstack([np.array([[-1.0,1.0],]*2)])
    controller = NNController(hidden_sizes, controller_tmp, subset_obs, nb_act, norm_values, scale, activation, env_id)
    nb_weights = controller.nb_weights

    # representer
    representer = MassPointRepresenter(nb_pt)
    initial_space = representer.initial_space
    goal_space = representer.initial_space
    nb_rep = representer.dim
    
    # scale engineer goal to[-1, 1]^N
    engineer_goal = scale_vec(engineer_goal, initial_space)
    
    # inverse model
    knn = KNNRegressor(n_neighbors=1)

    # exploration_noise
    noise = float(cus_noise)

    return nb_bootstrap, nb_explorations, nb_tests, nb_timesteps, offline_eval, \
           controller, representer, nb_rep, engineer_goal, goal_space, initial_space, knn, noise, nb_weights

def kobuki_train_config(env_id, nb_pt, cus_noise, nb_act):
    # run parameters
    nb_bootstrap = 300
    nb_explorations = 100
    nb_tests = 100
    nb_timesteps = 50
    offline_eval = (1e6, 20)

    # controller parameters
    hidden_sizes = []
    controller_tmp = 1.
    activation = 'relu'
    #if task == 'goal':
    #    subset_obs = range(2)
    #    scale = np.array([[-1.0,1.0],[-1.0, 1.0]])
    #    engineer_goal = np.random.uniform(-1.0, 1.0, (2,))
    #else:
    subset_obs = range(7)
    scale = np.array([[-1.9,1.9], [-1.3,1.3], [0.,1.], [0.,1.], [0.,1.], [0.,1.], [0.,1.]])
    engineer_goal = np.zeros(2)
    engineer_goal[0] = np.random.uniform(-1.9, 1.9)
    engineer_goal[1] = np.random.uniform(-1.3, 1.3)
    norm_values = None
    #scale = np.vstack([np.array([[-1.0,1.0],]*2)])
    controller = NNController(hidden_sizes, controller_tmp, subset_obs, nb_act, norm_values, scale, activation, env_id)
    nb_weights = controller.nb_weights

    # representer
    representer = KobukiRepresenter(nb_pt)
    initial_space = representer.initial_space
    goal_space = representer.initial_space
    nb_rep = representer.dim
    
    # scale engineer goal to[-1, 1]^N
    engineer_goal = scale_vec(engineer_goal, initial_space)
    
    # inverse model
    knn = KNNRegressor(n_neighbors=1)

    # exploration_noise
    noise = float(cus_noise)

    return nb_bootstrap, nb_explorations, nb_tests, nb_timesteps, offline_eval, \
           controller, representer, nb_rep, engineer_goal, goal_space, initial_space, knn, noise, nb_weights

def mass_test_config(nb_pt, env_id, nb_act):
    # run parameters
    nb_timesteps = 50
    # controller parameters
    hidden_sizes = []
    controller_tmp = 1.
    activation = 'relu'
    subset_obs = range(7)
    norm_values = None

    scale = np.array([[-1.,1.], [-1.,1.], [0.,1.], [0.,1.], [0.,1.], [0.,1.], [0.,1.]])
    controller = NNController(hidden_sizes, controller_tmp, subset_obs, nb_act, norm_values, scale, activation, env_id)

    # representer
    representer = MassPointRepresenter(nb_pt)
    
    # inverse model
    #knn = KNNRegressor(n_neighbors=1)

    return nb_timesteps, controller, representer
    
def reacher_test_config(nb_pt, env_id, nb_act):
    # run parameters
    nb_timesteps = 20
    # controller parameters
    hidden_sizes = []
    controller_tmp = 1.
    activation = 'relu'
    subset_obs = range(4)
    norm_values = None

    scale = np.array([[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    controller = NNController(hidden_sizes, controller_tmp, subset_obs, nb_act, norm_values, scale, activation, env_id)

    # representer
    representer = ReacherRepresenter(nb_pt)
    
    # inverse model
    #knn = KNNRegressor(n_neighbors=1)

    return nb_timesteps, controller, representer
