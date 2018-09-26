import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_units", type=int, help="number of hidden units in RAE")
parser.add_argument("--model_dir", help="where to restore and save the model")
parser.add_argument("--gpu_id", help="using which gpu")

args = parser.parse_args()

def mse(a, b):
    return np.mean(np.mean(np.power(a-b, 2)))

