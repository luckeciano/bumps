#!/usr/bin/env python
from mpi4py import MPI
from baselines.common import boolean_flag, set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger

import tensorflow as tf
import sys

sys.path.insert(0, '../../..')
from core.soccer_env import SoccerEnv

def eval(env, model_dir):
    from baselines.ppo1 import mlp_policy

    # Load variables
    U.make_session(num_cpu=1).__enter__()
    ob_space = env.observation_space
    ac_space = env.action_space

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy

    # Load variables
    U.load_state(osp.join(model_dir, "model"))

    ob = env.reset()
    while True:
        # print ("Obs: ", ob)
        # print (type(ob))
        ac, vpred = pi.act(True, ob)

        ob, rew, new, _ = env.step(ac)

        if new:
            ob = env.reset()
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-dir", type=str, default=None,
                        required=True, help="load model from this directory. ")
    args = parser.parse_args()
    # logger.configure()

    env = SoccerEnv(0)
    eval(env, model_dir=args.model_dir)

if __name__ == '__main__':
    main()
