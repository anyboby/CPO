#!/usr/bin/env python
import gym 
from gym.spaces import Box, Dict, Discrete
import safety_gym
import cpo_rl
from cpo_rl.utils.mpi_tools import mpi_fork
from cpo_rl.utils.run_utils import setup_logger_kwargs
from cpo_rl.pg.cpo_algo import cpo

from cpo_rl.model.constructor import construct_model, format_samples_for_training
from cpo_rl.model.fake_env import FakeEnv
from cpo_rl.model.perturbed_env import PerturbedEnv
from safety_preprocessed_wrapper import SafetyPreprocessedEnv
#from gym.utils import play

import numpy as np
import pandas
import random
import matplotlib.pyplot as plt
from copy import deepcopy
import time 

import pygame
from pygame.locals import K_UP
from pygame.locals import K_DOWN
from pygame.locals import K_LEFT
from pygame.locals import K_RIGHT

from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.vec_env import DummyVecEnv




def main(robot, task, algo, seed, exp_name, cpu):

    # Verify experiment
    robot_list = ['point', 'car', 'doggo']
    task_list = ['goal0','goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']
    algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']

    algo = algo.lower()
    task = task.capitalize()
    robot = robot.capitalize()
    assert algo in algo_list, "Invalid algo"
    assert task.lower() in task_list, "Invalid task"
    assert robot.lower() in robot_list, "Invalid robot"

    # Hyperparameters
    exp_name = algo + '_' + robot + task
    if robot=='Doggo':
        num_steps = 1e8
        steps_per_epoch = 60000
    else:
        num_steps = 3e7 #1e7
        steps_per_epoch = 100000 #def 30000
    
    stacks = 4 
    env_name = 'Safexp-'+robot+task+'-v0'
    env = gym.make(env_name)
    env = SafetyPreprocessedEnv(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=stacks)
    #env = VecFrameStack(env, 4)
    
    p_env = gym.make(env_name)
    #gym.utils.play.play(env, zoom=3)
    actions = [
                [0.8,0.4],
                [0.5,-0.6],
                [0.3,0.0],
                [-0.1,0.8],
                [-0.3,-0.1],
                [-0.6,0.0],
    ]
    
    clock = pygame.time.Clock()
    manual_control = False
    render = True
    step_on_key = True

    ##### model construction #####
    _model_train_freq = 5000
    _rollout_freq = 70000

    obs_dim = np.prod(env.observation_space.shape)
    act_dim = np.prod(env.action_space.shape)

    hidden_dim = 340 #280
    num_networks = 7
    num_elites = 5
 
    sensor_mse_weights = {
        # 'acc': 0.0000,
        # 'gyro': 0.000,
        # # 'goal_lidar':1,
        # # 'hazards_lidar':1,
        # # 'vases_lidar':1,
        # 'velo' : 1,
        #'action': 1, # this one doesn't matter (comes from action not obs)
    }

    mse_weights = np.ones(shape=(int(obs_dim/stacks)+1), dtype='float32')*100
    #mse_weights[3:19]=10
    #mse_weights = mse_weights*100

    #### act_dim +=3 because of additional spike info and last action
    model = construct_model(obs_dim=obs_dim, 
        act_dim=act_dim+3, 
        hidden_dim=hidden_dim, 
        num_networks=num_networks, 
        num_elites=num_elites, 
        mse_weights=mse_weights, 
        stacks=stacks,
        )
    pool = {
        'actions':np.zeros(shape=(_model_train_freq, act_dim+3), dtype='float32'),
        'action_processed':np.zeros(shape=(_model_train_freq, 1), dtype='float32'),
        'next_observations':np.zeros(shape=(_model_train_freq, obs_dim), dtype='float32'),
        'observations':np.zeros(shape=(_model_train_freq, obs_dim), dtype='float32'),
        'rewards':np.zeros(shape=(_model_train_freq, 1), dtype='float32'),
        'terminals':np.zeros(shape=(_model_train_freq, 1), dtype='bool'),
        'sim_states': np.empty(shape=(_model_train_freq,), dtype=object),
    }
    big_pool = {}

    ## fake env
    class StaticFns:

        @staticmethod
        def termination_fn(obs, act, next_obs):
            '''
            safeexp-pointgoal (like the other default safety-gym envs) doesn't terminate
            prematurely other than due to sampling errors etc., therefore just return Falses
            '''
            #assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            done = np.array([False]).repeat(len(obs))
            done = done[:,None]
            return done

    static_fns = StaticFns           # termination functions for the envs (model can't simulate those)
    fake_env = FakeEnv(model, static_fns)
    perturbed_env = PerturbedEnv(p_env, std_inc=0)#0.03)

    try:
        obs = env.reset()
        obs = np.squeeze(obs)

        action = np.array([0,0])
        #init_sim_state = env.get_sim_state()
        #p_env.reset(init_sim_state)
        if render: rendered=env.render( mode='human')

        video_size = [100,100]
        screen = pygame.display.set_mode(video_size)

        print("-----Safety Gym Environment is running-----")
        episode = 0
        episodes_successful = 0
        frame = 0
        total_timesteps = 0
        frames_per_episode = 1000
        ready_to_pool = False
        model_train_metrics = None
        rollouts=0

        while True:
            action_old = action
            if manual_control:

                milliseconds = clock.get_time()
                for event in pygame.event.get():
                    keys = pygame.key.get_pressed()
                    if keys[K_UP]:
                        a = 1.0
                    elif keys[K_DOWN]:
                        a = -1.0
                    else:
                        a = 0     

                    if  keys[K_LEFT]:
                        s = 0.7
                    elif keys[K_RIGHT]:
                        s = -0.7
                    else:
                        s = 0.0
                                
                action = np.array([a, s,])
            else:
                if total_timesteps%15==0:
                    action = np.array(actions[random.randint(0,len(actions)-1)])
            
            #### process action for spike prediction
            processed_act = process_act(action[0], action_old[0])
            processed_act = np.concatenate((action, action_old, [processed_act]))

            obs_old = obs
            obs, reward, done, info  = env.step(action)
            obs = np.squeeze(obs)

            if model_train_metrics:
                if rollouts == 0:
                    f_next_obs, f_rew, f_term, f_info = fake_env.step(obs, processed_act, deterministic=True)
                else:
                    f_next_obs, f_rew, f_term, f_info = fake_env.step(f_next_obs, processed_act, deterministic=True)
                if rollouts == 10:
                    print('restarting rollouts at 0')
                    rollouts = 0
                else: 
                    rollouts += 1
                print('predicted obs: \n', f_next_obs[-22:-6])
                print(f_next_obs[-6:-3])
                print(f_next_obs[-3:])

            #sim_state = env.get_sim_state()
            if render: env.render()
            delta_obs = obs-obs_old

            ## store samples
            obs_old_s = obs_old
            obs_next_s = obs

            print('real obs: \n', obs[-22:-6])
            print(obs[-6:-3])
            print(obs[-3:])
            print('---------------------------')

            pool['actions'][total_timesteps] = processed_act
            pool['action_processed'][total_timesteps] = processed_act[4]
            pool['next_observations'][total_timesteps] = obs_next_s
            pool['observations'][total_timesteps] = obs_old_s
            pool['rewards'][total_timesteps] = reward
            pool['terminals'][total_timesteps] = done
            #pool['sim_states'][total_timesteps] = sim_state

            frame += 1
            total_timesteps += 1
            clock.tick(300)

            ##### train model #####
            if total_timesteps % _model_train_freq == 0:
                total_timesteps = 0
                
                
                if big_pool:
                    for key in pool:
                        big_pool[key] = np.concatenate((big_pool[key], pool[key]), axis=0)
                else:
                    for key in pool:
                        big_pool[key] = np.copy(pool[key])

                #### train the model with input:(obs, act), outputs: (rew, delta_obs), inputs are divided into sets with holdout_ratio
                #model.reset()        #@anyboby debug
                model_train_metrics = _train_model(model, big_pool, stacks=stacks, batch_size=512, max_epochs=None, holdout_ratio=0.2)

            if total_timesteps % _rollout_freq == 0 and model_train_metrics and not done:
                for i in range(0,5):
                    roll_start_obs = pool['next_observations'][total_timesteps-1,:]
                    #roll_start_sim = pool['sim_states'][total_timesteps-1]

                    error_check = np.sum(roll_start_obs-obs)

                    #cur_f = base_next_obs
                    # cur_f = perturbed_env.reset(sim_state=roll_start_sim)
                    # reset_check = np.sum(cur_f-roll_start_obs)
                    cur_r = obs
                    cur_f = cur_r

                    print(f'sum (roll_start_obs-obs: \n {error_check}')
                    # print(f'sum (cur_f - roll_start_obs): \n {reset_check}')

                    for i in range(0,10):
                        rand_action = np.array(actions[random.randint(0,len(actions)-1)])
                        if i%3==0:
                            rand_action = np.array(actions[random.randint(0,len(actions)-1)])
                        
                        processed_act = process_act(rand_action[0], old_action[0])
                        processed_act = np.concatenate((rand_action, old_action, [processed_act]))
                        f_next_obs, f_rew, f_term, f_info = fake_env.step(cur_f, processed_act, deterministic=True)
                        #f_next_obs, f_rew, f_term, f_info = perturbed_env.step(rand_action)
                        r_next_obs, r_rew, r_term, r_info = env.step(rand_action)
                        
                        delta_f = f_next_obs-cur_f
                        delta_r = r_next_obs-cur_r
                        cur_f = f_next_obs
                        cur_r = r_next_obs

                        error = f_next_obs-r_next_obs
                        action_old = action

                        with np.printoptions(precision=3, suppress=True):
                            print("______________________________________________________")
                            print("action: ", action)
                            
                            print(error[0:3])
                            print(delta_f[0:3])
                            print(delta_r[0:3])
                            print('---')
                            print(error[3:19])
                            print(delta_f[3:19])
                            print(delta_r[3:19])
                            print('---')
                            print(error[19:22])
                            print(delta_f[19:22])
                            print(delta_r[19:22])
                            print('---')
                            print(error[22:38])
                            print(delta_f[22:38])
                            print(delta_r[22:38])
                            print('---')
                            print(error[38:41])
                            print(delta_f[38:41])
                            print(delta_r[38:41])
                            print('---')
                            print(error[41:57])
                            print(delta_f[41:57])
                            print(delta_r[41:57])
                            print('---')
                            print(error[57:60])
                            print(delta_f[57:60])
                            print(delta_r[57:60])
                            print('---')

                            print(f_rew-r_rew)
                            print("reward: ", reward)
                            print(f'sum error obs diff: \n {np.sum(error)}')
                            print(info)
                        
                        if f_term or r_term:
                            break


            if frame == frames_per_episode:
                print ("Success! 1000 frames reached! ending episode")
                episodes_successful+=1
                done = True
                ready_to_pool = False
            if done:
                
                # plt.plot(np.arange(100), pool['observations'][0:100,55], color='blue')                         # filtered y-vel
                #plt.plot(np.arange(100), env.obs_replay_vy_filt[0:100], color='red' )
                # plt.plot(np.arange(100), pool['observations'][0:100, 1], color='purple' )

                # plt.plot(np.arange(100), pool['actions'][0:100, 0], color='blue' )
                # plt.plot(np.arange(100), pool['action_processed'][0:100, 0], color='red' )
                # plt.plot(np.arange(100), pool['observations'][0:100, 0], color='purple' )

                #plt.plot(np.arange(100), env.obs_replay_acc_y_filt[0:100], color='orange' )
                #plt.plot(np.arange(100), np.array(env.obs_replay_acc_y_real[0:100])/100, color='green' )

                #plt.plot(np.arange(100), env.obs_replay_acc_y_real[400:500], color='green' )

                plt.show()
                frame = 0
                episode +=1
                print ("episode {} done".format(episode))
                #env.reset(init_sim_state)
                env.reset()

    finally:
        env.close()
        print("-----Safety Gym Environment is closed-----")


def _train_model(model, pool,stacks=1, **kwargs):

    #### format samples to fit: inputs: concatenate(obs,act), outputs: concatenate(rew, delta_obs)
    
    train_inputs, train_outputs = format_samples_for_training(pool, stacks=stacks)
    model_metrics = model.train(train_inputs, train_outputs, **kwargs)
    return model_metrics

def process_act(act, last_act):
    '''
    Predicts a spike based on 0-transition between actions
    !! very specifically designed for x-acceleration spike detection
    returns a normalized prediction signal for y-acceleration in mujoco envs
    a shape (1,) np array
    '''
    act_x = act
    last_act_x = last_act
    acc_spike = 0
    ### acc
    if last_act_x==act_x:
        acc_spike=0
    else:
        if last_act_x<=0<=act_x or act_x<=0<=last_act_x:
            #pass
            acc_spike = act_x-last_act_x
            acc_spike = acc_spike/abs(acc_spike) #normalize
    return acc_spike


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='Point')
    parser.add_argument('--task', type=str, default='Goal0')
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--cpu', type=int, default=1)
    args = parser.parse_args()
    exp_name = args.exp_name if not(args.exp_name=='') else None
    main(args.robot, args.task, args.algo, args.seed, exp_name, args.cpu)
