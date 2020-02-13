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

#from gym.utils import play

import numpy as np
import pandas
import random

import pygame
from pygame.locals import K_UP
from pygame.locals import K_DOWN
from pygame.locals import K_LEFT
from pygame.locals import K_RIGHT


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
    
    
    env_name = 'Safexp-'+robot+task+'-v0'
    env = gym.make(env_name)
    #gym.utils.play.play(env, zoom=3)
    actions = [
                [0.8,0.5],
                [0.8,-0.5],
                [0.8,0.0],
                [-0.8,0.5],
                [-0.8,-0.5],
                [-0.8,0.0],
    ]
    
    clock = pygame.time.Clock()
    manual_control = True
    render = True
    step_on_key = True

    ##### model construction #####
    _model_train_freq = 1000
    _rollout_freq = 1000

    obs_dim = np.prod(env.observation_space.shape)
    act_dim = np.prod(env.action_space.shape)

    hidden_dim = 340
    num_networks = 7
    num_elites = 5

    mse_weights = np.ones(shape=(obs_dim+1), dtype='float32')*100
    mse_weights[0:3] = 0.0001
    mse_weights[19:22] = 0.001
 
    sensors_to_stack = 3
    stacks_p_sensor = 4
    add_obs = 3*sensors_to_stack*(stacks_p_sensor-1)
    sensor_stack = {
        'acc':[],
        'gyro':[],
        'velo':[],
        #'action':[]
    }
    sensor_indcs = {
        'acc' : 3,
        'gyro' : 22,
        'velo' : 60,
        #'action':0,
    }
    assert len(sensor_stack)!=sensors_to_stack or len(sensor_indcs)!=sensor_stack


    model = construct_model(obs_dim=obs_dim, add_dim=add_obs, act_dim=act_dim, hidden_dim=hidden_dim, num_networks=num_networks, num_elites=num_elites, mse_weights=mse_weights)
    #pool = np.zeros(shape=(_model_train_freq, obs_dim+act_dim))
    pool = {
        'actions':np.zeros(shape=(_model_train_freq-stacks_p_sensor-1,act_dim), dtype='float32'),
        'next_observations':np.zeros(shape=(_model_train_freq-stacks_p_sensor-1, obs_dim+add_obs), dtype='float32'),
        'observations':np.zeros(shape=(_model_train_freq-stacks_p_sensor-1, obs_dim+add_obs), dtype='float32'),
        'rewards':np.zeros(shape=(_model_train_freq-stacks_p_sensor-1, 1), dtype='float32'),
        'terminals':np.zeros(shape=(_model_train_freq-stacks_p_sensor-1, 1), dtype='bool'),
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


    try:
        obs = env.reset()
        if render: rendered=env.render( mode='human')
        print(obs)
        #video_size=[rendered.shape[1],rendered.shape[0]]
        video_size = [100,100]
        screen = pygame.display.set_mode(video_size)

        a = 0
        print("-----Safety Gym Environment is running-----")
        y = 0
        episode = 0
        episodes_successful = 0
        frame = 0
        total_timesteps = 0
        frames_per_episode = 1000
        ready_to_pool = False

        while True:

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
                                
                action = [a, s,]
            else:
                action = [0,0]

            
            action = np.array(actions[random.randint(0,len(actions)-1)])

            # action = actions[2]
            obs_old = obs
            obs, reward, done, info = env.step(action)
            ### stack gyro and accelerometer data

            if render: env.render()
            delta_obs = obs-obs_old

            ## stack sensor
            if sensors_to_stack>0:
                if len(list(sensor_stack.values())[0])>=stacks_p_sensor:
                    for k in sensor_stack:
                        sensor_stack[k].pop(0)
                    ready_to_pool = True
                for k in sensor_stack:
                    sensor_stack[k].append(obs[sensor_indcs[k]-3:sensor_indcs[k]])

            ## store samples
            if ready_to_pool:
                obs_old_s = obs_old
                obs_next_s = obs
                for k in sensor_stack:
                    stack = np.array(sensor_stack[k][:-1]).flatten()
                    stack_next = np.array(sensor_stack[k][1:]).flatten()
                    obs_old_s = np.concatenate((obs_old_s, stack), axis=0)
                    obs_next_s  = np.concatenate((obs_next_s, stack_next), axis=0)
                    
                pool['actions'][total_timesteps-stacks_p_sensor-1] = action
                pool['next_observations'][total_timesteps-stacks_p_sensor-1] = obs_next_s
                pool['observations'][total_timesteps-stacks_p_sensor-1] = obs_old_s
                pool['rewards'][total_timesteps-stacks_p_sensor-1] = reward
                pool['terminals'][total_timesteps-stacks_p_sensor-1] = done


            frame += 1
            total_timesteps += 1
            clock.tick(300)

            ##### train model #####
            if total_timesteps % _model_train_freq == 0:
                total_timesteps = 0
                #model.reset()        #@anyboby debug
                
                
                if big_pool:
                    for key in pool:
                        big_pool[key] = np.concatenate((big_pool[key], pool[key]), axis=0)
                else:
                    for key in pool:
                        big_pool[key] = np.copy(pool[key])

                #### train the model with input:(obs, act), outputs: (rew, delta_obs), inputs are divided into sets with holdout_ratio
                model_train_metrics = _train_model(model, big_pool, batch_size=512, max_epochs=None, holdout_ratio=0.2, max_grad_updates=500, obs_stack_size=add_obs)

            if total_timesteps % _rollout_freq == 0:
                # ensemble_model_means, ensemble_model_vars = model.predict(inputs, factored=True)       #### self.model outputs whole ensembles outputs
                # ensemble_model_means[:,:,1:] += obs                                                         #### models output state change rather than state completely
                # ensemble_model_stds = np.sqrt(ensemble_model_vars)                                          #### std = sqrt(variance)
                # ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

                # #### choose one model from ensemble randomly
                # num_models, batch_size, _ = ensemble_model_means.shape
                # model_inds = model.random_inds(batch_size)
                # batch_inds = np.arange(0, batch_size)
                # samples = ensemble_samples[model_inds, batch_inds]
                # model_means = ensemble_model_means[model_inds, batch_inds]
                # model_stds = ensemble_model_stds[model_inds, batch_inds]
                # ####

                # rewards, pred_obs = samples[:,:1], samples[:,1:]
                # terminals = static_fns.termination_fn(obs, action, pred_obs)
                f_next_obs = env.reset()
                for _ in range(0,5):
                    f_next_obs, f_rew, f_term, f_info = fake_env.step(f_next_obs, action, deterministic=True)
                    r_next_obs, r_rew, r_term, r_info = env.step(action)
                    error = f_next_obs-r_next_obs

                    ### test: replace only sensor info for rollout
                    # f_next_obs[0:3]  = r_next_obs[0:3]
                    #f_next_obs[19:22]  = r_next_obs[0:3]
                    #f_next_obs[0:3]  = r_next_obs[0:3]
                    #f_next_obs[0:3]  = r_next_obs[0:3]

                    with np.printoptions(precision=3, suppress=True):
                        print("______________________________________________________")
                        print("action: ", action)
                        # print(f_next_obs[0:3])
                        # print(r_next_obs[0:3])
                        # print(f_next_obs[3:19])
                        # print(r_next_obs[3:19])
                        # print(f_next_obs[19:22])
                        # print(r_next_obs[19:22])
                        # print(f_next_obs[22:38])
                        # print(r_next_obs[22:38])
                        # print(f_next_obs[38:41])
                        # print(r_next_obs[38:41])
                        # print(f_next_obs[41:57])
                        # print(r_next_obs[41:57])
                        # print(f_next_obs[57:60])
                        # print(r_next_obs[57:60])
                        
                        print(error[0:3])
                        print(error[3:19])
                        print(error[19:22])
                        print(error[22:38])
                        print(error[38:41])
                        print(error[41:57])
                        print(error[57:60])

                        print(f_rew-r_rew)
                        print("reward: ", reward)
                        print(info)


            if frame == frames_per_episode:
                print ("Success! 1000 frames reached! ending episode")
                episodes_successful+=1
                done = True
                ready_to_pool = False
            if done:
                frame = 0
                episode +=1
                print ("episode {} done".format(episode))
                env.reset()

    finally:
        env.close()
        print("-----Safety Gym Environment is closed-----")


def _train_model(model, pool, obs_stack_size=0, **kwargs):

    #### format samples to fit: inputs: concatenate(obs,act), outputs: concatenate(rew, delta_obs)
    train_inputs, train_outputs = format_samples_for_training(pool, obs_stack_size)
    model_metrics = model.train(train_inputs, train_outputs, **kwargs)
    return model_metrics


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