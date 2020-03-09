import numpy as np
import tensorflow as tf
import pdb

class FakeEnv:

    def __init__(self, model, config,task, stacks = 1):
        self.model = model
        self.config = config
        self.stacks = stacks
        self.goal_size = 0.3
        self.task = task
        self.reward_distance = 1
        self.reward_clip = 10
        self.reward_goal = 1

    '''
        x : [ batch_size, obs_dim + 1 ]
        means : [ num_models, batch_size, obs_dim + 1 ]
        vars : [ num_models, batch_size, obs_dim + 1 ]
    '''
    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1/2 * (k * np.log(2*np.pi) + np.log(variances).sum(-1) + (np.power(x-means, 2)/variances).sum(-1))
        
        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means,0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, deterministic=False):
        assert len(obs.shape) == len(act.shape)
        obs_depth = len(obs.shape)
        if obs_depth == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        
        unstacked_obs_size = int(obs.shape[1]/self.stacks)               ### e.g. if a stacked obs is 88 with 4 stacks,
                                                                                            ### unstacking it yields 22


        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)       #### self.model outputs whole ensembles outputs
        ensemble_model_means[:,:,:-1] += obs[:,-unstacked_obs_size:]                                #### models output state change rather than state completely
        ensemble_model_stds = np.sqrt(ensemble_model_vars)                                          #### std = sqrt(variance)

        ### directly use means, if deterministic
        if deterministic:
            ensemble_samples = ensemble_model_means                     
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        #### choose one model from ensemble randomly
        num_models, batch_size, _ = ensemble_model_means.shape
        model_inds = self.model.random_inds(batch_size)
        batch_inds = np.arange(0, batch_size)
        samples = ensemble_samples[model_inds, batch_inds]
        model_means = ensemble_model_means[model_inds, batch_inds]
        model_stds = ensemble_model_stds[model_inds, batch_inds]
        ####

        log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)


        #### retrieve r and done for new state
        rewards, next_obs, dist_goal, last_dist_goal = samples[:,-1:], samples[:,:-1], samples[:,-2:-1], obs[:,-1]
        #### stack previous obs with newly predicted obs
        if self.stacks > 1:
            next_obs = np.concatenate((obs, next_obs), axis=-(obs_depth))
            next_obs = np.delete(next_obs, slice(unstacked_obs_size), -(obs_depth))

        terminals = self.config.termination_fn(obs, act, next_obs)

        batch_size = model_means.shape[0]
        return_means = np.concatenate((model_means[:,-1:], terminals, model_means[:,:-1]), axis=-1)
        return_stds = np.concatenate((model_stds[:,-1:], np.zeros((batch_size,1)), model_stds[:,:-1]), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]
            dist_goal = dist_goal[0]
            dist_goal = dist_goal[0]
            last_dist_goal = last_dist_goal[0]

        info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}

        rewards = self.reward(dist_goal, last_dist_goal)
        # Goal processing
        if self.goal_met(dist_goal):
            info['goal_met'] = True
            rewards += self.reward_goal
            terminals = True

        return next_obs, rewards, terminals, info

    ## for debugging computation graph
    def step_ph(self, obs_ph, act_ph, deterministic=False):
        assert len(obs_ph.shape) == len(act_ph.shape)

        inputs = tf.concat([obs_ph, act_ph], axis=1)
        # inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.create_prediction_tensors(inputs, factored=True)
        # ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        ensemble_model_means = tf.concat([ensemble_model_means[:,:,-1:], ensemble_model_means[:,:,:-1] + obs_ph[None]], axis=-1)
        # ensemble_model_means[:,:,1:] += obs_ph
        ensemble_model_stds = tf.sqrt(ensemble_model_vars)
        # ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            # ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds
            ensemble_samples = ensemble_model_means + tf.random.normal(tf.shape(ensemble_model_means)) * ensemble_model_stds

        samples = ensemble_samples[0]

        rewards, next_obs = samples[:,-1:], samples[:,:-1]
        terminals = self.config.termination_ph_fn(obs_ph, act_ph, next_obs)
        info = {}

        return next_obs, rewards, terminals, info

    def close(self):
        pass

    def goal_met(self, dist_goal):
        ''' Return true if the current goal is met this step '''
        if 'goal' in self.task:
            return dist_goal <= self.goal_size
        # if self.task == 'push':
        #     return self.dist_box_goal() <= self.goal_size
        # if self.task == 'button':
        #     for contact in self.data.contact[:self.data.ncon]:
        #         geom_ids = [contact.geom1, contact.geom2]
        #         geom_names = sorted([self.model.geom_id2name(g) for g in geom_ids])
        #         if any(n == f'button{self.goal_button}' for n in geom_names):
        #             if any(n in self.robot.geom_names for n in geom_names):
        #                 return True
        #     return False
        if self.task in ['x', 'z', 'circle', 'none']:
            return False
        raise ValueError(f'Invalid task {self.task}')

    def reward(self, dist_goal, last_dist_goal):
        ''' Calculate the dense component of reward.  Call exactly once per step '''
        reward = 0.0
        # Distance from robot to goal
        if 'goal' in self.task or 'button' in self.task:
            reward += (last_dist_goal - dist_goal) * self.reward_distance
        # # Distance from robot to box
        # if self.task == 'push':
        #     dist_box = self.dist_box()
        #     gate_dist_box_reward = (self.last_dist_box > self.box_null_dist * self.box_size)
        #     reward += (self.last_dist_box - dist_box) * self.reward_box_dist * gate_dist_box_reward
        #     self.last_dist_box = dist_box
        # # Distance from box to goal
        # if self.task == 'push':
        #     dist_box_goal = self.dist_box_goal()
        #     reward += (self.last_box_goal - dist_box_goal) * self.reward_box_goal
        #     self.last_box_goal = dist_box_goal
        # # Used for forward locomotion tests
        # if self.task == 'x':
        #     robot_com = self.world.robot_com()
        #     reward += (robot_com[0] - self.last_robot_com[0]) * self.reward_x
        #     self.last_robot_com = robot_com
        # # Used for jump up tests
        # if self.task == 'z':
        #     robot_com = self.world.robot_com()
        #     reward += (robot_com[2] - self.last_robot_com[2]) * self.reward_z
        #     self.last_robot_com = robot_com
        # # Circle environment reward
        # if self.task == 'circle':
        #     robot_com = self.world.robot_com()
        #     robot_vel = self.world.robot_vel()
        #     x, y, _ = robot_com
        #     u, v, _ = robot_vel
        #     radius = np.sqrt(x**2 + y**2)
        #     reward += (((-u*y + v*x)/radius)/(1 + np.abs(radius - self.circle_radius))) * self.reward_circle
        # # Intrinsic reward for uprightness
        # if self.reward_orientation:
        #     zalign = quat2zalign(self.data.get_body_xquat(self.reward_orientation_body))
        #     reward += self.reward_orientation_scale * zalign
        # Clip reward
        if self.reward_clip:
            in_range = reward < self.reward_clip and reward > -self.reward_clip
            if not(in_range):
                reward = np.clip(reward, -self.reward_clip, self.reward_clip)
                print('Warning: reward was outside of range!')
        return reward
