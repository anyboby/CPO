import numpy as np
import tensorflow as tf

from cpo_rl.model.fc import FC
from cpo_rl.model.bnn import BNN

def construct_model(obs_dim=11, add_dim = 0, act_dim=3, rew_dim=1, hidden_dim=200, num_networks=7, num_elites=5, session=None, mse_weights = 1):
	print('[ BNN ] Observation dim {} | Action dim: {} | Hidden dim: {}'.format(obs_dim, act_dim, hidden_dim))
	params = {'name': 'BNN', 'num_networks': num_networks, 'num_elites': num_elites, 'sess': session, 'mse_weights':mse_weights}
	model = BNN(params)

	model.add(FC(hidden_dim, input_dim=obs_dim+act_dim+add_dim, activation="swish", weight_decay=0.000025))	#0.000025))
	model.add(FC(hidden_dim, activation="swish", weight_decay=0.00005))			#0.00005))
	#model.add(FC(hidden_dim, activation="swish", weight_decay=0.00005))		#@anyboby optional
	#model.add(FC(hidden_dim, activation="swish", weight_decay=0.00005))			#@anyboby optional
	model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))		#0.000075))
	model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))		#0.000075))
	model.add(FC(obs_dim+rew_dim, weight_decay=0.0001))							#0.0001
	model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.0015})
	return model

def format_samples_for_training(samples, obs_stack_size):
	"""
	formats samples to fit training, specifically returns: 

	inputs, outputs:

	inputs = np.concatenate((observations, act), axis=-1)
	outputs = np.concatenate((rewards, delta_observations), axis=-1)

	"""
	obs = samples['observations']
	act = samples['actions']
	next_obs = samples['next_observations']
	rew = samples['rewards']
	delta_obs = next_obs - obs
	inputs = np.concatenate((obs, act), axis=-1)
	outputs = np.concatenate((delta_obs[:,:-obs_stack_size], rew), axis=-1)
	return inputs, outputs

#def reset_model(model):
#	model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model.name)
#	model.sess.run(tf.initialize_vars(model_vars))

if __name__ == '__main__':
	model = construct_model()
