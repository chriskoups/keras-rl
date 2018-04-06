import pytest
import random
import argparse

import numpy as np
from numpy.testing import assert_allclose
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import NAFAgent, DDPGAgent
from rl.random import OrnsteinUhlenbeckProcess, GaussianWhiteNoiseProcess
from rl.memory import SequentialMemory, EpisodicMemory, Memory

from gym.envs.debugging.two_round_deterministic_reward import TwoRoundDeterministicRewardEnv
from rl.processors import ContinuousToDiscreteActions

def test_cdqn():
    # TODO: replace this with a simpler environment where we can actually test if it finds a solution
    env = gym.make('Pendulum-v0')
    np.random.seed(123)
    env.seed(123)
    random.seed(123)
    nb_actions = env.action_space.shape[0]

    V_model = Sequential()
    V_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    V_model.add(Dense(16))
    V_model.add(Activation('relu'))
    V_model.add(Dense(1))

    mu_model = Sequential()
    mu_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    mu_model.add(Dense(16))
    mu_model.add(Activation('relu'))
    mu_model.add(Dense(nb_actions))

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    x = Concatenate()([action_input, Flatten()(observation_input)])
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = Dense(((nb_actions * nb_actions + nb_actions) // 2))(x)
    L_model = Model(inputs=[action_input, observation_input], outputs=x)

    memory = SequentialMemory(limit=1000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
    agent = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                     memory=memory, nb_steps_warmup=50, random_process=random_process,
                     gamma=.99, target_model_update=1e-3)
    agent.compile(Adam(lr=1e-3))

    agent.fit(env, nb_steps=400, visualize=False, verbose=0, nb_max_episode_steps=100)
    h = agent.test(env, nb_episodes=2, visualize=False, nb_max_episode_steps=100)
    # TODO: evaluate history

def test_ddpg_simple():
    env = TwoRoundDeterministicRewardEnv()
    np.random.seed(123)
    env.seed(123)
    random.seed(123)
    nb_actions = 1

    actor = Sequential()
    actor.add(Dense(16, input_shape=(1,)))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,), name='observation_input')
    x = Concatenate()([action_input, observation_input])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
 
    processor = ContinuousToDiscreteActions(min_action=0, max_action=1)
 
    memory = SequentialMemory(limit=500, window_length=1)
    random_process = GaussianWhiteNoiseProcess(mu=0., sigma=0.75, sigma_min=0.1, n_steps_annealing=1500, seed=123)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=32, nb_steps_warmup_actor=32,
                      random_process=random_process, gamma=.95, target_model_update=1e-2, processor=processor)
    agent.compile([Adam(lr=0.001), Adam(lr=0.005)], metrics=['mape'])
 
    agent.fit(env, nb_steps=1500, visualize=False, verbose=1)
    processor.max_action = None
    h = agent.test(env, nb_episodes=20, visualize=False)
    assert_allclose(np.mean(h.history['episode_reward']), 3.)
 

def test_ddpg():
    print 'Testing DDPG on Continuous Pendulum problem'
    env = gym.make('Pendulum-v0')
    np.random.seed(123)
    env.seed(123)
    random.seed(123)

    nb_actions = env.action_space.shape[0]

    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(32, activation='relu'))
    actor.add(Dense(32, activation='relu'))
    actor.add(Dense(32, activation='relu'))
    actor.add(Dense(nb_actions, activation='linear'))

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense( 1, activation='linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)

    memory = SequentialMemory(limit=2500)
    sigma = np.abs((env.action_space.high - env.action_space.low)) / 2.
    random_process = OrnsteinUhlenbeckProcess(0.15, mu=0., sigma=sigma, sigma_min=sigma*0.1, n_steps_annealing=10000, seed=123)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=32, nb_steps_warmup_actor=32,
                      random_process=random_process, gamma=.95, target_model_update=1e-2)
    agent.compile([Adam(lr=0.001), Adam(lr=0.005)], metrics=['mape'])

    agent.fit(env, nb_steps=10000, visualize=False, verbose=2)
    h = agent.test(env, nb_episodes=10, visualize=True)

    print 'Average score over {} trails is: '.format(len(h.history['episode_reward'])) + str(np.mean(h.history['episode_reward']))
    assert np.mean(h.history['episode_reward']) > -175.

if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', choices=['cdqn', 'ddpg'], default=None)
    args = parser.parse_args()

    if args.test == None:
        pytest.main([__file__])
    elif args.test == 'cdqn':
        test_cdqn()
    elif args.test == 'ddpg':
        #test_ddpg_simple()
        test_ddpg()
