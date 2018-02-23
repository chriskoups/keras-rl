import pytest
import random
import argparse

import numpy as np
from numpy.testing import assert_allclose
import gym
from gym.envs.debugging.two_round_deterministic_reward import TwoRoundDeterministicRewardEnv

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, TimeDistributed, LSTM, Reshape
from keras.optimizers import Adam
from rl.agents import DQNAgent, CEMAgent, SARSAAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import Memory, SequentialMemory, EpisodeParameterMemory, EpisodicMemory, PrioritizedEpisodicMemory, WindowedMemory
from rl.processors import MultiModeCartpole

def test_dqn():
    print 'Testing DQN on two round deterministic problem'
    env = TwoRoundDeterministicRewardEnv()
    np.random.seed(123)
    env.seed(123)
    random.seed(123)
    nb_actions = env.action_space.n
    
    # Next, we build a very simple model.
    model = Sequential()
    model.add(Dense(16, input_shape=(1,)))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    
    memory = Memory(limit=100)
    policy = EpsGreedyQPolicy(eps=.1)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50,
                   target_model_update=1e-1, policy=policy, enable_double_dqn=False)
    dqn.compile(Adam(lr=1e-3))
    
    dqn.fit(env, nb_steps=2000, visualize=False, verbose=0)
    policy.eps = 0.
    h = dqn.test(env, nb_episodes=20, visualize=False)
    assert_allclose(np.mean(h.history['episode_reward']), 3.)

    
    print 'Testing DQN on cartpole problem'
    env = gym.make('CartPole-v0')
    np.random.seed(123)
    env.seed(123)
    random.seed(123)
    nb_actions = env.action_space.n
    
    # Next, we build a very simple model.
    model = Sequential()
    model.add(Dense(16, input_shape=env.observation_space.shape, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    memory = Memory(limit=1000)
    policy = EpsGreedyQPolicy(eps=.1)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                   target_model_update=1e-1, policy=policy, enable_double_dqn=False, batch_size=64)
    dqn.compile(Adam(lr=1e-3))

    dqn.fit(env, nb_steps=20000, visualize=False, verbose=0)
    policy.eps = 0.
    h = dqn.test(env, nb_episodes=20, visualize=False)
    assert np.mean(h.history['episode_reward']) > 195.
    
def test_double_dqn():
    env = TwoRoundDeterministicRewardEnv()
    np.random.seed(123)
    env.seed(123)
    random.seed(123)
    nb_actions = env.action_space.n

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Dense(16, input_shape=(1,)))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    memory = Memory(limit=500)
    policy = EpsGreedyQPolicy(eps=.1)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50,
                   target_model_update=1e-1, policy=policy, enable_double_dqn=True)
    dqn.compile(Adam(lr=1e-3))

    dqn.fit(env, nb_steps=2000, visualize=False, verbose=0)
    policy.eps = 0.
    h = dqn.test(env, nb_episodes=20, visualize=False)
    assert_allclose(np.mean(h.history['episode_reward']), 3.)


def test_recurrent_dqn():
#     print 'Testing DQN on two round deterministic environment'
#     env = TwoRoundDeterministicRewardEnv()
#     np.random.seed(123)
#     env.seed(123)
#     random.seed(123)
#     nb_actions = env.action_space.n
#     batch_size = 1
#           
#     # Next, we build a very simple model.
#     policy_model = Sequential()
#     policy_model.add(LSTM(16, return_sequences=True, batch_input_shape=(1, None, 1,), stateful=True))
#     policy_model.add(TimeDistributed(Dense(nb_actions)))
#     policy_model.add(Activation('linear'))
#           
#     model = Sequential()
#     model.add(LSTM(16, return_sequences=True, batch_input_shape=(batch_size, None, 1,)))
#     model.add(TimeDistributed(Dense(nb_actions)))
#     model.add(Activation('linear'))
#           
#     memory = EpisodicMemory(limit=100)
#     policy = EpsGreedyQPolicy(eps=.1)
#     dqn = DQNAgent(model=model, policy_model=policy_model, nb_actions=nb_actions, memory=memory,
#                    nb_steps_warmup=50, target_model_update=1e-1, policy=policy, enable_double_dqn=False,
#                    batch_size=batch_size, train_interval=2)
#     dqn.compile(Adam(lr=1e-3))
#          
#     callbacks = [ResetStatesCallback()]
#     dqn.fit(env, nb_steps=2000, visualize=False, verbose=0,callbacks=callbacks)
#     policy.eps = 0.
#     h = dqn.test(env, nb_episodes=20, visualize=False)
#     assert_allclose(np.mean(h.history['episode_reward']), 3.)
# 
# 
#     print 'Testing DQN on cartpole problem with hidden LSTM and unrolling episodes'
#     env = gym.make('CartPole-v0')
#     np.random.seed(123)
#     env.seed(123)
#     random.seed(123)
#     nb_actions = env.action_space.n
#     batch_size = 5
#    
#     # Next, we build a very simple model.
#     policy_model = Sequential()
#     policy_model.add(TimeDistributed(Dense(16, activation='relu'), batch_input_shape=(1,None,) + env.observation_space.shape))
#     policy_model.add(LSTM(16, stateful=True, return_sequences=True, activation='softsign'))
#     policy_model.add(TimeDistributed(Dense(nb_actions, activation='linear')))
#       
#     model = Sequential()
#     model.add(TimeDistributed(Dense(16, activation='relu'), batch_input_shape=(batch_size,None,) + env.observation_space.shape))
#     model.add(LSTM(16, stateful=True, return_sequences=True, activation='softsign'))
#     model.add(TimeDistributed(Dense(nb_actions, activation='linear')))
#        
#     print(model.summary())
#        
#     memory = EpisodicMemory(limit=100)
#        
#     policy = EpsGreedyQPolicy(eps=.1)
#     dqn = DQNAgent(model=model, policy_model=policy_model, nb_actions=nb_actions, memory=memory,
#                    nb_steps_warmup=1000, target_model_update=1e-1, policy=policy, enable_double_dqn=False,
#                    batch_size=batch_size, train_interval=2, nb_max_steps_recurrent_unrolling=10)
#     dqn.compile(Adam(lr=1e-3))
#    
#     dqn.fit(env, nb_steps=20000, visualize=True, verbose=2)
#     policy.eps = 0.
#     h = dqn.test(env, nb_episodes=20, visualize=True)
#     print np.mean(h.history['episode_reward'])
#     assert np.mean(h.history['episode_reward']) > 195.
#      
#      
#     print 'Testing DQN on cartpole problem with hidden LSTM'
#     env = gym.make('CartPole-v0')
#     np.random.seed(123)
#     env.seed(123)
#     random.seed(123)
#     nb_actions = env.action_space.n
#     batch_size = 5
#    
#     # Next, we build a very simple model.
#     policy_model = Sequential()
#     policy_model.add(TimeDistributed(Dense(16, activation='relu'),
#                                       batch_input_shape=(1,None,) + env.observation_space.shape))
#     policy_model.add(LSTM(16, stateful=True, return_sequences=True, activation='softsign'))
#     policy_model.add(TimeDistributed(Dense(nb_actions, activation='linear')))
#       
#     model = Sequential()
#     model.add(TimeDistributed(Dense(16, activation='relu'),
#                                batch_input_shape=(batch_size,None,) + env.observation_space.shape))
#     model.add(LSTM(16, return_sequences=True, activation='softsign'))
#     model.add(TimeDistributed(Dense(nb_actions, activation='linear')))
#        
#     print(model.summary())
#        
#     memory = EpisodicMemory(limit=100)
#        
#     policy = EpsGreedyQPolicy(eps=.1)
#     dqn = DQNAgent(model=model, policy_model=policy_model, nb_actions=nb_actions, memory=memory,
#                    nb_steps_warmup=1000, target_model_update=1e-1, policy=policy,
#                    enable_double_dqn=False, batch_size=batch_size, train_interval=2)
#     dqn.compile(Adam(lr=1e-3))
#    
#     dqn.fit(env, nb_steps=20000, visualize=False, verbose=2)
#     policy.eps = 0.
#     h = dqn.test(env, nb_episodes=20, visualize=True)
#     print np.mean(h.history['episode_reward'])
#     assert np.mean(h.history['episode_reward']) > 195.
#     
#     
#     print 'Testing DQN on cartpole problem with input LSTM'
#     env = gym.make('CartPole-v0')
#     np.random.seed(123)
#     env.seed(123)
#     random.seed(123)
#     nb_actions = env.action_space.n
#     batch_size = 5
# 
#     # Next, we build a very simple model.
#     policy_model = Sequential()
#     policy_model.add(LSTM(16, return_sequences=True, 
#                           batch_input_shape=(1, None,) + env.observation_space.shape, 
#                           stateful=True, activation='softsign'))
#     policy_model.add(TimeDistributed(Dense(16, activation='relu')))
#     policy_model.add(TimeDistributed(Dense(nb_actions, activation='linear')))
#     
#     model = Sequential()
#     model.add(LSTM(16, return_sequences=True, 
#                    batch_input_shape=(batch_size, None,) + env.observation_space.shape, 
#                    stateful=True, activation='softsign'))
#     model.add(TimeDistributed(Dense(16, activation='relu')))
#     model.add(TimeDistributed(Dense(nb_actions, activation='linear')))
# 
#     print(model.summary())
#     
#     memory = EpisodicMemory(limit=100)
#     
#     policy = EpsGreedyQPolicy(eps=.1)
#     dqn = DQNAgent(model=model, policy_model=policy_model, nb_actions=nb_actions, memory=memory,
#                    nb_steps_warmup=1000, target_model_update=1e-1, policy=policy, 
#                    enable_double_dqn=False, batch_size=batch_size, train_interval=2,
#                    nb_max_steps_recurrent_unrolling=10)
#     dqn.compile(Adam(lr=1e-3))
# 
#     dqn.fit(env, nb_steps=20000, visualize=False, verbose=2)
#     policy.eps = 0.
#     h = dqn.test(env, nb_episodes=20, visualize=True)
#     print np.mean(h.history['episode_reward'])
#     assert np.mean(h.history['episode_reward']) > 195.
    
    print 'Testing DQN on cartpole problem with hidden LSTM and unrolling episodes'
    env = gym.make('CartPole-v0')
    np.random.seed(123)
    env.seed(123)
    random.seed(123)
    nb_actions = env.action_space.n
    batch_size = 32
    
    # Next, we build a very simple model.
    policy_model = Sequential()
    policy_model.add(TimeDistributed(Dense(16, activation='relu'), batch_input_shape=(1,None,3)))
    policy_model.add(LSTM(16, stateful=True, return_sequences=True, activation='softsign'))
    policy_model.add(TimeDistributed(Dense(nb_actions, activation='linear')))
       
    model = Sequential()
    model.add(TimeDistributed(Dense(16, activation='relu'), batch_input_shape=(batch_size,None,3)))
    model.add(LSTM(16, stateful=True, return_sequences=True, activation='softsign'))
    model.add(TimeDistributed(Dense(nb_actions, activation='linear')))
        
    print(model.summary())
    
    processor = MultiModeCartpole()
    memory = EpisodicMemory(limit=200)
    policy = EpsGreedyQPolicy(eps=.1)
    dqn = DQNAgent(model=model, policy_model=policy_model, nb_actions=nb_actions, memory=memory,
                   nb_steps_warmup=500, target_model_update=1e-1, policy=policy, 
                   enable_double_dqn=False, batch_size=batch_size, train_interval=1,
                   nb_max_steps_recurrent_unrolling=10, processor=processor)
    dqn.compile(Adam(lr=1e-3))
    
    dqn.fit(env, nb_steps=40000, visualize=True, verbose=2)
    policy.eps = 0.
    h = dqn.test(env, nb_episodes=20, visualize=True)
    print np.mean(h.history['episode_reward'])
    assert np.mean(h.history['episode_reward']) > 195.


    print 'Testing DQN on multi-mode cartpole problem'
    env = gym.make('CartPole-v0')
    np.random.seed(123)
    env.seed(123)
    random.seed(123)
    nb_actions = env.action_space.n
    batch_size = 5
    
    # Next, we build a very simple model.
    policy_model = Sequential()
    policy_model.add(LSTM(32, return_sequences=True, 
                          batch_input_shape=(1, None,3), 
                          stateful=True, activation='softsign'))
    policy_model.add(TimeDistributed(Dense(32, activation='relu')))
    policy_model.add(TimeDistributed(Dense(nb_actions, activation='linear')))
     
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, 
                   batch_input_shape=(batch_size, None,3), 
                   stateful=True, activation='softsign'))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(nb_actions, activation='linear')))

    print(model.summary())
    
    processor = MultiModeCartpole()
    memory = EpisodicMemory(limit=200)
    policy = EpsGreedyQPolicy(eps=.1)
    dqn = DQNAgent(model=model, policy_model=policy_model, nb_actions=nb_actions, memory=memory,
                   nb_steps_warmup=1000, target_model_update=1e-1, policy=policy, 
                   enable_double_dqn=False, batch_size=batch_size, train_interval=2,
                   nb_max_steps_recurrent_unrolling=10, processor=processor)
    dqn.compile(Adam(lr=1e-3))

    dqn.fit(env, nb_steps=30000, visualize=True, verbose=2)
    policy.eps = 0.
    h = dqn.test(env, nb_episodes=20, visualize=True)
    print np.mean(h.history['episode_reward'])
    assert np.mean(h.history['episode_reward']) > 195.
    

def test_cem():
    env = TwoRoundDeterministicRewardEnv()
    np.random.seed(123)
    env.seed(123)
    random.seed(123)
    nb_actions = env.action_space.n

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Dense(16, input_shape=(1,)))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    memory = EpisodeParameterMemory(limit=1000, window_length=1)
    dqn = CEMAgent(model=model, nb_actions=nb_actions, memory=memory)
    dqn.compile()

    dqn.fit(env, nb_steps=2000, visualize=False, verbose=1)
    h = dqn.test(env, nb_episodes=20, visualize=False)
    assert_allclose(np.mean(h.history['episode_reward']), 3.)


def test_duel_dqn():
    env = TwoRoundDeterministicRewardEnv()
    np.random.seed(123)
    env.seed(123)
    random.seed(123)
    nb_actions = env.action_space.n

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Dense(16, input_shape=(1,)))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions, activation='linear'))

    memory = SequentialMemory(limit=1000, window_length=1)
    policy = EpsGreedyQPolicy(eps=.1)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50,
                   target_model_update=1e-1, policy=policy, enable_double_dqn=False, enable_dueling_network=True)
    dqn.compile(Adam(lr=1e-3))

    dqn.fit(env, nb_steps=2000, visualize=False, verbose=0)
    policy.eps = 0.
    h = dqn.test(env, nb_episodes=20, visualize=False)
    assert_allclose(np.mean(h.history['episode_reward']), 3.)


def test_sarsa():
    env = TwoRoundDeterministicRewardEnv()
    np.random.seed(123)
    env.seed(123)
    random.seed(123)
    nb_actions = env.action_space.n

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Dense(16, input_shape=(1,)))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions, activation='linear'))

    policy = EpsGreedyQPolicy(eps=.1)
    sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=50, policy=policy)
    sarsa.compile(Adam(lr=1e-3))

    sarsa.fit(env, nb_steps=20000, visualize=False, verbose=0)
    policy.eps = 0.
    h = sarsa.test(env, nb_episodes=20, visualize=False)
    assert_allclose(np.mean(h.history['episode_reward']), 3.)

if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', choices=['dqn', 'double_dqn', 'recurrent_dqn', 'cem', 'sarsa'], default=None)
    args = parser.parse_args()
    
    if args.test == None:
        pytest.main([__file__])
    elif args.test == 'dqn':
        test_dqn()
    elif args.test == 'double_dqn':
        test_double_dqn()
    elif args.test == 'recurrent_dqn':
        test_recurrent_dqn()
    elif args.test == 'cem':
        test_cem()
    elif args.test == 'sarsa':
        test_sarsa()
