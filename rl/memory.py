from __future__ import absolute_import
from collections import deque, namedtuple
import warnings
import random

import numpy as np


# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')

Timesteps = namedtuple('Timesteps', 'states, actions, rewards, terminals')


def sample_batch_indexes(low, high, size):
    if high - low >= size:
        # We have enough data. Draw without replacement, that is each index is unique in the
        # batch. We cannot use `np.random.choice` here because it is horribly inefficient as
        # the memory grows. See https://github.com/numpy/numpy/issues/2764 for a discussion.
        # `random.sample` does the same thing (drawing without replacement) and is way faster.
        try:
            r = xrange(low, high)
        except NameError:
            r = range(low, high)
        batch_idxs = random.sample(r, size)
    else:
        # Not enough data. Help ourselves with sampling from the range, but the same index
        # can occur multiple times. This is not good and should be avoided by picking a
        # large enough warm-up phase.
        warnings.warn('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs

def shape_from_object(object):
    if hasattr(object, 'shape'):
        return object.shape
    #elif hasattr(object, '__iter__'):
    #    return (objectshape_from_object(object[0])
    #    return out.shape()
    else:
        return ()


class Memory(object):
    """ Memory.
    
    First-In-First-Out (FIFO) data storage structure. Input data is appended
    sequentially to a buffer array, once the buffer is filled, new input data
    overwrites the oldest data first.
    
    # Arguments
        limit: int > 0 Maximum size of the memory buffer
    """
    def __init__(self, limit):
        if limit < 1:
            limit = 1

        self.limit = limit
            
        self.states = deque(maxlen=limit)
        self.actions = deque(maxlen=limit)
        self.rewards = deque(maxlen=limit)
        self.terminals = deque(maxlen=limit)

    def sample(self, batch_size, batch_idxs=[]):
        # Ensure requested batch ids are of correct size
        while len(batch_idxs) < batch_size:
            batch_idxs = sample_batch_indexes(0, len(self.states) - 1, size=batch_size)
        assert len(batch_idxs) == batch_size

        # Ensure all requested indexes are within current memory range
        idx = 0
        while batch_idxs[idx] > len(self.states) - 1:
            batch_idxs = sample_batch_indexes(0, self.state.size - 1, size=batch_size)
            
        batch_idxs = np.array(batch_idxs)
        
        # check if multiple inputs, if so we have to restructure the batch of observations
        # todo replace append with set, initialise states_batch with correct size and just assign values
        if isinstance(self.states[0], (list, tuple)):
            states_batch = list(self.states[batch_idxs[0]])
            states1_batch = list(self.states[batch_idxs[0]+1])
            for idx in batch_idxs[1:]:
                for n, obs in enumerate(self.states[idx]):
                    states_batch[n] = np.append(states_batch[n], obs, axis=0)
                for n, obs in enumerate(self.states[idx+1]):
                    states1_batch[n] = np.append(states1_batch[n], obs, axis=0)
            
            assert len(states_batch[0]) == batch_size
            assert len(states1_batch[0]) == batch_size
        else:
            states_batch = np.array(self.states)[batch_idxs]
            states1_batch = np.array(self.states)[batch_idxs+1]
            
            assert len(states_batch) == batch_size
            assert len(states1_batch) == batch_size
            
        experiences = []
        for idx in batch_idxs:
            experiences.append(Experience(state0=self.states[idx], action=self.actions[idx], reward=self.rewards[idx],
                                          state1=self.states[idx+1], terminal1=self.terminals[idx]))
            
        assert len(experiences) == batch_size

        return experiences, states_batch, np.array(self.actions)[batch_idxs], np.array(self.rewards)[batch_idxs], states1_batch, np.array(self.terminals)[batch_idxs]

    def append(self, observation, action, reward, terminal, training=True):
        if training:
            self.states.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)

    def get_recent_state(self, current_observation):
        state = current_observation
        # add batch dimention
        # state = state.reshape((1,) + state.shape)

        return state

    def get_config(self):
        config = {
            'limit': self.limit,
        }
        return config


class WindowedMemory(Memory):
    """ WindowedMemory
    
    First-In-First-Out (FIFO) data storage structure. Input data is appended
    sequentially to a buffer array, once the buffer is filled, new input data
    overwrites the oldest data first.
    
    # Arguments
        limit: int > 0 Maximum size of the memory buffer
    """
    def __init__(self, limit, window_length=1, ignore_episode_boundaries=False):
        super(WindowedMemory, self).__init__(limit)
        self.window_length = window_length
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_states = None
        self.recent_terminals = deque([False]*window_length,maxlen=window_length)

    def get_recent_state(self, current_observation):
        if self.recent_states == None:
            # bootstrap the running state
            self.recent_states = deque(np.zeros((self.window_length,) + shape_from_object(current_observation)), maxlen=self.window_length)
        
        state = np.zeros((self.window_length,) + shape_from_object(current_observation))
        state[-1] = current_observation
        for idx in range(self.window_length-1, 0, -1):
            if not self.ignore_episode_boundaries and self.recent_terminals[idx-1]:
                #found terminal so leave the rest of the array as zeros
                break
            
            state[idx-1] = self.recent_states[idx]

        assert len(state) is self.window_length
        
        # add batch dimention
        state = state.reshape((1,) + state.shape)
        
        return state
    
    def sample(self, batch_size, batch_idxs=None):
        # It is not possible to tell whether the first state in the memory is terminal, because it
        # would require access to the "terminal" flag associated to the previous state. As a result
        # we will never return this first state (only using `self.terminals[0]` to know whether the
        # second state is terminal).
        # In addition we need enough entries to fill the desired window length.
        assert self.nb_entries >= self.window_length + 2, 'not enough entries in the memory'

        if batch_idxs is None:
            # Draw random indexes such that we have enough entries before each index to fill the
            # desired window length.
            batch_idxs = sample_batch_indexes(self.window_length, self.nb_entries - 1, size=batch_size)
            
        # ensure that none of the batches starts with a terminal condition
        # replace any that do
        # this may result in the batches to start at the same index
        i = 0
        for idx in batch_idxs:
            while(not self.ignore_episode_boundaries and self.terminals[idx-1]):
                idx = sample_batch_indexes(self.window_length, self.nb_entries - 1, size=batch_size)[0]
            batch_idxs[i] = idx
            i = i + 1
                
        assert np.min(batch_idxs) >= self.window_length
        assert np.max(batch_idxs) < self.nb_entries - 1
        assert len(batch_idxs) == batch_size
        
        # Create experiences
        state0_batch = np.zeros((batch_size, self.window_length,) + shape_from_object(self.states[0]))
        action_batch = np.zeros((batch_size, self.window_length), dtype=type(self.actions[0]))
        reward_batch = np.zeros((batch_size, self.window_length))
        state1_batch = np.zeros((batch_size, self.window_length,) + shape_from_object(self.states[0]))
        terminal_batch = np.zeros((batch_size, self.window_length), dtype=bool)
        
        experiences = []
        batch = 0
        for idx in batch_idxs:
            state0 = np.zeros((self.window_length,) + shape_from_object(self.states[0]))
            state_idx = self.window_length - 1
            for offset in range(idx, idx - self.window_length, -1):
                if not self.ignore_episode_boundaries and self.terminals[offset-1]:
                    #found terminal so leave the rest of the array as zeros
                    break
                state0_batch[batch][state_idx] = self.states[offset]
                action_batch[batch][state_idx] = self.actions[offset]
                reward_batch[batch][state_idx] = self.rewards[offset]
                terminal_batch[batch][state_idx] = self.terminals[offset]
                
                state_idx = state_idx - 1
            
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.states[idx+1])
            state1 = np.array(state1)
            assert len(state1) == self.window_length
            
            state1_batch[batch] = state1
                                                                                            
            experiences.append(Experience(state0=state0, action=self.actions[idx], reward=self.rewards[idx],
                                          state1=state1, terminal1=self.terminals[idx]))
            batch = batch + 1
        assert len(experiences) == batch_size
        return experiences, state0_batch, action_batch, reward_batch, state1_batch, terminal_batch

    def append(self, observation, action, reward, terminal, training=True):
        if self.recent_states == None:
            self.recent_states = deque(np.zeros((self.window_length,) + shape_from_object(observation)), maxlen=self.window_length)
        
        self.recent_states.append(observation)
        self.recent_terminals.append(terminal)
        
        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            self.states.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)

    @property
    def nb_entries(self):
        return len(self.states)

    def get_config(self):
        config = super(SequentialMemory, self).get_config()
        config = {
            'window_length': self.window_length,
            'ignore_episode_boundaries': self.ignore_episode_boundaries,
            'limit': self.limit,
        }
        return config

    @property
    def is_episodic(self):
        return False


class SequentialMemory(WindowedMemory):
    """ SequentialMemory
    
    First-In-First-Out (FIFO) data storage structure. Input data is appended
    sequentially to a buffer array, once the buffer is filled, new input data
    overwrites the oldest data first.
    
    # Arguments
        limit: int > 0 Maximum size of the memory buffer
    """
    def sample(self, batch_size, batch_idxs=None):
        # It is not possible to tell whether the first state in the memory is terminal, because it
        # would require access to the "terminal" flag associated to the previous state. As a result
        # we will never return this first state (only using `self.terminals[0]` to know whether the
        # second state is terminal).
        # In addition we need enough entries to fill the desired window length.
        assert self.nb_entries >= self.window_length + 2, 'not enough entries in the memory'

        if batch_idxs is None:
            # Draw random indexes such that we have enough entries before each index to fill the
            # desired window length.
            batch_idxs = sample_batch_indexes(self.window_length, self.nb_entries - 1, size=batch_size)
            
        # ensure that none of the batches starts with a terminal condition
        # replace any that do
        # this may result in the batches to start at the same index
        i = 0
        for idx in batch_idxs:
            while(not self.ignore_episode_boundaries and self.terminals[idx-1]):
                idx = sample_batch_indexes(self.window_length, self.nb_entries - 1, size=batch_size)[0]
            batch_idxs[i] = idx
            i = i + 1
                
        assert np.min(batch_idxs) >= self.window_length
        assert np.max(batch_idxs) < self.nb_entries - 1
        assert len(batch_idxs) == batch_size
        
        # Create experiences
        state0_batch = np.zeros((batch_size, self.window_length,) + shape_from_object(self.states[0]))
        action_batch = np.zeros((batch_size,), dtype=type(self.actions[0]))
        reward_batch = np.zeros((batch_size,))
        state1_batch = np.zeros((batch_size, self.window_length,) + shape_from_object(self.states[0]))
        terminal_batch = np.zeros((batch_size,), dtype=bool)
        
        experiences = []
        batch = 0
        for idx in batch_idxs:
            state0 = np.zeros((self.window_length,) + shape_from_object(self.states[0]))
            state_idx = self.window_length - 1
            for offset in range(idx, idx - self.window_length, -1):
                if not self.ignore_episode_boundaries and self.terminals[offset-1]:
                    #found terminal so leave the rest of the array as zeros
                    break
                state0[state_idx] = self.states[offset]
                
                state_idx = state_idx - 1
            
            assert len(state0) == self.window_length
            
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.states[idx+1])
            state1 = np.array(state1)
            assert len(state1) == self.window_length
            
            state0_batch[batch] = state0
            action_batch[batch] = self.actions[idx]
            reward_batch[batch] = self.rewards[idx]
            state1_batch[batch] = state1
            terminal_batch[batch] = self.terminals[idx]
                                                                                            
            experiences.append(Experience(state0=state0, action=self.actions[idx], reward=self.rewards[idx],
                                          state1=state1, terminal1=self.terminals[idx]))
            batch = batch + 1
        assert len(experiences) == batch_size
        return experiences, state0_batch, action_batch, reward_batch, state1_batch, terminal_batch


class EpisodeParameterMemory(SequentialMemory):
    def __init__(self, limit):
        super(EpisodeParameterMemory, self).__init__(limit)
        self.limit = limit

        self.params = RingBuffer(limit)
        self.intermediate_rewards = []
        self.total_rewards = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        if batch_idxs is None:
            batch_idxs = sample_batch_indexes(0, self.nb_entries, size=batch_size)
        assert len(batch_idxs) == batch_size

        batch_params = []
        batch_total_rewards = []
        for idx in batch_idxs:
            batch_params.append(self.params[idx])
            batch_total_rewards.append(self.total_rewards[idx])
        return batch_params, batch_total_rewards

    def append(self, observation, action, reward, terminal, training=True):
        super(EpisodeParameterMemory, self).append(observation, action, reward, terminal, training=training)
        if training:
            self.intermediate_rewards.append(reward)

    def finalize_episode(self, params):
        total_reward = sum(self.intermediate_rewards)
        self.total_rewards.append(total_reward)
        self.params.append(params)
        self.intermediate_rewards = []

    @property
    def nb_entries(self):
        return len(self.total_rewards)

    def get_config(self):
        config = super(Memory, self).get_config()
        config['limit'] = self.limit
        return config

    @property
    def is_episodic(self):
        return True


class EpisodicMemory(object):
    def __init__(self, limit):
        self.limit = limit
        self.episodes = deque(maxlen=limit)
        self.terminal = False
        
        self.states = deque()
        self.actions = deque()
        self.rewards = deque()
        self.terminals = deque()
        
    def get_recent_state(self, current_observation):
        state = np.array(current_observation)
        # add batch and time dimention
        state = state.reshape((1,1,) + state.shape)
        return state
    
    def sample(self, batch_size, batch_idxs=None):
        if len(self.episodes) <= 1:
            # We don't have a complete episode yet ...
            return []

        if batch_idxs is None:
            # Draw random indexes such that we never use the last episode yet, which is
            # always incomplete by definition.
            batch_idxs = sample_batch_indexes(0, self.nb_entries - 1, size=batch_size)
        assert np.min(batch_idxs) >= 0
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # get max episode length, used for padding
        max_eps_len = 0
        for idx in batch_idxs:
            #while len(self.episodes[idx]) == 0:
            #    idx = sample_batch_indexes(0, self.nb_entries, size=1)[0]
            if len(self.episodes[idx].states) - 1 > max_eps_len:
                max_eps_len = len(self.episodes[idx].states) - 1


        # Create sequence of experiences.
        sequences = []
        state0_eps = np.zeros((batch_size, max_eps_len,) + shape_from_object(self.episodes[0].states[0]))
        action_eps = np.zeros((batch_size, max_eps_len,), dtype=type(self.episodes[0].actions[0]))
        reward_eps = np.zeros((batch_size, max_eps_len,))
        state1_eps = np.zeros((batch_size, max_eps_len,) + shape_from_object(self.episodes[0].states[0]))
        terminal1_eps = np.ones((batch_size, max_eps_len,), dtype=bool)
        
        for iter, batch_idx in enumerate(batch_idxs):
            episode = self.episodes[batch_idx]

            # Transform into experiences (to be consistent).
            sequence = []
            eps_len = len(episode.states) - 1
            for idx in range(1,eps_len+1):
                sequence.append(Experience(state0=episode.states[idx], state1=episode.states[idx-1],
                                        reward=episode.rewards[idx], action=episode.actions[idx], terminal1=episode.terminals[idx]))
                
            assert len(sequence) == eps_len
            sequences.append(sequence)
            
            state0_eps[iter, 0:eps_len] = episode.states[0:-1]
            action_eps[iter, 0:eps_len] = episode.actions[0:-1]
            reward_eps[iter, 0:eps_len] = episode.rewards[0:-1]
            terminal1_eps[iter, 0:eps_len] = episode.terminals[0:-1]
            state1_eps[iter, 0:eps_len] = episode.states[1:]
        assert len(sequences) == batch_size
        return sequences, state0_eps, action_eps, reward_eps, state1_eps, terminal1_eps

    def append(self, observation, action, reward, terminal, training=True):
        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            self.states.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)
            
            if self.terminal:
                self.episodes.append(Timesteps(states=np.array(self.states), 
                                               actions=np.array(self.actions, dtype=type(self.actions[0])),
                                               rewards=np.array(self.rewards),
                                               terminals=np.array(self.terminals, dtype=bool)))
                
                self.states.clear()
                self.actions.clear()
                self.rewards.clear()
                self.terminals.clear()
                
            self.terminal = terminal
            
    @property
    def nb_entries(self):
        return len(self.episodes)

    def get_config(self):
        config['limit'] = self.limit
        return config

    @property
    def is_episodic(self):
        return True


class PrioritizedEpisodicMemory(EpisodicMemory):
    def __init__(self, limit, key=None):
        super(PrioritizedEpisodicMemory, self).__init__(limit)
        
        if key is None:
            self.key = lambda episode: len(episode.actions)
        else:
            self.key = key
            
    def append(self, observation, action, reward, terminal, training=True):
        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            self.states.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)
            
            if self.terminal:
                if len(self.episodes) == self.limit:
                    self.episodes = deque(sorted(self.episodes, key=self.key), maxlen=self.limit)
                
                self.episodes.append(Timesteps(states=np.array(self.states), 
                                               actions=np.array(self.actions, dtype=type(self.actions[0])),
                                               rewards=np.array(self.rewards),
                                               terminals=np.array(self.terminals, dtype=bool)))
                    
                self.states.clear()
                self.actions.clear()
                self.rewards.clear()
                self.terminals.clear()
                
            self.terminal = terminal
    
