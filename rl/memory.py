from __future__ import absolute_import
from collections import deque, namedtuple
import warnings
import random

import numpy as np


# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')

EpisodicTimestep = namedtuple('EpisodicTimestep', 'observation, action, reward, terminal')


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


class RingBuffer(object):
    """ RingBuffer.
    
    First-In-First-Out (FIFO) data storage structure. Input data is appended
    sequentially to a buffer array, once the buffer is filled, new input data
    overwrites the oldest data first.
    
    # Arguments
        maxlen: int > 0 Size of the buffer
    
    """
    def __init__(self, maxlen):
        if maxlen <= 0:
            maxlen = 1
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0:
            idx = self.length + idx
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


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
        window_length: int > 0 Size of the buffer
        ignore_episode_boundaries: bool 
    """
    
    def __init__(self, window_length, ignore_episode_boundaries=False):
        self.window_length = window_length 
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_observations = None
        self.recent_terminals = deque([False]*window_length,maxlen=window_length)

    def sample(self, batch_size, batch_idxs=None):
        raise NotImplementedError()

    def append(self, observation, action, reward, terminal, training=True):
        if self.recent_observations == None:
            self.recent_observations = deque(np.zeros((self.window_length,) + shape_from_object(observation)), maxlen=self.window_length)
        
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)

    def get_recent_states(self, current_observation):
        if self.recent_observations == None:
            self.recent_observations = deque(np.zeros((self.window_length,) + shape_from_object(current_observation)), maxlen=self.window_length)
        
        state = np.zeros((self.window_length,) + shape_from_object(current_observation))
        state[-1] = current_observation
        for idx in range(self.window_length-1, 0, -1):
            if not self.ignore_episode_boundaries and self.recent_terminals[idx-1]:
                #found terminal so leave the rest of the array as zeros
                break
            
            state[idx-1] = self.recent_observations[idx]

        assert len(state) is self.window_length
        
        return state

    def get_config(self):
        config = {
            'window_length': self.window_length,
            'ignore_episode_boundaries': self.ignore_episode_boundaries,
        }
        return config


class SequentialMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(SequentialMemory, self).__init__(**kwargs)
        
        self.limit = limit
        self.actions = deque(maxlen=limit)
        self.rewards = deque(maxlen=limit)
        self.terminals = deque(maxlen=limit)
        self.observations = deque(maxlen=limit)

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
        experiences = []
        for idx in batch_idxs:
            state0 = np.zeros((self.window_length,) + shape_from_object(self.observations[0]))
            state_idx = self.window_length - 1
            for offset in range(idx, idx - self.window_length, -1):
                if not self.ignore_episode_boundaries and self.terminals[offset-1]:
                    #found terminal so leave the rest of the array as zeros
                    break
                state0[state_idx] = self.observations[offset]
                
                state_idx = state_idx - 1
            
            assert len(state0) == self.window_length
            
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.observations[idx+1])
            assert len(state1) == self.window_length
            
            experiences.append(Experience(state0=state0, action=self.actions[idx], reward=self.rewards[idx],
                                          state1=state1, terminal1=self.terminals[idx]))
        assert len(experiences) == batch_size
        return experiences

    def append(self, observation, action, reward, terminal, training=True):
        super(SequentialMemory, self).append(observation, action, reward, terminal, training=training)
        
        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)

    @property
    def nb_entries(self):
        return len(self.observations)

    def get_config(self):
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config

    @property
    def is_episodic(self):
        return False


class EpisodicMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(EpisodicMemory, self).__init__(**kwargs)
        
        self.limit = limit
        self.episodes = RingBuffer(limit)
        self.terminal = False

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

        # Create sequence of experiences.
        sequences = []
        for idx in batch_idxs:
            episode = self.episodes[idx]
            while len(episode) == 0:
                idx = sample_batch_indexes(0, self.nb_entries, size=1)[0]

            # Bootstrap state.
            running_state = deque(maxlen=self.window_length)
            for _ in range(self.window_length - 1):
                running_state.append(np.zeros(episode[0].observation.shape))
            assert len(running_state) == self.window_length - 1

            states, rewards, actions, terminals = [], [], [], []
            terminals.append(False)
            for idx, timestep in enumerate(episode):
                running_state.append(timestep.observation)
                states.append(np.array(running_state))
                rewards.append(timestep.reward)
                actions.append(timestep.action)
                terminals.append(timestep.terminal)  # offset by 1, see `terminals.append(False)` above
            assert len(states) == len(rewards)
            assert len(states) == len(actions)
            assert len(states) == len(terminals) - 1

            # Transform into experiences (to be consistent).
            sequence = []
            for idx in range(len(episode) - 1):
                state0 = states[idx]
                state1 = states[idx + 1]
                reward = rewards[idx]
                action = actions[idx]
                terminal1 = terminals[idx + 1]
                experience = Experience(state0=state0, state1=state1, reward=reward, action=action, terminal1=terminal1)
                sequence.append(experience)
            sequences.append(sequence)
            assert len(sequence) == len(episode) - 1
        assert len(sequences) == batch_size
        return sequences

    def append(self, observation, action, reward, terminal, training=True):
        super(EpisodicMemory, self).append(observation, action, reward, terminal, training=training)
        
        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            timestep = EpisodicTimestep(observation=observation, action=action, reward=reward, terminal=terminal)
            if len(self.episodes) == 0:
                self.episodes.append([])  # first episode
            self.episodes[-1].append(timestep)
            if self.terminal:
                self.episodes.append([])
            self.terminal = terminal

    @property
    def nb_entries(self):
        return len(self.episodes)

    def get_config(self):
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config

    @property
    def is_episodic(self):
        return True


class EpisodeParameterMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(EpisodeParameterMemory, self).__init__(**kwargs)
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
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config

    @property
    def is_episodic(self):
        return True
