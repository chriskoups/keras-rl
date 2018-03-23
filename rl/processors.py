import numpy as np

from rl.core import Processor
from rl.util import WhiteningNormalizer

from gym import spaces


class MultiInputProcessor(Processor):
    """Converts observations from an environment with multiple observations for use in a neural network
    policy.

    In some cases, you have environments that return multiple different observations per timestep 
    (in a robotics context, for example, a camera may be used to view the scene and a joint encoder may
    be used to report the angles for each joint). Usually, this can be handled by a policy that has
    multiple inputs, one for each modality. However, observations are returned by the environment
    in the form of a tuple `[(modality1_t, modality2_t, ..., modalityn_t) for t in T]` but the neural network
    expects them in per-modality batches like so: `[[modality1_1, ..., modality1_T], ..., [[modalityn_1, ..., modalityn_T]]`.
    This processor converts observations appropriate for this use case.

    # Arguments
        nb_inputs (integer): The number of inputs, that is different modalities, to be used.
            Your neural network that you use for the policy must have a corresponding number of
            inputs.
    """
    def __init__(self, nb_inputs):
        self.nb_inputs = nb_inputs

    def process_state_batch(self, state_batch):
        input_batches = [[] for x in range(self.nb_inputs)]
        for state in state_batch:
            processed_state = [[] for x in range(self.nb_inputs)]
            for observation in state:
                assert len(observation) == self.nb_inputs
                for o, s in zip(observation, processed_state):
                    s.append(o)
            for idx, s in enumerate(processed_state):
                input_batches[idx].append(s)
        return [np.array(x) for x in input_batches]


class WhiteningNormalizerProcessor(Processor):
    """Normalizes the observations to have zero mean and standard deviation of one,
    i.e. it applies whitening to the inputs.

    This typically helps significantly with learning, especially if different dimensions are
    on different scales. However, it complicates training in the sense that you will have to store
    these weights alongside the policy if you intend to load it later. It is the responsibility of
    the user to do so.
    """
    def __init__(self):
        self.normalizer = None

    def process_state_batch(self, batch):
        if self.normalizer is None:
            self.normalizer = WhiteningNormalizer(shape=batch.shape[1:], dtype=batch.dtype)
        self.normalizer.update(batch)
        return self.normalizer.normalize(batch)
    
class ContinuousToDiscreteActions(Processor):
    """Converts continuous actions to discrete
    """
    def __init__(self, min_action, max_action):
        self.min = round(min_action)
        self.max = round(max_action)
        self.action = None

    def process_action(self, action):
        if type(action) is np.ndarray:
            action = action[0]
          
        if self.action is None:
            self.action = action
        elif abs(action) > abs(self.action):
            self.action = action

        action = round(action)
        if action < self.min:
            action = self.min
        elif action > self.max:
            action = self.max
            
        return int(action)
    
    def process_reward(self, reward):
        if self.action < self.min or self.action > self.max:
            reward -= abs(self.action)
        self.action = None
        return reward
    
class MultiModeCartpole(Processor):
    """Augment classical cartpole problem to reduce the observability of the state.
    
    The states of cart speed and pole speed are removed leaving only cart position and pole angle.
    
    Additinally, the effectiveness of the action is augmented with a mode switch. In mode A, action 0
    causes the cart to move left and action 1 causes the cart to move right. In mode B, the is reversed
    with action 0 moving the cart right and action 1 moving it left. This mode is randomly choosen at
    the start of a new episode. The mode is also added to the observation for the first 50 steps of 
    an episode after which the observation of the mode is set to 0.
    
    This problem is only solvable with a recurrent netwrok architecture.  
    """
    def __init__(self):
        self.steps = 0  # total steps in episode
        self.mode = 1   # mode 1: action 1 will push left and action 2 will push right. mode 2: inversed behavior

    def process_step(self, observation, reward, done, info):
        self.steps += 1
        if done:
            self.steps = 0
            self.mode = 2 * np.random.random_integers(low=0, high=1, size=1)[0] - 1 # random number either -1 or 1
            
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        
        return observation, reward, done, info
        
    def process_observation(self, observation):
        # remove observations of cart and pole velocities
        observation = observation[[0,2]]
        
        # add action mode for first second of the episode
        if self.steps < 50:
            observation = np.append(observation, self.mode)
        else:
            observation = np.append(observation, 0)
        return np.array(observation)
        
    def process_action(self, action):
        return (action * self.mode)
