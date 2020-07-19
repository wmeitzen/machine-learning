from collections import defaultdict
#from env import BlobEnv
from simple_lawnmower_sjhatfield.env import BlobEnv
import pickle
import numpy as np
from tqdm import tqdm

"""
# - 5x5
DISCOUNT = 0.99
NUM_EPSIODES = 10_00  # originally: 100_000
EPSILON_DECAY = 0.999
# If EPSILON_MIN = 1 then always randomly explore
EPSILON_MIN = 0.1
LEARNING_RATE_MIN = 0.05
LEARNING_RATE_DECAY = 0.999
"""

"""
# - 10x10
# - best so far
DISCOUNT = 0.99
NUM_EPSIODES = 10_00  # originally: 100_000
EPSILON_DECAY = 0.997
# If EPSILON_MIN = 1 then always randomly explore
EPSILON_MIN = 0.1
LEARNING_RATE_MIN = 0.05
LEARNING_RATE_DECAY = 0.9999
"""


class QLearning:
    def __init__(
        self,
        env: BlobEnv,
        DISCOUNT: float = 0.99,
        NUM_EPSIODES: int = 1_000,
        EPSILON_DECAY: float = 0.997,
        EPSILON_MIN: float = 0.1,
        LEARNING_RATE_MIN: float = 0.05,
        LEARNING_RATE_DECAY: float = 0.999
    ):
        self.DISCOUNT = DISCOUNT
        self.NUM_EPSIODES = NUM_EPSIODES
        self.EPSILON_DECAY = EPSILON_DECAY
        self.EPSILON_MIN = EPSILON_MIN
        self.LEARNING_RATE_MIN = LEARNING_RATE_MIN
        self.LEARNING_RATE_DECAY = LEARNING_RATE_DECAY
        self.env = env

    def set_discount(self, DISCOUNT):
        self.DISCOUNT = DISCOUNT

    def set_num_episodes(self, NUM_EPSIODES):
        self.NUM_EPSIODES = NUM_EPSIODES

    def set_epsilon_decay(self, EPSILON_DECAY):
        self.EPSILON_DECAY = EPSILON_DECAY

    def set_epsilon_min(self, EPSILON_MIN):
        self.EPSILON_MIN = EPSILON_MIN

    def set_learning_rate_min(self, LEARNING_RATE_MIN):
        self.LEARNING_RATE_MIN = LEARNING_RATE_MIN

    def set_learning_rate_decay(self, LEARNING_RATE_DECAY):
        self.LEARNING_RATE_DECAY = LEARNING_RATE_DECAY

    def set_size(self, size):
        self.env.set_size(size)

    def launch_qlearning(self):

        # Store the Q-value for each state, action pair encountered
        # default value for each is given by the defaultdict
        Q = defaultdict(lambda: [0, 0, 0])
        # It nice to know how many times each state, action pair
        # has been visited
        Q_count = defaultdict(lambda: [0, 0, 0])

        # Initial epsilon and learning rate before decay begins
        epsilon = 1
        learning_rate = 1

        # self.env = BlobEnv()
        segs = self.env.get_size()
        #print(f"self.env.get_size(): {segs}")
        # self.env.set_size(self.env.get_size())

        # Store the episode durations
        episode_durations = []

        #print(f"self.NUM_EPSIODES = {self.NUM_EPSIODES}")
        #print(f"self.LEARNING_RATE_DECAY = {self.LEARNING_RATE_DECAY}")
        #print(f"self.LEARNING_RATE_MIN = {self.LEARNING_RATE_MIN}")

        for i in tqdm(range(self.NUM_EPSIODES), ascii=True, unit="episodes"):
            state, _, done = self.env.reset()

            while not done:
                # With chance epsilon select a random choice
                if np.random.random() < epsilon:
                    action = np.random.randint(3)
                # Otherwise select the action with maximum Q-value
                else:
                    action = np.argmax(Q[state.tobytes()])
                # Make a copy of state as it gets altered in the env.step
                # unsure why it is changing...
                state = state.copy()
                # Execute the action
                next_state, reward, done = self.env.step(action)
                # print(f"reward: {reward}, done: {done}, learning_rate: {learning_rate}")
                # Perform the Q-value update according to the standard update rule
                Q[state.tobytes()][action] += learning_rate * (
                    reward
                    + self.DISCOUNT * np.max(Q[next_state.tobytes()])
                    - Q[state.tobytes()][action]
                )
                # Update the state,action pair count
                Q[state.tobytes()][action] += 1
                # Update the current state for next iteration
                state = next_state.copy()

            # Decay epsilon as episode is over
            epsilon = max(self.EPSILON_MIN, epsilon * self.EPSILON_DECAY)
            #print(f"epsilon: {epsilon}")
            learning_rate = max(
                self.LEARNING_RATE_MIN, learning_rate * self.LEARNING_RATE_DECAY
            )
            #print(f"learning_rate = {learning_rate}")
            # Store the episode length
            # print(f"get_episode_step(): {env.get_episode_step()}")
            episode_durations.append(self.env.get_episode_step())

            # We store the Q-values and counts found in a file in case we wish to end
            # learning early.
            # Save it 100 times.
            # if i % (NUM_EPSIODES / 100) == 0:
            # Save it every 100 episodes.
            if i % 100 == 0:
                with open("Q_values.pickle", "wb") as f:
                    pickle.dump(dict(Q), f)
                with open("counts.pickle", "wb") as f:
                    pickle.dump(dict(Q_count), f)
                #print(learning_rate)
                #print(epsilon)

        # One last save
        with open("Q_values.pickle", "wb") as f:
            pickle.dump(dict(Q), f)
        with open("counts.pickle", "wb") as f:
            pickle.dump(dict(Q_count), f)
