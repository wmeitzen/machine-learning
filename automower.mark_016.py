
"""
Like mark 15, except:
use multiprocessing (set "use_multiprocessing = True" where possible)
Adjust pentalties
Plot one line, not several
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
import tensorflow as tf
from keras.optimizers import Adam
from collections import deque
from tqdm import tqdm
from PIL import Image
from datetime import datetime
import random
import cv2
import os
from matplotlib import pyplot as plt
import sys
import sqlite3 as sql

# Stats settings
SHOW_PREVIEW = True # - was False in original code

RUN_FROM_IDE = False # - False: run from CLI, use "python3 automower ..."

# Environment settings
if RUN_FROM_IDE == True:
    EPISODES = 20
    AGGREGATE_STATS_EVERY = 5 # was 50 # number of episodes
else:
    EPISODES = 20_000
    AGGREGATE_STATS_EVERY = 25 # was 50 # number of episodes

if RUN_FROM_IDE == True:
    MODEL_NAME = "Herba_IDE_016"
else:
    MODEL_NAME = "Herba_CLI_016"

class Blob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)
        self.direction = np.random.randint(0, 3) # [0 - 2)

    def __init__(self, size, x, y, direction):
        self.size = size
        self.x = x
        self.y = y
        self.direction = direction

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        '''
        3 total movement options: (0, 1, 2)
        1: turn left
        0: move forward
        2: turn right
        '''
        if choice == 1: # turn left
            self.direction = (self.direction - 1) % 8
        elif choice == 2: # turn right
            self.direction = (self.direction + 1) % 8
        elif choice == 0 and self.direction == 0:
            self.move(x=0, y=-1)
        elif choice == 0 and self.direction == 1:
            self.move(x=1, y=-1)
        elif choice == 0 and self.direction == 2:
            self.move(x=0, y=-1)
        elif choice == 0 and self.direction == 3:
            self.move(x=1, y=1)
        elif choice == 0 and self.direction == 4:
            self.move(x=0, y=1)
        elif choice == 0 and self.direction == 5:
            self.move(x=-1, y=1)
        elif choice == 0 and self.direction == 6:
            self.move(x=-1, y=0)
        elif choice == 0 and self.direction == 7:
            self.move(x=-1, y=-1)


    def move(self, x, y):

        self.x += x
        self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1

class BlobEnv:
    SIZE = 10 # - was 10; 6 leads to errors; 7 is min
    RETURN_IMAGES = True
    MOVE_PENALTY_OR_REWARD = -1 # pos value = reward, neg = penalty
    #ENEMY_PENALTY = 300 # - subtract 300 from reward
    #FOOD_REWARD = 2000 # - add 2000 to reward
    MOWED_GRASS_PENALTY = -5 # was -70; subtract from reward
    UNMOWED_GRASS_REWARD = 5 # was +90; add to reward
    COMPLETED_REWARD = 10000
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # I think 3 is: mower, unmowed grass, mowed grass
    ACTION_SPACE_SIZE = 3 # was 9, b/c zero movement was a choice

    #PLAYER_N = 1  # player key in dict
    #FOOD_N = 2  # food key in dict
    #ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    #d = {1: (255, 175, 0),
    #     2: (0, 255, 0),
    #     3: (0, 0, 255)}
    MOWER_COUNT = 1 # number of mowers, not used yet
    ANIMAL_COUNT = 1 # number of animals, not used yet
    MOWER_KEY = 1  # mower key in dict
    UNMOWED_GRASS_KEY = 2  # unmowed grass key in dict
    MOWED_GRASS_KEY = 3  # mowed grass key in dict
    # the dict! (colors)
    d = {MOWER_KEY: (255, 51, 255) # pink
         ,UNMOWED_GRASS_KEY: (0, 51, 25) # dark green
         ,MOWED_GRASS_KEY: (102, 255, 102) # light green
         }
    env = np.zeros((SIZE, SIZE, 3), dtype = np.uint8)
    UNITS_TO_MOW = int(SIZE * SIZE * 0.5) # require 50% mowed before trying again, 2-3 sec / episode
    #UNITS_TO_MOW = int(SIZE * SIZE * 1.0)  # require 100% mowed before trying again, 10-12 sec / episode
    remaining_units_to_mow = 0

    def reset(self):
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                self.env[i][j] = self.d[self.UNMOWED_GRASS_KEY]

        #self.player = Blob(self.SIZE)
        #self.food = Blob(self.SIZE)
        #while self.food == self.player:
        #    self.food = Blob(self.SIZE)
        #self.enemy = Blob(self.SIZE)
        #while self.enemy == self.player or self.enemy == self.food:
        #    self.enemy = Blob(self.SIZE)

        # def __init__(self, size, x, y, direction):
        self.mower = Blob(self.SIZE, 0, 0, 2)  # start mower in the corner, point right

        self.episode_step = 0

        #observation = np.array(self.get_image())

        self.remaining_units_to_mow = self.UNITS_TO_MOW

        return self.env

    def step(self, action):
        self.episode_step = self.episode_step + 1
        self.mower.action(action)

        #### MAYBE ###
        #self.enemy.move()
        #self.food.move()
        ##############

        reward = 0

        new_observation = np.array(self.get_image())

        # - calculate reward based on if the mower is on mowed or unmowed grass
        if all(new_observation[self.mower.x][self.mower.y] == self.d[self.MOWED_GRASS_KEY]):
            reward = reward + self.MOWED_GRASS_PENALTY

        if all(new_observation[self.mower.x][self.mower.y] == self.d[self.UNMOWED_GRASS_KEY]):
            reward = reward + self.UNMOWED_GRASS_REWARD
            self.remaining_units_to_mow -= 1

        self.env[self.mower.x][self.mower.y] = self.d[self.MOWED_GRASS_KEY]

        reward = reward + self.MOVE_PENALTY_OR_REWARD

        if self.remaining_units_to_mow <= 0:
            reward = reward + self.COMPLETED_REWARD # add bonus when finished

        done = False
        if self.remaining_units_to_mow <= 0: # - was ">= 90"
            done = True

        return new_observation, reward, done

    def render(self):
        #img = self.get_image()
        #img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        #cv2.imshow("image", np.array(img))  # show it!
        env = np.array(self.get_image())  # - not wrapping with "np.array(" smears the mower everywhere
        env[self.mower.x][self.mower.y] = self.d[self.MOWER_KEY]
        #cv2.resize(env, 300, 300)
        if RUN_FROM_IDE == True:
            title = f"{MODEL_NAME} from IDE"
        else:
            title = f"{MODEL_NAME} from CLI"
        try:
            resized = cv2.resize(env, (300, 300))
            cv2.imshow(title, resized)
            #cv2.waitKey(1)  # - 100 ms
        except:
            pass
        cv2.waitKey(1)  # - 1 ms, must be int

    # FOR CNN #
    # Image used to be just for display. Now, it's an input value.
    def get_image(self):
        img = self.env
        #env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        #env[self.food.x][self.food.y] = self.d[self.FOOD_N]  # sets the food location tile to green color
        #env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]  # sets the enemy location to red
        #env[self.player.x][self.player.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        #img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img


# Own Tensorboard class
# pasted from https://stackoverflow.com/questions/58711624/modifying-tensorboard-in-tensorflow-2-0
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, MODEL_NAME)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overridden, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overridden
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overridden, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

class DQNAgent:
    def __init__(self, env, REPLAY_MEMORY_SIZE, LEARNING_RATE):

        # - main model - gets trained every step
        self.model = self.create_model(env, LEARNING_RATE)

        # - target model - this is what we .predict against every step
        self.target_model = self.create_model(env, LEARNING_RATE)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen = REPLAY_MEMORY_SIZE)

        #self.tensorboard = ModifiedTensorBoard(log_dir = f"logs/{MODEL_NAME}-{int(time.time())}")
        ts = datetime.now()
        full_model_name = f"{MODEL_NAME}-{ts.year}-{str(ts.month).zfill(2)}-{str(ts.day).zfill(2)}-{str(ts.hour).zfill(2)}{str(ts.minute).zfill(2)}"
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{full_model_name}.tb")

        objConn = sql.connect(f'models/modelinfo.sqlite')
        objDB = objConn.cursor()
        #objDB.execute(f"drop table if exists modelinfo;")
        objDB.execute(f"create table if not exists modelinfo (started_at timestamp not null, MODEL_NAME varchar not null, EPISODES int, AGGREGATE_STATS_EVERY int, DISCOUNT float, LEARNING_RATE float, REPLAY_MEMORY_SIZE int, MIN_REPLAY_MEMORY_SIZE int, MINIBATCH_SIZE int, UPDATE_TARGET_EVERY int, MIN_REWARD int, EPSILON_DECAY float, MIN_EPSILON float, MOVE_PENALTY_OR_REWARD int, MOWED_GRASS_PENALTY int, UNMOWED_GRASS_REWARD int, UNITS_TO_MOW int);")
        objConn.commit()
        objConn.close()

        self.target_update_counter = 0

    def write_model_info_to_db(self, MODEL_NAME, EPISODES, AGGREGATE_STATS_EVERY, DISCOUNT, LEARNING_RATE, REPLAY_MEMORY_SIZE, MIN_REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, UPDATE_TARGET_EVERY, MIN_REWARD, EPSILON_DECAY, MIN_EPSILON, MOVE_PENALTY_OR_REWARD, MOWED_GRASS_PENALTY, UNMOWED_GRASS_REWARD, UNITS_TO_MOW):
        objConn = sql.connect(f'models/modelinfo.sqlite')
        objDB = objConn.cursor()
        objDB.execute(f"insert into modelinfo (started_at, MODEL_NAME, EPISODES, AGGREGATE_STATS_EVERY, DISCOUNT, LEARNING_RATE, REPLAY_MEMORY_SIZE, MIN_REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, UPDATE_TARGET_EVERY, MIN_REWARD, EPSILON_DECAY, MIN_EPSILON, MOVE_PENALTY_OR_REWARD, MOWED_GRASS_PENALTY, UNMOWED_GRASS_REWARD, UNITS_TO_MOW) values (datetime(current_timestamp, 'localtime'), '{MODEL_NAME}', {EPISODES}, {AGGREGATE_STATS_EVERY}, {DISCOUNT}, {LEARNING_RATE}, {REPLAY_MEMORY_SIZE}, {MIN_REPLAY_MEMORY_SIZE}, {MINIBATCH_SIZE}, {UPDATE_TARGET_EVERY}, {MIN_REWARD}, {EPSILON_DECAY}, {MIN_EPSILON}, {MOVE_PENALTY_OR_REWARD}, {MOWED_GRASS_PENALTY}, {UNMOWED_GRASS_REWARD}, {UNITS_TO_MOW});")
        objConn.commit()
        objConn.close()

    def create_model(self, envir, LEARNING_RATE):
        #model = Sequential()
        # - kernel size (in parens) must be odd: (1, 1); (3, 3); etc.
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape = envir.OBSERVATION_SPACE_VALUES))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2)) # - 0.2 = 20%

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2)) # - 0.2 = 20%

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(envir.ACTION_SPACE_SIZE, activation = "linear"))
        #model.compile(loss = "mse", optimizer = Adam(lr = 0.001), metrics = ["accuracy"])
        model.compile(loss="mse", optimizer = Adam(lr = LEARNING_RATE), metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        #return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]
        return self.model.predict(x = np.array(state).reshape(-1, *state.shape), use_multiprocessing = True)[0]

    def train(self, terminal_state, step, MIN_REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, DISCOUNT, UPDATE_TARGET_EVERY):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model_prediction for current Q values
        current_state = np.array([transition[0] for transition in minibatch]) # removed " / 255"
        current_qs_list = self.model.predict(current_state) # - the "crazy" model

        new_current_states = np.array([transition[3] for transition in minibatch]) # removed " / 255"
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        # verbose=0: show nothing; verbose = 2: show everything
        #self.model.fit(np.array(X) / 255, np.array(y), batch_size = MINIBATCH_SIZE, verbose = 0, shuffle = False, callbacks = [self.tensorboard] if terminal_state else None)
        self.model.fit(x = np.array(X), y = np.array(y), batch_size=MINIBATCH_SIZE, verbose=0,
                use_multiprocessing = True, shuffle = False, callbacks = [self.tensorboard] if terminal_state else None)

        #self.model.fit(np.array(X) / 255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard])
        #if terminal_state:
        #    self.model.fit(np.array(X) / 255, np.array(y), batch_size = MINIBATCH_SIZE, verbose = 0, shuffle = False)

        # - updating to determine if we want to update the target_model yet
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

def learn_stuff():

    env = BlobEnv()
    global epsilon # declare any global variables like this that are modified within the fn - stupid hack, if you ask me

    # For stats
    ep_rewards = [-200]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    #tf.set_random_seed(1) # - does not exist
    tf.random.set_seed(1)

    # Memory fraction, used mostly when training multiple agents
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    #backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    DISCOUNT = 0.99
    REPLAY_MEMORY_SIZE = 50000
    MIN_REPLAY_MEMORY_SIZE = 1000
    # MODEL_NAME = "256x2"
    MIN_REWARD = -1500  # was -200 for the chicken / fox / seed learning
    # MEMORY_FRACTION = 0.20 - # used by GPU code
    # Exploration settings
    MIN_EPSILON = 0.001

    MINIBATCH_SIZE = 10 # - originally 64
    UPDATE_TARGET_EVERY = 5
    EPSILON_DECAY = 0.99975
    LEARNING_RATE = 0.001 # originally 0.001
    SAVE_MODEL_IF_MIN_REWARD_REACHED = False

    #plottype = 0

    # Exploration starting point
    epsilon = 1.0  # - not a constant, this will decay

    agent = DQNAgent(env, REPLAY_MEMORY_SIZE, LEARNING_RATE)

    agent.write_model_info_to_db(MODEL_NAME, EPISODES, AGGREGATE_STATS_EVERY, DISCOUNT, LEARNING_RATE, REPLAY_MEMORY_SIZE, MIN_REPLAY_MEMORY_SIZE,
       MINIBATCH_SIZE, UPDATE_TARGET_EVERY, MIN_REWARD, EPSILON_DECAY, MIN_EPSILON,
       env.MOVE_PENALTY_OR_REWARD, env.MOWED_GRASS_PENALTY, env.UNMOWED_GRASS_REWARD, env.UNITS_TO_MOW)

    for episode in tqdm(range(1, EPISODES + 1), ascii = True, unit = "episodes"):
        agent.tensorboard.step = episode

        episode_reward = 0
        step = 1
        current_state = env.reset()

        done = False

        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state))
            else:
                action = np.random.randint(0, env.ACTION_SPACE_SIZE)

            new_state, reward, done = env.step(action)

            episode_reward = episode_reward + reward

            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                #for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episodes"):
                #print(f"Step: {step}, Episode reward: {episode_reward}")
                current_time = datetime.now().time()
                current_hour = current_time.hour
                show_lawn_until = 22 # - 20 = 8pm, 22 = 10pm, 23 = 11pm; at this hour:00, it will stop drawing the lawn
                show_lawn_at = 6
                if not((current_hour >= show_lawn_until and current_hour <= 23) or (current_hour >= 0 and current_hour <= show_lawn_at)):
                    s = f"Step: {step}, Episode reward: {episode_reward}"
                    sys.stdout.write("\r" + s)
                    sys.stdout.flush()
                    try:
                        env.render()
                    except:
                        pass

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step, MIN_REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, DISCOUNT, UPDATE_TARGET_EVERY)

            current_state = new_state

            step = step + 1

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if episode == 1 or episode >= EPISODES or not episode % AGGREGATE_STATS_EVERY:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            #print(f"min_reward:{min_reward}, MIN_REWARD: {MIN_REWARD}")
            #if min_reward >= MIN_REWARD:
            #    print("if min_reward >= MIN_REWARD:")
            #    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

            print(f"\naverage_reward:{average_reward}, MIN_REWARD: {MIN_REWARD}")
            if average_reward >= MIN_REWARD and SAVE_MODEL_IF_MIN_REWARD_REACHED == True:
                print("if average_reward >= MIN_REWARD:")
                agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min.model')

            if episode >= 5:
                moving_average = np.convolve(ep_rewards, np.ones((AGGREGATE_STATS_EVERY,)) / AGGREGATE_STATS_EVERY, mode="valid")
                #plottype = (plottype + 1) % 3
                try:
                    plt.plot(moving_average, color="blue")
                    #if plottype == 0:
                    #    plt.plot(moving_average, color="red")
                    #elif plottype == 1:
                    #    plt.plot([i for i in range(len(moving_average))], color="blue")
                    #else:
                    #    plt.plot([i for i in range(len(moving_average))], moving_average, color="green")
                    plt.ylabel(f"reward {AGGREGATE_STATS_EVERY} moving average")
                    plt.xlabel("episode #")
                    # plt.draw()
                    plt.pause(0.3)
                    plt.show(block = False)
                except:
                    pass
                if episode >= EPISODES:
                    # - we're finished
                    #plt.ioff()
                    plt.ion()
                    print("finished script")
                    #input("Press Enter to continue...")
                else:
                    plt.ion()  # ion: turn on interactive mode; ioff: turn off interactive mode (wait for user to close)

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon = epsilon * EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)


learn_stuff()
