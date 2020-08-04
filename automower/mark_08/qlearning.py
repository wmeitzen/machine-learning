from collections import defaultdict
#from env import BlobEnv
from simple_lawnmower_sjhatfield.env import BlobEnv
import pickle
import numpy as np
import sqlite3 as sql
import os
import time
from tqdm import tqdm

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
        self.db_trial = -1
        self.env = env

        strSQLiteDB = self.env.get_db_filename()
        if os.path.exists(strSQLiteDB) == False:
            objConn = sql.connect(strSQLiteDB)
            objDB = objConn.cursor()
            objDB.execute("create table modelinfo (trial integer primary key, ts timestamp default (datetime('now','localtime')) not null, average_run_length float, discount float not null, elapsed_learning_time_sec int, elapsed_learning_time_string varchar(64) default 'Processing ...', epsilon_decay float, episode_min float not null, learning_rate_decay float not null, lawn_size int not null, lawn_name varchar(256) not null, num_episodes int not null, perfect_run_length int);")
            # create table modelinfo_new (trial integer primary key, ts timestamp default (datetime('now','localtime')) not null, average_run_length float, discount float, elapsed_learning_time_sec int, elapsed_learning_time_string varchar(64) default 'Processing ...', epsilon_decay float, episode_min float, learning_rate_decay float, lawn_size int, lawn_name varchar(256), num_episodes int, perfect_run_length int);
            # insert into modelinfo (trial, ts, average_run_length, discount, elapsed_learning_time_sec, elapsed_learning_time_string, epsilon_decay, episode_min, learning_rate_decay, lawn_size, lawn_name, num_episodes, perfect_run_length) select trial, ts, average_run_length, discount, elapsed_learning_time_sec, elapsed_learning_time_string, epsilon_decay, episode_min, learning_rate_decay, lawn_size, lawn_name, num_episodes, perfect_run_length from modelinfo_new;
            objDB.execute("create unique index modelinfo_uq on modelinfo (discount, epsilon_decay, episode_min, learning_rate_decay, lawn_size, lawn_name, num_episodes);")
            objConn.commit()
            objConn.close()

    def set_discount(self, DISCOUNT):
        self.DISCOUNT = DISCOUNT

    def set_db_trial(self, db_trial):
        self.db_trial = db_trial

    def get_db_trial(self):
        return self.db_trial

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

    def get_size(self):
        return self.env.get_size()

    def get_lawn_name(self):
        return self.env.get_lawn_name()

    def set_lawn_name(self, lawn_name):
        self.env.set_lawn_name(lawn_name)

    def launch_qlearning(self):

        start_time = time.time()

        strSQLiteDB = self.env.get_db_filename()
        objConn = sql.connect(strSQLiteDB)
        objConn.row_factory = sql.Row
        objDB = objConn.cursor()
        strSQL = f"select trial from modelinfo where discount = {self.DISCOUNT} and epsilon_decay = {self.EPSILON_DECAY} and episode_min = {self.EPSILON_MIN} and learning_rate_decay = {self.LEARNING_RATE_DECAY} and lawn_size = {self.get_size()} and num_episodes = {self.NUM_EPSIODES} and lawn_name = '{self.get_lawn_name()}';"
        objDB.execute(strSQL)
        objRow = objDB.fetchone()
        objConn.close()
        if objRow is not None:
            db_trial = objRow["trial"]
            self.set_db_trial(db_trial)
            return

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
            # Save it every 1000 episodes.
            if i % 1000 == 0:
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

        elapsed_time_sec = int(time.time() - start_time)
        elapsed_time_string = time.strftime("%H hr %M min %S sec", time.gmtime(elapsed_time_sec))
        print(f"elapsed time: {elapsed_time_string}")

        strSQL = f"insert into modelinfo (discount, elapsed_learning_time_sec, elapsed_learning_time_string, epsilon_decay, episode_min, learning_rate_decay, lawn_size, lawn_name, num_episodes) values ({self.DISCOUNT}, {elapsed_time_sec}, '{elapsed_time_string}', {self.EPSILON_DECAY}, {self.EPSILON_MIN}, {self.LEARNING_RATE_DECAY}, {self.get_size()}, '{self.get_lawn_name()}', {self.NUM_EPSIODES});"
        strSQLiteDB = self.env.get_db_filename()
        objConn = sql.connect(strSQLiteDB)
        objDB = objConn.cursor()
        objDB.execute(strSQL)
        objConn.commit()
        objConn.close()

        strSQLiteDB = self.env.get_db_filename()
        objConn = sql.connect(strSQLiteDB)
        objConn.row_factory = sql.Row
        objDB = objConn.cursor()
        strSQL = f"select trial from modelinfo order by trial desc limit 1;"
        objDB.execute(strSQL)
        objRow = objDB.fetchone()
        objConn.close()
        db_trial = objRow["trial"]
        self.set_db_trial(db_trial)
