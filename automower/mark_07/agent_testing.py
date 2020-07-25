import pickle
#from env import BlobEnv
from simple_lawnmower_sjhatfield.env import BlobEnv
import sqlite3 as sql
import numpy as np

class AgentTesting:
    def __init__(
        self
        ,env: BlobEnv
        ,number_of_runs: int = 10
        ,debug_render = True
    ):
        self.env = env
        self.number_of_runs = number_of_runs
        self.debug_render = debug_render


    def set_number_of_runs(self, number_of_runs):
        self.number_of_runs = number_of_runs

    def set_debug_render(self, debug_render):
        self.debug_render = debug_render

    def get_debug_render(self):
        return self.debug_render

    def get_number_of_runs(self):
        return self.number_of_runs

    def set_render_waitkey(self, render_waitkey):
        self.env.set_render_waitkey(render_waitkey)

    def get_render_waitkey(self):
        return self.env.get_render_waitkey()

    def launch_agent_testing(self, db_trial):
        # env = BlobEnv()

        # - exit if we already ran trials
        strSQLiteDB = self.env.get_db_filename()
        objConn = sql.connect(strSQLiteDB)
        objConn.row_factory = sql.Row
        objDB = objConn.cursor()
        strSQL = f"select average_run_length from modelinfo where trial = {db_trial};"
        objDB.execute(strSQL)
        objRow = objDB.fetchone()
        average_run_length = objRow["average_run_length"]
        objConn.close()
        # exit if we already ran through testing
        if average_run_length is not None:
            return

        # Load the Q-values from file to use for decision making
        with open("Q_values.pickle", "rb") as f:
            Q = pickle.load(f)

        # This executes one run to completion
        def one_run(exit_after_n_steps = -1):
            unseen_states = 0
            state, _, done = self.env.reset()
            # The agent could get stuck when following maximal Q-value
            # actions. Therefore, if the agent stays on the same square
            # for 3 or more time steps then we will take a random action
            # It will take 2 time steps to rotate in a corner
            stuck = 0
            number_of_steps = 0
            while not done and (exit_after_n_steps == -1 or number_of_steps < exit_after_n_steps):
                number_of_steps += 1
                #if stuck >= 3:
                #    action = np.random.randint(3)
                #else:
                # The state may not have been seen by the learner
                # as there are so many. We execute a random
                # action if the state was not seen
                if state.tobytes() in Q.keys() and stuck < 3:
                    # Count how many unseen states are encountered
                    unseen_states += 1
                    action = np.argmax(Q[state.tobytes()])
                else:
                    action = np.random.randint(3)
                    stuck = 0

                state = state.copy()

                next_state, reward, done = self.env.step(action, debug_render = self.get_debug_render())

                # next_state, reward, done = env.step(action, render=True)
                if np.where(state >= 0) == np.where(next_state >= 0):
                    stuck += 1
                else:
                    stuck = 0
                state = next_state

            return self.env.get_episode_step(), unseen_states

        # This shows the Q-values for moving straight on from the starting
        # state for 9 steps
        def print_Q_vals():
            state, _, _ = self.env.reset()
            print(Q[state.tobytes()])
            for _ in range(self.get_number_of_runs() - 1):
                state, _, _ = self.env.step(0)
                print(Q[state.tobytes()])

        # print_Q_vals()

        # calculate the perfect run length
        # - (not sure it's accurate, but it's close)
        perfect_run_length = self.env.get_size() ** 2 + ((self.env.get_size()+1) * 2)
        # We run self.get_number_of_runs() times to completion showing and provide some summary statistics
        # at the end
        strSQL = f"update modelinfo set perfect_run_length = {perfect_run_length} where trial = {db_trial}"
        strSQLiteDB = self.env.get_db_filename()
        objConn = sql.connect(strSQLiteDB)
        objDB = objConn.cursor()
        objDB.execute(strSQL)
        objConn.commit()
        objConn.close()

        total = 0
        total_unseen_states = 0
        for i in range(self.get_number_of_runs()):
            # exit_after_n_steps: quit if we're 2x as many steps as it should be - don't waste time
            run_length, unseen_states = one_run(exit_after_n_steps = perfect_run_length * 2)
            #print(f"Run {i+1} length: {run_length}")
            total += run_length
            total_unseen_states += unseen_states
        average_run_length = total / self.get_number_of_runs()
        print(f"average run length over the {self.get_number_of_runs()}: {average_run_length}")
        print(f"perfect run length for {self.env.get_size()}: {perfect_run_length}")
        print(f"average number of unseen states encountered in the {self.get_number_of_runs()}: {unseen_states / self.get_number_of_runs()}")
        strSQL = f"update modelinfo set average_run_length = {average_run_length} where trial = {db_trial}"
        strSQLiteDB = self.env.get_db_filename()
        objConn = sql.connect(strSQLiteDB)
        objDB = objConn.cursor()
        objDB.execute(strSQL)
        objConn.commit()
        objConn.close()
