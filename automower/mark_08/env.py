# Imports
from typing import Union
import numpy as np
from random import choice
import cv2

SEED = 1
POSSIBLE_DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# Set seed for reproducibility
np.random.seed(SEED)


# Main function performs some testing
def main():
    """
    blob_env = BlobEnv(
        size=10, initial_mower_x=0, initial_mower_y=0, initial_mower_direction="E"
    )
    """
    blob_env = BlobEnv()

    # Shows rotation anti-clockwise of what the CNN uses
    for _ in range(16):
        blob_env.step(action=1, debug_render=True)

    blob_env.reset()

    # Shows movement Eastward
    for _ in range(10):
        blob_env.step(action=0, render=True)

    """
    blob_env = BlobEnv(size=10)
    """
    blob_env = BlobEnv()

    # Shows 5 random episodes of length 100
    for _ in range(5):
        for _ in range(100):
            blob_env.step(np.random.choice([0, 1, 2]), render=True)
        blob_env.reset()

    # One run showing the CNN image render for 50 steps
    for _ in range(50):
        blob_env.step(np.random.choice([0, 1, 2]), debug_render=True)
    blob_env.reset()

    # Shows one run to completion for 30% coverage
    # blob_env = BlobEnv(size=10, percent_mowed_to_end=0.3)
    # blob_env = BlobEnv(percent_mowed_to_end=0.3)
    # Shows one run to completion for whatever % coverage is in the class
    blob_env = BlobEnv()
    done = False
    total_reward = 0
    print("Rewards experienced: ")
    while not done:
        _, reward, done = blob_env.step(np.random.choice([0, 1, 2]), render=True)
        total_reward += reward
        print(reward, end=", ")
    print(f"\nTotal reward for running until completion: {total_reward}")


# Classes
class Blob:
    """
    Blob represents a one square size character on the grid. In this context it
    will be the lawnmower. It is initialized in a space with an orientation.
    It can then move forward or rotate by 90 degrees either way.
    """

    def __init__(
        self,
        env_size: int,
        x: int = None,
        y: int = None,
        direction: str = None,
        max_x: int = None,
        max_y: int = None,
    ):
        """
        Initializes a Blob either at coordinates and direction given or randomly chosen
        """
        assert direction in POSSIBLE_DIRECTIONS, (
            "Direction must be one of N, NE, E, SE, S, SW, W, NW"
            " which are the eight compass directions"
        )
        self.env_size = env_size

        if x != None:
            self.x = x
        else:
            self.x = 0

        if y != None:
            self.y = y
        else:
            self.y = 0

        if direction != None:
            self.direction = direction
        else:
            self.direction = np.random.choice(POSSIBLE_DIRECTIONS)

        if max_x != None:
            self.max_x = max_x
        else:
            self.max_x = self.env_size - 1

        if max_y != None:
            self.max_y = max_y
        else:
            self.max_y = self.env_size - 1

    def set_max_x(self, max_x):
        self.max_x = max_x

    def get_max_x(self):
        return self.max_x

    def set_max_y(self, max_y):
        self.max_y = max_y

    def get_max_y(self):
        return self.max_y

    def action(self, choice):
        """
        3 total movement options: (0, 1, 2)
        1: turn left (anti-clockwise)
        0: move forward
        2: turn right (clockwise)
        """
        assert choice in [0, 1, 2,], (
            "Choice must be one of 0: move forward, "
            "1: turn left (anti-clockwise), 2: turn right (clockwise)"
        )
        if choice == 1:  # turn left
            self.direction = POSSIBLE_DIRECTIONS[
                (POSSIBLE_DIRECTIONS.index(self.direction) - 1) % 8
            ]

        elif choice == 2:  # turn right
            self.direction = POSSIBLE_DIRECTIONS[
                (POSSIBLE_DIRECTIONS.index(self.direction) + 1) % 8
            ]

        elif choice == 0 and self.direction == "N":
            self.move(x_step=0, y_step=-1)

        elif choice == 0 and self.direction == "NE":
            self.move(x_step=1, y_step=-1)

        elif choice == 0 and self.direction == "E":
            self.move(x_step=1, y_step=0)

        elif choice == 0 and self.direction == "SE":
            self.move(x_step=1, y_step=1)

        elif choice == 0 and self.direction == "S":
            self.move(x_step=0, y_step=1)

        elif choice == 0 and self.direction == "SW":
            self.move(x_step=-1, y_step=1)

        elif choice == 0 and self.direction == "W":
            self.move(x_step=-1, y_step=0)

        elif choice == 0 and self.direction == "NW":
            self.move(x_step=-1, y_step=-1)

    def move(self, x_step, y_step):

        # Check whether a valid move is being performed. Return if not
        if self.x + x_step > self.get_max_x() or self.x + x_step < 0:
            return

        if self.y + y_step > self.get_max_y() or self.y + y_step < 0:
            return

        self.x += x_step
        self.y += y_step

    def get_direction(self):
        return self.direction

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y


class BlobEnv:
    """
    Blob env is the environment the Blob lives inside
    """

    def __init__(
        self,
        size: int = 15,
        move_reward: float = -1,
        mowed_reward: float = -100,
        unmowed_reward: float = 50,
        change_direction_reward: float = -4,
        failed_move_reward: float = -100,
        final_reward: float = 10000,
        percent_mowed_to_end: float = 0.99,
        initial_mower_x: int = 0,
        initial_mower_y: int = 0,
        initial_mower_direction: str = "E",
        render_waitkey: int = 5,
    ):
        assert 0 < percent_mowed_to_end <= 1, (
            "Percent_mowed_to_end should be a decimal representing the"
            "percentage of the field to mow before learning is halted"
        )
        self.size = size
        self.move_reward = move_reward
        self.mowed_reward = mowed_reward
        self.unmowed_reward = unmowed_reward
        self.final_reward = final_reward
        self.percent_mowed_to_end = percent_mowed_to_end
        self.change_direction_reward = change_direction_reward
        self.failed_move_reward = failed_move_reward
        self.render_waitkey = render_waitkey
        self.strSQLiteDB = f"simple_lawnmower_sjhatfield.sqlite"

        if initial_mower_x != None:
            self.initial_mower_x = initial_mower_x
        else:
            self.initial_mower_x = 0

        if initial_mower_y != None:
            self.initial_mower_y = initial_mower_y
        else:
            self.initial_mower_y = 0

        if initial_mower_direction != None:
            self.initial_mower_direction = initial_mower_direction
        else:
            self.initial_mower_direction = choice(POSSIBLE_DIRECTIONS)

        self.keys = {"unmowed": -1, "mowed": -2, "lawnmower": -3, "obstacle": -4}

        # Environment is an array with state of each square given by an integer
        self.board = np.zeros((self.size, self.size), dtype=np.int16)
        # self.board = np.zeros((self.get_size(), self.get_size(), 3), dtype=np.uint8)

        # These are the colors for rendering
        self.colors = {
            "lawnmower": np.array(
                [255, 51, 255], dtype=np.uint8
            ),  # pink indicates lawnmower
            "unmowed": np.array(
                [0, 51, 25], dtype=np.uint8
            ),  # dark green indicates unmowed lawn
            "mowed": np.array(
                [102, 255, 102], dtype=np.uint8
            ),  # light green indicates mowed lawn
            "obstacle": np.array(
                [210, 105, 30], dtype=np.uint8
            ),  # brown indicates obstacle
            "N": np.array([255, 25, 25], dtype=np.uint8),  # red
            "NE": np.array([255, 255, 102], dtype=np.uint8),  # yellow
            "E": np.array([230, 115, 0], dtype=np.uint8),  # orange
            "SE": np.array([179, 255, 191], dtype=np.uint8),  # mint
            "S": np.array([77, 195, 255], dtype=np.uint8),  # light blue
            "SW": np.array([0, 21, 128], dtype=np.uint8),  # dark blue
            "W": np.array([255, 255, 255], dtype=np.uint8),  # white
            "NW": np.array([0, 0, 0], dtype=np.uint8),  # black
        }

        #self.reset()

    def set_render_waitkey(self, render_waitkey):
        self.render_waitkey = render_waitkey

    def get_render_waitkey(self):
        return self.render_waitkey

    def get_db_filename(self):
        return self.strSQLiteDB

    def reset(self) -> None:
        """
        Makes the whole field unmowed and returns the
        Blob to where it started
        """
        self.board = np.zeros((self.size, self.size), dtype=np.int16)
        self.squares_to_mow = int(self.size * self.size * self.percent_mowed_to_end) - 1

        # Set the board to unmowed
        self.board[:, :] = self.keys["unmowed"]

        # populate with all obstacles
        # TESTING
        if self.size == 8:
            self.board[4, 4] = self.keys["obstacle"]
            self.board[4, 5] = self.keys["obstacle"]
            self.board[5, 4] = self.keys["obstacle"]
            self.board[5, 5] = self.keys["obstacle"]
            self.squares_to_mow -= 4

        # Create a mower
        self.mower = Blob(
            env_size=self.size,
            x=self.initial_mower_x,
            y=self.initial_mower_y,
            direction=self.initial_mower_direction,
        )
        self.mower.set_max_x(self.get_size() - 1)
        self.mower.set_max_y(self.get_size() - 1)

        # Set the mower's starting point to the direction integer
        self.board[self.mower.get_y(), self.mower.get_x()] = POSSIBLE_DIRECTIONS.index(
            self.mower.get_direction()
        )

        self.episode_step = 0

        self.remaining_squares_to_mow = self.squares_to_mow

        return self.board, 0, False

    def get_size(self):
        return self.size

    def set_size(self, size):
        self.size = size

    def get_lawn_name(self):
        return self.lawn_name

    def set_lawn_name(self, lawn_name):
        self.lawn_name = lawn_name

    def get_episode_step(self):
        return self.episode_step

    def step(
        self, action: int, render: bool = False, debug_render: bool = False
    ) -> (np.array, float, bool):
        """
        Moves the mower in the environment according to the chosen action
        and returns the reward for the action, new state of the environment and
        whether the mower has completed the task
        """
        assert action in [0, 1, 2], (
            "Choice must be one of 0: move forward, "
            "1: turn left (counterclockwise), 2: turn right (clockwise)"
        )
        state = self.board.copy()
        self.episode_step = self.episode_step + 1

        # Perform the movement and update the environment
        mower_x, mower_y = self.mower.get_x(), self.mower.get_y()
        self.mower.action(action)
        new_mower_x, new_mower_y = self.mower.get_x(), self.mower.get_y()
        # go back if we hit an obstacle
        #if self.board[mower_y, mower_x] == self.keys["obstacle"]:
        #    new_mower_x = mower_x
        #    new_mower_y = mower_y
        self.board[mower_y, mower_x] = self.keys["mowed"]
        self.board[new_mower_y, new_mower_x] = POSSIBLE_DIRECTIONS.index(
            self.mower.get_direction()
        )

        # If mower just turned then no need to continue
        if action in [1, 2]:
            return (
                self.board,
                self.change_direction_reward,
                self.remaining_squares_to_mow <= 0,
            )

        # This penalizes the mower on each time step regardless of whether they
        # move or not. Could be changed to only penalize if they move space
        reward = self.move_reward

        # Reward (positive) if mowed a piece of grass that was unmowed
        if (state[new_mower_y, new_mower_x] == self.keys["unmowed"]).all():
            reward += self.unmowed_reward
            self.remaining_squares_to_mow -= 1
        # Reward (negative) if mowed an already mowed square of grass
        elif (state[new_mower_y, new_mower_x] == self.keys["mowed"]).all():
            reward += self.mowed_reward

        # Check if mower did not move
        if new_mower_x == mower_x and new_mower_y == mower_y:
            reward += self.failed_move_reward

        if render:
            self.render()

        if debug_render:
            self.debug_render()

        return self.board, reward, self.remaining_squares_to_mow <= 0

    def debug_render(self):
        img = cv2.resize(
            self.get_img_direction()[..., ::-1].copy() / 255.0,
            (300, 300),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow("Mower Environment Debug Render", img)
        if self.get_render_waitkey() > 0:
            cv2.waitKey(self.get_render_waitkey())

    def render(self):
        img = cv2.resize(
            self.get_img()[..., ::-1].copy() / 255.0,
            (300, 300),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow("Mower Environment Render", img)
        if self.get_render_waitkey() > 0:
            cv2.waitKey(self.get_render_waitkey())

    # Return the numpy array of RGB values for use of the CNN
    def get_img_direction(self):
        img = np.zeros((self.get_size(), self.get_size(), 3), dtype=np.uint8)
        mower_idx = np.where(self.board >= 0)
        mower_direction = self.mower.get_direction()
        img[self.board == self.keys["mowed"], :] = self.colors["mowed"]
        img[self.board == self.keys["unmowed"], :] = self.colors["unmowed"]
        img[self.board == self.keys["obstacle"], :] = self.colors["obstacle"]
        if mower_direction == "N":
            img[mower_idx[0], mower_idx[1], :] = self.colors["N"]
        elif mower_direction == "NE":
            img[mower_idx[0], mower_idx[1], :] = self.colors["NE"]
        elif mower_direction == "E":
            img[mower_idx[0], mower_idx[1], :] = self.colors["E"]
        elif mower_direction == "SE":
            img[mower_idx[0], mower_idx[1], :] = self.colors["SE"]
        elif mower_direction == "S":
            img[mower_idx[0], mower_idx[1], :] = self.colors["S"]
        elif mower_direction == "SW":
            img[mower_idx[0], mower_idx[1], :] = self.colors["SW"]
        elif mower_direction == "W":
            img[mower_idx[0], mower_idx[1], :] = self.colors["W"]
        elif mower_direction == "NW":
            img[mower_idx[0], mower_idx[1], :] = self.colors["NW"]
        return img

    # old version
    def get_img(self):
        img = np.zeros((self.get_size(), self.get_size(), 3), dtype=np.uint8)
        mower_idx = self.board >= 0
        img[self.board == self.keys["mowed"], :] = self.colors["mowed"]
        img[self.board == self.keys["unmowed"], :] = self.colors["unmowed"]
        img[self.board == self.keys["lawnmower"], :] = self.colors["lawnmower"]
        return img

    """
    def get_img(self):
        # new version
        img = np.zeros((self.get_size(), self.get_size(), 3), dtype=np.uint8)
        mower_idx = self.board >= 0
        img[self.board == self.keys["mowed"], :] = self.colors["mowed"]
        img[self.board == self.keys["unmowed"], :] = self.colors["unmowed"]
        mower_idx = np.where(self.board >= 0)
        img[mower_idx] = self.colors["lawnmower"]
        return img
    """


if __name__ == "__main__":
    main()
