from itertools import product, chain
import numpy as np
import pickle, time

def rotate_matrix(matrix):
    all = []
    current = []

    assert len(matrix) == 9
    matrix = np.reshape(matrix, (3, 3))

    for x in range(len(matrix)):
        for row in matrix:
            current.append(row[x])
        all.append(current)
        current = []

    for row in all:
        row.reverse()

    all = list(chain.from_iterable(all))
    return all


def flip_horizontal(matrix):
    assert len(matrix) == 9
    matrix = np.reshape(matrix, (3, 3))

    cols_matrix = list(zip(*matrix))
    flipped = []

    for col in cols_matrix[::-1]:
        flipped.append(col)

    matrix = zip(*flipped)
    matrix = list(chain.from_iterable(matrix))
    return matrix


def flip_vertical(matrix):
    assert len(matrix) == 9
    matrix = np.reshape(matrix, (3, 3))

    flipped = []

    for row in matrix[::-1]:
        flipped.append(row)

    flipped = list(chain.from_iterable(flipped))
    return flipped


class QLearnBot:
    def __init__(self):
        self.LEARNING_RATE = 0.15
        self.DISCOUNT = 0.9
        self.START_EPSILON = 0.7
        self.epsilon = self.START_EPSILON
        self.START_EPSILON_DECAY = 200_000
        self.STOP_EPSILON_DECAY = 7_000_000
        self.EPSILON_DECREASE_RATE = self.epsilon/(self.STOP_EPSILON_DECAY-self.START_EPSILON_DECAY)

        self.WIN_REWARD = 200
        self.LOSE_PENALTY = -100
        self.OVERLAY_PENALTY = -200
        self.TIME_PENALTY = -5

        self.q_table = {}

        one_dimensional_list = [0, 1, 2]

        one_dimensional_combinations = list(
            product(one_dimensional_list, one_dimensional_list, one_dimensional_list)
        )

        for x in one_dimensional_combinations:
            for y in one_dimensional_combinations:
                for z in one_dimensional_combinations:
                    state = tuple(chain.from_iterable([x, y, z]))
                    if self.get_q_table(state) is None:
                        self.q_table[state] = np.random.uniform(-5, 0, 9)

    def get_q_table(self, k, return_key=False):
        """
        Searches through the Q Table for rotated or flipped versions of the given states. it does this by:
        for normal, horizontally flipped, vertically flipped and both:
            check if it is in Q Table
            if it is, get values and reverse their changes, to reformat to the given states
            if it isn't, rotate and check up to 4 times

        :param k: states to check for
        :param return_key: if True, returns the key, rotational, function data and Q values
        :return: None if not found, else, the Q values of given state
        """
        original = k

        funcs = [None, flip_vertical, flip_horizontal, [flip_vertical, flip_horizontal]]
        current_func = 0

        q_vals_for_state = None

        found = False
        for func in funcs:  # iterate over flipping and rotation functions
            current_rotation = 0

            for _ in range(4):  # rotate 4 times and check
                try:
                    q_vals_for_state = self.q_table[tuple(k)]
                    if return_key:
                        return tuple(k), current_func, current_rotation
                    found = True
                    break
                except KeyError:
                    k = rotate_matrix(k)
                    current_rotation += 1

            if found:
                break

            if func is not None:
                if type(func) == list:
                    k = func[0](original)
                    k = func[1](k)

                else:
                    k = func(original)
            else:
                k = original

            current_func += 1

        else:
            return None

        if current_rotation != 0:
            for _ in range(current_rotation):
                q_vals_for_state = rotate_matrix(q_vals_for_state)
        if funcs[current_func] is not None:
            if type(funcs[current_func]) == list:
                q_vals_for_state = funcs[current_func][0](q_vals_for_state)
                q_vals_for_state = funcs[current_func][1](q_vals_for_state)
            else:
                q_vals_for_state = funcs[current_func](q_vals_for_state)

        return q_vals_for_state

    def set_q_table(self, key, q_index, data, rotations, functions, use_index=False):
        """
        updates the q_table[key][q_index] to the given data after it is rotated and flipped
        :param key: key of q table to update
        :param q_index: list index for table key
        :param data: data to set (unrotated and unflipped)
        :param rotations: rotations
        :param functions: functions
        :param use_index: whether to use q_index
        """
        funcs = [None, flip_vertical, flip_horizontal, [flip_vertical, flip_horizontal]]

        if rotations != 0:
            for _ in range(rotations):
                data = rotate_matrix(data)

        if funcs[functions] is not None:
            if type(funcs[functions]) == list:
                data = funcs[functions][0](data)
                data = funcs[functions][1](data)
            else:
                data = funcs[functions](data)

        if use_index:
            self.q_table[key][q_index] = data
        else:
            self.q_table[key] = data

    def action(self, action):
        actions = {
            0: (0, 0),
            1: (0, 1),
            2: (0, 2),
            3: (1, 0),
            4: (1, 1),
            5: (1, 2),
            6: (2, 0),
            7: (2, 1),
            8: (2, 2)
        }

        return actions[action]

    def best_action(self, board):
        obs = self.get_obs(board)
        return np.argmax(self.get_q_table(obs))

    def get_obs(self, board):
        return tuple(chain.from_iterable(board))

    def reward(self, reward_info: int) -> int:
        """
        :param reward_info: 0 if no one has won
                            1 if we have won
                            2 if we have lost
                            3 if bot went over existing block
        :return: reward
        """
        if reward_info == 1:
            reward = self.WIN_REWARD
        elif reward_info == 2:
            reward = self.LOSE_PENALTY
        elif reward_info == 3:
            reward = self.OVERLAY_PENALTY
        else:
            reward = self.TIME_PENALTY

        return reward

    def new_q(self, old_board, board, reward, action):
        old_obs = self.get_obs(old_board)
        new_obs = self.get_obs(board)
        max_future_q = np.max(self.get_q_table(new_obs))
        current_q = np.max(self.get_q_table(old_obs)) # < ------
        if reward == self.WIN_REWARD:
            new_q = 0
        else:
            new_q = (1-self.LEARNING_RATE) * current_q + self.LEARNING_RATE * (reward + self.DISCOUNT * max_future_q)

        new_q_vals = self.get_q_table(old_obs)
        new_q_vals[action] = new_q

        data, f, r = self.get_q_table(old_obs, True)
        self.set_q_table(data, action, new_q_vals, r, f)

    def decay_epsilon(self, ep: int):
        if self.START_EPSILON_DECAY < ep < self.STOP_EPSILON_DECAY:
            self.epsilon -= self.EPSILON_DECREASE_RATE

    def save_model(self):
        with open(f'models/model-time-{str(time.time())}.pickle', 'wb') as f:
            pickle.dump(self, f)


"""
bot = QLearnBot()

d, f, r = bot.get_q_table((0, 1, 0,
                           0, 1, 0,
                           1, 1, 0), True)
d = key
f = functions from funcs list to get to the key, from the given data
r = rotations to get from given data to found key

new_q = [-0.09154845,  -3.4186234,  -4.28697418, 
         -3.35937076, -1.55733377, -0.24098873, 
         -2.87850518, -0.30364944, -2.72069107]

bot.set_q_table(d, 0, new_q, r, f, use_index=False)


print(bot.get_q_table(d))
"""
