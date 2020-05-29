import gym
import numpy as np

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
# how important future rewards are, compared to current rewards
EPISODES = 25000
# measure of how often to do a 'random' action and explore
epsilon = 0.5
# we dont want to always do a random action. decay lowers the amount over time
START_EPSILON_DECAYING = 1
# end decay halfway through
END_EPSILON_DECAYING = EPISODES//2
# amount to decrement by
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING-START_EPSILON_DECAYING)

RENDER_FREQ = 1000

"""
Q Learning is like a table, where you look up the state variables and get an action.
However, the observations are very accurate numbers, and getting a action for every one would be inefficient
We can group the variables to make them smaller and 'discrete'.
DISCRETE_OS_SIZE is the 'chunks' to round the observations to
length of observation_space.high is the number of observations
"""

DISCRETE_OS_SIZE = [20]*len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

print(
    f"""
Highest Observations: {env.observation_space.high}
Lowest Observations: {env.observation_space.low}
Observations in chunks: {discrete_os_win_size}
    """
      )

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

"""
creates a 20x20x3 table (DISCRETE_OS_SIZE is 20x20 list, there are 3 actions for this environment)
20x20 is every combination of every observation from the environment
x3 is what action to take for each combination of observations
Q values are random to start with
Refer to notes for 2D table
"""

def get_discrete_state(state):
    """
    :param state: observations of current state as np array
    :return: the state as discrete values (rounded to the nearest chunk) as tuple
    """
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

for ep in range(EPISODES):
    if ep % RENDER_FREQ == 0:
        render = True
    else:
        render = False

    # env.reset returns the starting state
    discrete_state = get_discrete_state(env.reset())

    done = False
    while not done:
        action = np.argmax(q_table[discrete_state])
        # get the action for the current state

        if render:
            env.render()

        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            # get highest Q value
            current_q = q_table[discrete_state + (action, )]
            # get q value of action
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            # new q value to update with using the q learning formula
            q_table[discrete_state + (action, )] = new_q
            # update the action that we took previously to the new q value. this isn't the new discrete state

        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action, )] = 0
            print('Finished')
            # if we have finished, update q value to 0 (a reward)

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= ep >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

env.close()
