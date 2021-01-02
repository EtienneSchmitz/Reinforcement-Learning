import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import sys
import itertools

cost_internal_goal = -0.5  # -0.5
cost_internal_w_goal = -0.05  # -0.05
cost_internal_self_thinking = -1  # -1
cost_internal_self_thinking_cc = -0.5
bad_goal = -6.0
good_goal = 9.5
illegal_action = -1
legal_move = -0.05

# parameter values for the algorithm
alpha = 0.01
discount = 0.95
epsilon = 0.01
decay_rate = 0.25
theta = 0.01


# action dictionary
UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3
UPDATE = 4

# T-maze for the task
#   D - B - C
#       |
#       A
# location states are A, B(choice point), C (right arm), D (left arm)
# the empty memory state is E


# epsilon greedy policy for the actor
def policy(Q, epsilon, state, nA):
    random_prob = np.random.uniform(0, 1)
    if(random_prob < epsilon) or (not np.any(Q[state])):
        action = np.random.randint(low=0, high=nA)
    else:
        action = np.argmax(Q[state])
    return action


# the reward alternates between left and right after every trial
def alternate_reward(reward_table):

    if reward_table[5][2] == bad_goal:  # reward was at left
        reward_table[5] = (illegal_action, illegal_action,
                           good_goal, bad_goal, cost_internal_goal)
        reward_table[6] = (illegal_action, illegal_action,
                           good_goal, bad_goal, cost_internal_goal)
        reward_table[7] = (illegal_action, illegal_action,
                           good_goal, bad_goal, cost_internal_self_thinking)
        reward_table[8] = (illegal_action, illegal_action,
                           good_goal, bad_goal, cost_internal_goal)
        reward_table[9] = (illegal_action, illegal_action,
                           good_goal, bad_goal, cost_internal_goal)
    elif reward_table[5][2] == good_goal:  # reward was at right
        reward_table[5] = (illegal_action, illegal_action,
                           bad_goal, good_goal, cost_internal_goal)
        reward_table[6] = (illegal_action, illegal_action,
                           bad_goal, good_goal, cost_internal_goal)
        reward_table[7] = (illegal_action, illegal_action,
                           bad_goal, good_goal, cost_internal_self_thinking)
        reward_table[8] = (illegal_action, illegal_action,
                           bad_goal, good_goal, cost_internal_goal)
        reward_table[9] = (illegal_action, illegal_action,
                           bad_goal, good_goal, cost_internal_goal)
    return reward_table


def take_action(state, action, reward_table):

    transition_table = [
        (5, 0, 0, 0, 1),  # AE
        (6, 1, 1, 1, 1),  # AA
        (7, 2, 2, 2, 1),  # AB
        (8, 3, 3, 3, 1),  # AC
        (9, 4, 4, 4, 1),  # AD
        (5, 5, 10, 15, 7),  # BE
        (6, 6, 11, 16, 7),  # BA
        (7, 7, 12, 17, 7),  # BB
        (8, 8, 13, 18, 7),  # BC
        (9, 9, 14, 19, 7),  # BD
        (10, 10, 10, 10, 13),  # CE
        (11, 11, 11, 11, 13),  # CA
        (12, 12, 12, 12, 13),  # CB
        (13, 13, 13, 13, 13),  # CC
        (14, 14, 14, 14, 13),  # CD
        (15, 15, 15, 15, 19),  # DE
        (16, 16, 16, 16, 19),  # DA
        (17, 17, 17, 17, 19),  # DB
        (18, 18, 18, 18, 19),  # DC
        (19, 19, 19, 19, 19)]  # DD

    next_state = transition_table[state][action]
    reward = reward_table[state][action]
    return next_state, reward


if __name__ == '__main__':

    n_states = 20
    n_actions = 5
    actions_dict = {UP: 'up', DOWN: 'down',
                    RIGHT: 'right', LEFT: 'left', UPDATE: 'update'}
    states_dict = {
        0: 'AE', 1: 'AA', 2: 'AB', 3: 'AC', 4: 'AD',
        5: 'BE', 6: 'BA', 7: 'BB', 8: 'BC', 9: 'BD',
        10: 'CE', 11: 'CA', 12: 'CB', 13: 'CC', 14: 'CD',
        15: 'DE', 16: 'DA', 17: 'DB', 18: 'DC', 19: 'DD'
    }

    reward_table = [(legal_move, illegal_action, illegal_action, illegal_action, cost_internal_goal),  # AE
                    (legal_move, illegal_action, illegal_action,
                     illegal_action, cost_internal_self_thinking),   # AA
                    (legal_move, illegal_action, illegal_action,
                     illegal_action, cost_internal_goal),  # AB
                    (legal_move, illegal_action, illegal_action,
                     illegal_action, cost_internal_goal),  # AC
                    (legal_move, illegal_action, illegal_action,
                     illegal_action, cost_internal_goal),  # AD
                    (illegal_action, illegal_action, good_goal,
                     good_goal, cost_internal_goal),  # BE
                    (illegal_action, illegal_action, good_goal,
                     good_goal, cost_internal_goal),  # BA
                    (illegal_action, illegal_action, good_goal,
                     good_goal, cost_internal_self_thinking),    # BB
                    (illegal_action, illegal_action, good_goal,
                     good_goal, cost_internal_goal),  # BC
                    (illegal_action, illegal_action, good_goal,
                     good_goal, cost_internal_goal),  # BD
                    (illegal_action, illegal_action, illegal_action,
                     illegal_action, cost_internal_w_goal),   # CE
                    (illegal_action, illegal_action, illegal_action,
                     illegal_action, cost_internal_w_goal),   # CA
                    (illegal_action, illegal_action, illegal_action,
                     illegal_action, cost_internal_w_goal),   # CB
                    (illegal_action, illegal_action, illegal_action,
                     illegal_action, cost_internal_self_thinking_cc),    # CC
                    (illegal_action, illegal_action, illegal_action,
                     illegal_action, cost_internal_w_goal),   # CD
                    (illegal_action, illegal_action, illegal_action,
                     illegal_action, cost_internal_w_goal),   # DE
                    (illegal_action, illegal_action, illegal_action,
                     illegal_action, cost_internal_w_goal),   # DA
                    (illegal_action, illegal_action, illegal_action,
                     illegal_action, cost_internal_w_goal),   # DB
                    (illegal_action, illegal_action, illegal_action,
                     illegal_action, cost_internal_w_goal),   # DC
                    (illegal_action, illegal_action, illegal_action,
                     illegal_action, cost_internal_self_thinking)       # DD
                    ]

    num_episodes = 100
    num_runs = 1
    n_steps = 100

    episode_rewards = np.zeros((num_runs, num_episodes))
    episode_lengths = np.zeros((num_runs, num_episodes))
    correct_responses = np.zeros((num_runs, num_episodes))

    for i_run in range(num_runs):

        ncr = 0
        state_values = np.zeros(n_states)
        action_values = np.zeros((n_states, n_actions))

        # trial run
        state = 0
        for t in range(n_steps):

            # actor, take action with highest probability
            action = policy(action_values, epsilon, state, n_actions)
            if state >= 4 and state <= 7:
                remember = action
            next_state, reward = take_action(state, action, reward_table)
            print(states_dict[state], actions_dict[action],
                  states_dict[next_state], reward)

            # critic, caluculate td error
            td_error = reward + discount * \
                state_values[next_state] - state_values[state]
            state_values[state] += alpha*td_error
            action_values[state][action] += alpha*td_error

            state = next_state
            if state == 13 or state == 19:  # states CC and DD
                break

        print("action taken", actions_dict[remember], reward)
        # for the sample trial, going right or left obtains a reward
        if remember == 2:  # right
            reward_table[5] = (illegal_action, illegal_action,
                               good_goal, bad_goal, cost_internal_goal)
            reward_table[6] = (illegal_action, illegal_action,
                               good_goal, bad_goal, cost_internal_goal)
            reward_table[7] = (illegal_action, illegal_action,
                               good_goal, bad_goal, cost_internal_self_thinking)
            reward_table[8] = (illegal_action, illegal_action,
                               good_goal, bad_goal, cost_internal_goal)
            reward_table[9] = (illegal_action, illegal_action,
                               good_goal, bad_goal, cost_internal_goal)
        elif remember == 3:
            reward_table[5] = (illegal_action, illegal_action,
                               bad_goal, good_goal, cost_internal_goal)
            reward_table[6] = (illegal_action, illegal_action,
                               bad_goal, good_goal, cost_internal_goal)
            reward_table[7] = (illegal_action, illegal_action,
                               bad_goal, good_goal, cost_internal_self_thinking)
            reward_table[8] = (illegal_action, illegal_action,
                               bad_goal, good_goal, cost_internal_goal)
            reward_table[9] = (illegal_action, illegal_action,
                               bad_goal, good_goal, cost_internal_goal)

        for i_episode in range(num_episodes):

            print("\rEpisode {}/{}".format(i_episode, num_episodes))
            reward_table = alternate_reward(reward_table)

            if state == 13:  # if state was CC, reset to AC
                state = 3
            elif state == 19:  # if state was DD, reset to AD
                state = 4

            for t in range(n_steps):

                action = policy(action_values, epsilon, state, n_actions)
                next_state, reward = take_action(state, action, reward_table)

                print(states_dict[state], actions_dict[action],
                      states_dict[next_state], reward)
                # if state == 8:
                #    print(action_values[8])
                # if state == 9:
                #    print(action_values[9])

                if reward == good_goal:  # correct response
                    ncr += 1
                correct_responses[i_run][i_episode] = ncr/(i_episode+1)

                episode_rewards[i_run][i_episode] += reward
                episode_lengths[i_run][i_episode] = t

                td_error = reward + discount * \
                    state_values[next_state] - state_values[state]
                state_values[state] += alpha*td_error
                action_values[state][action] += alpha*td_error

                state = next_state
                if state == 13 or state == 19:
                    break

    plt.plot(np.average(episode_rewards, axis=0))
    plt.title('Average rewards over 10 runs')
    plt.show()

    plt.plot(np.average(episode_lengths, axis=0))
    plt.title('Number of steps taken over 10 runs')
    plt.show()

    plt.plot(np.average(correct_responses, axis=0))
    plt.title('Performance')
    plt.show()
