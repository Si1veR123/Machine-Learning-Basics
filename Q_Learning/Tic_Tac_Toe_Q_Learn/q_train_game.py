"""
Tic Tac Toe with Q learning agent vs unbeatable bot
"""
import pygame
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
from random import choice
from Q_Learning.Tic_Tac_Toe_Q_Learn.tic_tac_toe_player_bot import basic_bot
from Q_Learning.Tic_Tac_Toe_Q_Learn.tic_tac_toe_player_bot import medium_bot
from Q_Learning.Tic_Tac_Toe_Q_Learn.tic_tac_toe_player_bot import advanced_rule_bot
from Q_Learning.Tic_Tac_Toe_Q_Learn.q_learn_bot import QLearnBot

player2 = input('Enter player2 (h/b): ')
player_mouse_reset = True

pygame.init()

SIZE = 900

LINE_SPACING = 300
LINES = [x for x in range(SIZE-1) if not x % LINE_SPACING and x]

font = pygame.sysfont.SysFont(pygame.font.get_default_font(), 500)
x_im = font.render('X', False, (0, 0, 0))
y_im = font.render('O', False, (0, 0, 0))

im_dict = {1: x_im, 2: y_im}


def check_win(board: np.ndarray) -> int:
    # check row
    for p in range(0, 2):
        p += 1
        for row in board:
            count = 0
            for col in row:
                if col == p:
                    count += 1
                if count == 3:
                    return p

    # check col
    for p in range(0, 2):
        p += 1
        for col in zip(*board):
            count = 0
            for row in col:
                if row == p:
                    count += 1
                if count == 3:
                    return p

    # diag top left to bottom right
    down_diag = [row[r_count] for r_count, row in enumerate(board)]

    for p in range(0, 2):
        p += 1
        count = 0
        for col in down_diag:
            if col == p:
                count += 1
            if count == 3:
                return p

    # diag bottom left to top right
    up_diag = [row[len(row) - r_count - 1] for r_count, row in enumerate(board)]

    for p in range(0, 2):
        p += 1
        count = 0
        for col in up_diag:
            if col == p:
                count += 1
            if count == 3:
                return p


def check_draw(board):
    for row in board:
        for col in row:
            if col == 0:
                return False
    return True


def get_empty_index(board):
    empty = []
    for r_count, row in enumerate(board):
        for c_count, col in enumerate(row):
            if col == 0:
                empty.append((c_count, r_count))
    indexes = []
    for pos in empty:
        indexes.append(pos[0] + pos[1]*3)

    return indexes


def end():
    QBot.save_model()

    plt.plot(range(len(rewards)), rewards, linewidth=0.7, label='Ave. Reward')
    plt.plot(range(len(min_reward)), min_reward, linewidth=0.7, label='Min. Reward')
    plt.plot(range(len(max_reward)), max_reward, linewidth=0.7, label='Max. Reward')
    plt.plot(range(len(epsilon)), epsilon, linewidth=0.7, label='Epsilon')

    plt.legend(loc=1)

    plt.savefig(f'models/graphs/model-time-{str(time.time())}.png')
    plt.show()


class Board:
    def __init__(self):
        self.board = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        # 0 = empty, 1 = cross, 2 = circle
        self.turn = np.random.randint(1, 3)
        self.placements = 1

    def place(self, pos: Union[tuple, list]):
        pos = pos[::-1]
        self.board[pos] = self.turn

        # pygame.time.wait(800)

        if self.turn == 1:
            self.turn = 2
        elif self.turn == 2:
            self.turn = 1
        else:
            raise ValueError(f"Player number {self.turn} isn't 1 or 2")

        self.placements += 1


DRAW = False
EPISODES = 1_000_000
STATS_EVERY = 10_000

BASIC_CHANCE = 0.2
MID_CHANCE = 0.7
ADVANCED_CHANCE = 1

SAVED_BOT = 'models/model-time-1582644480.5359461.pickle'

if DRAW:
    root = pygame.display.set_mode((SIZE, SIZE))
    pygame.display.set_caption('Tic Tac Toe')
else:
    root = pygame.display.set_mode((50, 50))
    pygame.display.set_caption('Stop and Save')

if SAVED_BOT:
    with open(SAVED_BOT, 'rb') as f:
        print('Loading: ', SAVED_BOT)
        QBot = pickle.load(f)
        QBot.epsilon = QBot.START_EPSILON
else:
    QBot = QLearnBot()


rewards_per_stat_show = []
# these lists will store data that will plotted at the end.
# data will be added every STATS_EVERY
rewards = []
epsilon = []
min_reward = []
max_reward = []


for ep in range(EPISODES):
    board = Board()

    prev = {}

    current_bot = None

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        if DRAW:
            root.fill((255, 255, 255))
            for line in LINES:
                pygame.draw.line(root, (0, 0, 0), (0, line), (SIZE, line), 3)
                pygame.draw.line(root, (0, 0, 0), (line, 0), (line, SIZE), 3)

            for r_count, row in enumerate(board.board):
                for c_count, col in enumerate(row):
                    if col:
                        pos = (c_count*LINE_SPACING, r_count*LINE_SPACING)
                        root.blit(im_dict[col], pos)
        else:
            root.fill((255, 0, 0))
            if pygame.mouse.get_pressed()[0]:
                end()
                quit()

        if board.turn == 1:
            overlay = False

            prev_state = board.board

            if np.random.random(1) > QBot.epsilon:
                action = QBot.best_action(prev_state)
            else:
                action = choice(get_empty_index(board.board))

            if QBot.action(action) in prev.values():
                overlay = True
            prev[board.placements] = QBot.action(action)

            board.place(QBot.action(action))

            new_state = board.board

            player_won = check_win(board.board)
            if player_won:
                reward_info = 1 if player_won == 1 else 2
            else:
                reward_info = 0
            if overlay:
                reward_info = 3

            reward = QBot.reward(reward_info)
            rewards_per_stat_show.append(reward)

            QBot.new_q(prev_state, new_state, reward, action)

            if reward == QBot.WIN_REWARD or reward == QBot.LOSE_PENALTY or check_draw(board.board) or overlay:
                break

        if player_mouse_reset is False and not pygame.mouse.get_pressed()[0]:
            player_mouse_reset = True

        elif board.turn == 2:
            if player2 == 'b':
                if current_bot is None:
                    num = np.random.random()
                    if num < BASIC_CHANCE:
                        current_bot = 'basic'
                    elif BASIC_CHANCE < num < MID_CHANCE:
                        current_bot = 'mid'
                    else:
                        current_bot = 'hard'

                if current_bot == 'basic':
                    bot_turn = basic_bot(2, board.board, 1)
                elif current_bot == 'mid':
                    bot_turn = medium_bot(board.board, 2, 1)
                else:
                    bot_turn = advanced_rule_bot(prev, board.placements, board.board, 2, 1)

                prev[board.placements] = bot_turn
                board.place(bot_turn)

            elif player2 == 'h':
                if player_mouse_reset:
                    mouse_left = pygame.mouse.get_pressed()[0]
                    if mouse_left:
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        mouse_x_box = mouse_x // LINE_SPACING
                        mouse_y_box = mouse_y // LINE_SPACING
                        turn = (mouse_x_box, mouse_y_box)
                        prev[board.placements] = turn
                        board.place(turn)
                        player_mouse_reset = False

        player_won = check_win(board.board)

        if player_won:
            run = False
            break

        if check_draw(board.board):
            run = False
            break

        pygame.display.flip()

    QBot.decay_epsilon(ep)

    if ep % STATS_EVERY == 0:

        rewards.append(np.mean(rewards_per_stat_show))
        min_reward.append(min(rewards_per_stat_show))
        max_reward.append(max(rewards_per_stat_show))
        epsilon.append(QBot.epsilon*200)

        rewards_per_stat_show = []

        print(f"""
#Ave. Reward: {np.mean(rewards[-STATS_EVERY:])}
#Epsilon: {QBot.epsilon}
#Episode: {ep}
            """)

end()
