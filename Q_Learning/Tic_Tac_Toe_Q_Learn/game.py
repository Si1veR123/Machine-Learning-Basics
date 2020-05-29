"""
Tic Tac Toe with unbeatable bot
"""
import pygame
from typing import Union
import numpy as np
from Q_Learning.Tic_Tac_Toe_Q_Learn.tic_tac_toe_player_bot import rule_bot
import os

pygame.init()

os.environ['SDL_VIDEO_CENTERED'] = '1'

SIZE = 900

root = pygame.display.set_mode((SIZE, SIZE))
pygame.display.set_caption('Tic Tac Toe')

LINE_SPACING = 300
LINES = [x for x in range(SIZE-1) if not x % LINE_SPACING and x]

font = pygame.sysfont.SysFont(pygame.font.get_default_font(), 500)
x_im = font.render('X', False, (0, 0, 0))
y_im = font.render('O', False, (0, 0, 0))

im_dict = {1: x_im, 2: y_im}

player_mouse_reset = True


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

class Board:
    def __init__(self):
        self.board = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        # 0 = empty, 1 = cross, 2 = circle
        self.turn = np.random.randint(1, 3)
        self.placements = 1

    def place(self, pos: Union[tuple, list]):
        self.board[pos] = self.turn
        if self.turn == 1:
            self.turn = 2
        elif self.turn == 2:
            self.turn = 1
        else:
            raise ValueError(f"Player number {self.turn} isn't 1 or 2")
        self.placements += 1

board = Board()

prev = {}

run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            break

    root.fill((255, 255, 255))

    for line in LINES:
        pygame.draw.line(root, (0, 0, 0), (0, line), (SIZE, line), 3)
        pygame.draw.line(root, (0, 0, 0), (line, 0), (line, SIZE), 3)

    for r_count, row in enumerate(board.board):
        for c_count, col in enumerate(row):
            if col:
                pos = ((c_count)*LINE_SPACING, (r_count)*LINE_SPACING)
                root.blit(im_dict[col], pos)

    player_won = check_win(board.board)
    if player_won:
        print(f'Player {player_won} has won')
        run = False
        break

    if check_draw(board.board):
        print('Draw')
        run = False
        break

    if player_mouse_reset is False and not pygame.mouse.get_pressed()[0]:
        player_mouse_reset = True

    if board.turn == 1 and player_mouse_reset:
        mouse_left = pygame.mouse.get_pressed()[0]
        if mouse_left:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            mouse_x_box = mouse_x//LINE_SPACING
            mouse_y_box = mouse_y//LINE_SPACING
            turn = (mouse_x_box, mouse_y_box)
            prev[board.placements] = turn
            board.place(turn[::-1])
            player_mouse_reset = False

    elif board.turn == 2:
        bot_turn = rule_bot(prev, board.placements, board.board, 2, 1)
        if bot_turn is None:
            run = False
            break
        prev[board.placements] = bot_turn
        board.place(bot_turn[::-1])

    pygame.display.flip()
