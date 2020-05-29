import numpy as np
import random

BASIC_BLOCK_CHANCE = 0.6
BASIC_WIN_CHANCE = 0.6

MID_BLOCK_CHANCE = 0.9
MID_WIN_CHANCE = 0.9


class BotBadAction(Exception):
    pass


def check_win_possible(p: int, board: np.ndarray, check_for: str = 'win') -> tuple:
    """
    :param p: player number
    :param board: [[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]]
    Board array
    :param check_for: whether to check for win this go, or win in future (e.g. 2 empty spaces, 1 X or O)
    :return: Tuple of (x, y) slice of position to go to win
    """
    assert check_for == 'win' or check_for == 'future'
    
    # check row
    for r_count, row in enumerate(board):
        if 0 in row:
            count = 0
            zero_count = 0
            row_win = False
            for c_count, col in enumerate(row):
                if col == p:
                    count += 1
                elif col != 0:
                    break
                else: # col is 0
                    zero_count += 1

                if (count == 2 and check_for == 'win') or (check_for == 'future' and count == 1 and zero_count == 2):
                    row_win = (list(row).index(0), r_count)
                    break

            if row_win:
                return row_win

    # check col
    for c_count, col in enumerate(zip(*board)):
        if 0 in col:
            count = 0
            zero_count = 0
            col_win = False
            for r_count, row in enumerate(col):
                if row == p:
                    count += 1
                elif row != 0:
                    break
                else:
                    zero_count += 1

                if (count == 2 and check_for == 'win') or (check_for == 'future' and count == 1 and zero_count == 2):
                    col_win = (c_count, list(col).index(0))
                    break

            if col_win:
                return col_win

    # diag top left to bottom right
    down_diag = [row[r_count] for r_count, row in enumerate(board)]

    if 0 in down_diag:
        count = 0
        zero_count = 0
        diag_win = False
        for c_count, col in enumerate(down_diag):
            if col == p:
                count += 1
            elif col != 0:
                break
            else:
                zero_count += 1

            if (count == 2 and check_for == 'win') or (check_for == 'future' and count == 1 and zero_count == 2):
                empty_pos = down_diag.index(0)
                diag_win = (empty_pos, empty_pos)
                break

        if diag_win:
            return diag_win

    # diag bottom left to top right
    up_diag = [row[len(row) - r_count - 1] for r_count, row in enumerate(board)]

    if 0 in up_diag:
        count = 0
        zero_count = 0
        diag_win = False
        for c_count, col in enumerate(up_diag):
            if col == p:
                count += 1
            elif col != 0:
                break
            else:
                zero_count += 1

            if (count == 2 and check_for == 'win') or (check_for == 'future' and count == 1 and zero_count == 2):
                empty_pos = up_diag.index(0)
                diag_win = (2-empty_pos, empty_pos)
                break
        if diag_win:
            return diag_win


def get_adjacent_from_corner(pos: tuple) -> tuple:
    """
    :param pos: position to get adjacent places of
    :return: tuple of 1 adjacent place
    """
    adjs = {
        (0, 0): [(1, 0), (0, 1)],
        (2, 0): [(1, 0), (2, 1)],
        (0, 2): [(0, 1), (1, 2)],
        (2, 2): [(2, 1), (1, 2)]
            }

    assert pos in adjs.keys()

    return random.choice(adjs[pos])


def get_adjacent_from_sides(pos1, pos2):
    adjs = {
        ((1, 0), (0, 1)): (0, 0),
        ((1, 0), (2, 1)): (2, 0),
        ((0, 1), (1, 2)): (0, 2),
        ((2, 1), (1, 2)): (2, 2)
    }

    for x in adjs.items():
        k, v = x
        if pos1 in k and pos2 in k:
            return v


def get_any_space(board, pos_random=False):
    empty = []
    for r_count, row in enumerate(board):
        for c_count, col in enumerate(row):
            if not col and not pos_random:
                return c_count, r_count
            elif not col:
                empty.append((c_count, r_count))
    return random.choice(empty)


def basic_bot(p, board, otherp):
    # chance to miss a winning or losing move
    if random.random() < BASIC_WIN_CHANCE:
        winning_move = check_win_possible(p, board)
        if winning_move:
            return winning_move

    if random.random() < BASIC_BLOCK_CHANCE:
        blocking_move = check_win_possible(otherp, board)
        if blocking_move:
            return blocking_move

    return get_any_space(board, pos_random=True)


def medium_bot(board, p, otherp):
    # chance to miss a winning or losing move
    if random.random() < MID_WIN_CHANCE:
        winning_move = check_win_possible(p, board)
        if winning_move:
            return winning_move

    if random.random() < MID_BLOCK_CHANCE:
        blocking_move = check_win_possible(otherp, board)
        if blocking_move:
            return blocking_move

    future_move = check_win_possible(p, board, check_for='future')
    if future_move:
        return future_move

    block_future_move = check_win_possible(p, board, check_for='future')
    if block_future_move:
        return block_future_move

    return get_any_space(board, pos_random=True)


def advanced_rule_bot(prev, turn, board, p, otherp) -> tuple:
    """
    prev = previous turns
    turn = turn number
    board = current board as 2D numpy array
    returns = place to change as a tuple (row, col)
    """
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    edges = [(1, 0), (2, 1), (0, 1), (1, 2)]

    winning_move = check_win_possible(p, board)
    if winning_move:
        return winning_move

    blocking_move = check_win_possible(otherp, board)
    if blocking_move:
        return blocking_move

    if turn == 1: # done
        # go in corner
        return random.choice(corners)

    elif turn == 2: # done
        if prev[1] == (1, 1):
            return random.choice(corners)
        else:
            # if other player starts in corner or sides, go middle
            return 1, 1

    elif turn == 3: # done
        if not board[(1, 1)]: # go in middle if not taken
            return 1, 1
        else:   # else go opposite of first move
            corner_pair_tl_br = [(0, 0), (2, 2)] # top left, bottom right corners
            corner_pair_tr_bl = [(2, 0), (0, 2)] # top right, bottom left corners
            if prev[1] in corner_pair_tl_br:
                first_placement = corner_pair_tl_br.index(prev[1])
                if first_placement:
                    opposite = corner_pair_tl_br[0]
                else:
                    opposite = corner_pair_tl_br[1]
                return opposite

            if prev[1] in corner_pair_tr_bl:
                first_placement = corner_pair_tr_bl.index(prev[1])
                if first_placement:
                    opposite = corner_pair_tr_bl[0]
                else:
                    opposite = corner_pair_tr_bl[1]
                return opposite

    elif turn == 4: # done
        if prev[1] in corners:
            if prev[3] in corners:
                return random.choice(edges)

            if prev[3] in edges:
                possible_move = check_win_possible(p, board, check_for='future')
                if possible_move:
                    return possible_move
                else:
                    return get_any_space(board)

        elif prev[1] in edges:
            h_opposites = [(1, 0), (1, 2)]
            v_opposites = [(0, 1), (2, 1)]

            if (prev[3] in h_opposites and prev[1] in h_opposites) or (prev[3] in v_opposites and prev[1] in v_opposites):
                while True:
                    choice = random.choice(corners)
                    if choice not in prev.values():
                        return choice

            elif prev[3] in edges:
                next_corner = get_adjacent_from_sides(prev[1], prev[3])
                return next_corner

            else: # opponent on corner
                while True:
                    next_side = random.choice(edges)
                    # if the next side isn't opposite 1st side and they arent the same
                    if not (prev[1] in h_opposites and next_side in h_opposites and prev[1] != next_side) and not (prev[1] in v_opposites and next_side in v_opposites and prev[1] != next_side):
                        return next_side

        elif prev[1] == (1, 1): # in middle
            if prev[3] in corners:
                next_place = get_any_space(board)
                if next_place:
                    return next_place
                else:
                    raise BotBadAction('Bot should be blocking #2')
            elif prev[3] in edges:
                raise BotBadAction('Bot should be blocking #3')

    elif turn == 5: # done
        if prev[3] in corners: # bot went on corner in turn 3
            if prev[4] in corners: # enemy went on corner in turn 4
                remaining_corner = corners
                print(remaining_corner, prev)
                remaining_corner.remove(prev[1])
                remaining_corner.remove(prev[3])
                remaining_corner.remove(prev[4])
                return remaining_corner[0] # return remaining corner

        elif prev[3] == (1, 1):
            starting_placement = prev[1]
            adjacent_placement = get_adjacent_from_corner(starting_placement)
            return adjacent_placement

    elif turn == 6:
        future_win = check_win_possible(p, board, check_for='future')
        if future_win:
            return future_win

    # do any leftover place if here (probably 1 space left)
    return get_any_space(board)
