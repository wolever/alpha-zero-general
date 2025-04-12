from Game import Game
import numpy as np

BOARD_SIZE = 5

def _gex_axial_to_index_map(side_length: int):
  res = []
  board_height = 2 * side_length - 1
  for r in range(side_length - 1, -side_length, -1):
    row_length = board_height - abs(r)
    row_start = max(
      1 - side_length,
      r - (side_length - 1),
    )
    for q in range(row_start, row_start + row_length):
      res.append((q, r))

  ax_to_index = {}
  index_to_ax = {}
  for i, (q, r) in enumerate(res):
    ax_to_index[(q, r)] = i
    index_to_ax[i] = (q, r)
  return ax_to_index, index_to_ax

ax_to_ix, ix_to_ax = _gex_axial_to_index_map(BOARD_SIZE)

def adjacent_idxs(idx: int):
    q, r = ix_to_ax[idx]
    adj_coords = [
        (q + 1, r),
        (q - 1, r),
        (q, r + 1),
        (q, r - 1),
        (q + 1, r - 1),
        (q - 1, r + 1),
    ]
    res = []
    for coord in adj_coords:
        idx = ax_to_ix.get(coord)
        if idx is not None:
            res.append(idx)
    return res

def flood_fill(board: np.ndarray[int, int], player_idx: int, start_idx: int, count: int):
    valid_moves = []
    # Players can't move coins into their own city
    visited = set([PLAYER_CITY_IDXS[player_idx]])
    queue = [(start_idx, count)]
    while queue:
        idx, count = queue.pop()
        if idx in visited:
            continue
        visited.add(idx)

        if board[idx, player_idx] == 0:
            # If the tile is empty or owned by the opponent, it's a valid move
            valid_moves.append(idx)

        # Continue moving if:
        # - count > 0, and
        # - the current tile is empty
        if count > 0 and np.sum(board[idx]) == 0:
            queue.extend(
                (adj_idx, count - 1)
                for adj_idx in adjacent_idxs(idx)
                if adj_idx not in visited
            )

    return valid_moves

def parse_idxs(positions: str):
    res = []
    for line in positions.splitlines():
        line = line.strip()
        for bit in line.split("("):
            bit = bit.replace(")", "").strip()
            if not bit:
                continue
            q, r = bit.split()
            res.append(ax_to_ix[(int(q), int(r))])
    return res

# Board layout:
#                  ( 0  4) ( 1  4) ( 2  4) ( 3  4) ( 4  4)
#
#              (-1  3) ( 0  3) ( 1  3) ( 2  3) ( 3  3) ( 4  3)
#
#          (-2  2) (-1  2) ( 0  2) ( 1  2) ( 2  2) ( 3  2) ( 4  2)
#
#      (-3  1) (-2  1) (-1  1) ( 0  1) ( 1  1) ( 2  1) ( 3  1) ( 4  1)
#
#  (-4  0) (-3  0) (-2  0) (-1  0) ( 0  0) ( 1  0) ( 2  0) ( 3  0) ( 4  0)
#
#      (-4 -1) (-3 -1) (-2 -1) (-1 -1) ( 0 -1) ( 1 -1) ( 2 -1) ( 3 -1)
#
#          (-4 -2) (-3 -2) (-2 -2) (-1 -2) ( 0 -2) ( 1 -2) ( 2 -2)
#
#              (-4 -3) (-3 -3) (-2 -3) (-1 -3) ( 0 -3) ( 1 -3)
#
#                  (-4 -4) (-3 -4) (-2 -4) (-1 -4) ( 0 -4)

PLAYER_CITY_IDXS = [
    ax_to_ix[(2, 4)],
    ax_to_ix[(-2, -4)],
]

PLAYER_STARTING_IDXS = [
    parse_idxs("""
                    ( 0  4) ( 1  4) (     ) ( 3  4) ( 4  4)
                (-1  3) ( 0  3) ( 1  3) ( 2  3) ( 3  3) ( 4  3)
            (-2  2) (-1  2) ( 0  2) ( 1  2) ( 2  2) ( 3  2) ( 4  2)
    """),
    parse_idxs("""
            (-4 -2) (-3 -2) (-2 -2) (-1 -2) ( 0 -2) ( 1 -2) ( 2 -2)
                (-4 -3) (-3 -3) (-2 -3) (-1 -3) ( 0 -3) ( 1 -3)
                    (-4 -4) (-3 -4) (     ) (-1 -4) ( 0 -4)
    """),
]

def action_pack(skip: bool, src_idx_player: int, dst_idx: int, count: int):
    # Actions are:
    # - skip turn: action == 0b11111111111111
    # - src idx: 4 (index into player tiles)
    # - dst idx: 6
    # - count: 4
    if skip:
        return 0b11111111111111
    assert 0 <= src_idx_player <= 0b1111, f"src_idx_player: {src_idx_player}"
    assert 0 <= dst_idx <= 0b111100, f"dst_idx: {dst_idx}"
    assert 1 <= count <= 0b1111, f"count: {count}"
    return (
       (src_idx_player << 10) |
       (dst_idx << 4) |
       count
    )

def action_unpack(action: int):
    if action == 0b11111111111111:
        return True, 0, 0, 0
    return (
       False, # skip
       (action >> 10) & 0b1111, # src_idx_player
       (action >> 4) & 0b111111, # dst_idx
       (action & 0b1111), # count
    )

def get_board():
    # Board is a 2D array of [[playerID, coinCount], ...]
    # Where the last row is the number of coins the player has left to add to
    # the board (ie, if board[-1, x] > 0, then player x is adding coins to the
    # board)
    return np.zeros((len(ix_to_ax) + 1, 2), dtype=np.int8)

class JGGame(Game):
    def getInitBoard(self):
        # return initial board (numpy board)
        board = get_board()
        board[-1] = [18, 18]
        return board

    def getBoardSize(self):
        return get_board().shape

    def getActionSize(self):
        return 2 ** (4 + 6 + 4)

    def getNextState(self, board: np.ndarray[int, int], player: int, action: int):
        if action == 0b11111111111111:
            return board, -player

        orig_board = board.copy()
        skip, src_idx_player, dst_idx, count = action_unpack(action)

        player_idx = 0 if player < 0 else 1
        opponent_idx = 1 - player_idx

        board = board.copy()

        if board[-1, player_idx] > 0:
            # Player has coins left to add to the board
            board[-1, player_idx] -= count
        else:
            # Coins are moving on the board
            src_idx = np.where(board[:-1, player_idx])[0][src_idx_player]
            board[src_idx, player_idx] -= count
            if board[src_idx, player_idx] < 0:
                breakpoint()

        if board[dst_idx, player_idx] > 0:
            # Coins are being added to an existing stack
            board[dst_idx, player_idx] += count
        else:
            # Coins are being moved to an empty stack, or capturing an
            # opponent's stack.
            board[dst_idx, opponent_idx] = 0
            board[dst_idx, player_idx] = count

        player_coins = bool(board[-1, player_idx])
        opponent_coins = bool(board[-1, opponent_idx])
        if player_coins ^ opponent_coins:
            return board, (
                -player if opponent_coins else player
            )

        return board, -player

    def getValidMoves(self, board: np.ndarray[int, int], player: int) -> np.ndarray[bool]:
        actions: list[int] = []
        player_idx = 0 if player < 0 else 1

        def add_action(skip: bool, src_idx_player: int, dst_idx: int, count: int):
            action = action_pack(skip, src_idx_player, dst_idx, count)
            check_board, _ = self.getNextState(board, player, action)
            if np.any(check_board < 0):
                breakpoint()
            actions.append(action)

        coins_available = board[-1, player_idx]
        if coins_available > 0:
            # Placing coins on the board
            starting_idxs = PLAYER_STARTING_IDXS[player_idx]
            available_idxs = np.where(board[starting_idxs, player_idx] == 0)[0]
            for available_idx in available_idxs:
                for coin_count in range(1, min(0b1111, coins_available + 1)):
                    add_action(
                        False,
                        0,
                        available_idx,
                        coin_count,
                    )
        else:
            # Moving coins on the board
            # 1. Moves
            player_coin_idxs = np.where(board[:-1, player_idx])[0]
            for src_idx_player, src_idx in enumerate(player_coin_idxs):
                coin_count = board[src_idx, player_idx]
                if not coin_count:
                    breakpoint()
                valid_move_idxs = flood_fill(
                    board,
                    player_idx,
                    src_idx,
                    coin_count,
                )
                for idx in valid_move_idxs:
                    add_action(
                        False,
                        src_idx_player,
                        idx,
                        coin_count,
                    )

            # 2. Splits
            opponent_idx = 1 - player_idx
            for src_idx_player, src_idx in enumerate(player_coin_idxs):
                for adj_idx in adjacent_idxs(src_idx):
                    if board[adj_idx, opponent_idx] > 0:
                        continue
                    coin_count = board[src_idx, player_idx]
                    for split_count in range(1, min(0b1111, coin_count + 1)):
                        add_action(
                            False,
                            src_idx_player,
                            adj_idx,
                            split_count,
                        )

        res = np.zeros(self.getActionSize(), dtype=bool)
        res[actions] = True
        return res

    def getGameEnded(self, board: np.ndarray[int, int], player: int):
        player_idx = 0 if player < 0 else 1
        if np.sum(board[PLAYER_CITY_IDXS[player_idx]]) > 0:
            return -1

        opponent_idx = 1 - player_idx
        if np.sum(board[PLAYER_CITY_IDXS[opponent_idx]]) > 0:
            return 1

        if np.sum(board[:, player_idx]) == 0:
            return -1

        if np.sum(board[:, opponent_idx]) == 0:
            return 1

        return 0

    def getCanonicalForm(self, board: np.ndarray[int, int], player: int):
        if player == 1:
            return board
        return board[:, ::-1]

    def getSymmetries(self, board: np.ndarray[int, int], pi: np.ndarray[float, int]):
        return [(board, pi)]

    def stringRepresentation(self, board: np.ndarray[int, int]):
        return board.tobytes()

    @staticmethod
    def display(board: np.ndarray[int, int]):
        print(board)
