import random
import numpy as np
from Game import Game
from colorist import Color

BOARD_SIZE = 5

def generate_board(side_length: int = BOARD_SIZE, mirror: bool = False, flip: bool = False):
  """ Generates a hex board with side_length tiles on each side,
  using axial coordinates.

  For example::

    >>> print_board(generate_board())
            ( 0  2) ( 1  2) ( 2  2)

        (-1  1) ( 0  1) ( 1  1) ( 2  1)

    (-2  0) (-1  0) ( 0  0) ( 1  0) ( 2  0)

        (-2 -1) (-1 -1) ( 0 -1) ( 1 -1)

            (-2 -2) (-1 -2) ( 0 -2)
  """
  board_height = 2 * side_length - 1

  start, stop, step = [
    (side_length - 1, -side_length, -1),
    (-side_length + 1, side_length, 1),
  ][flip]

  for r in range(start, stop, step):
    row_length = board_height - abs(r)
    row_start = max(
      1 - side_length,
      r - (side_length - 1),
    )
    start, stop, step = [
      (row_start, row_start + row_length, 1),
      (row_start + row_length- 1, row_start - 1, -1),
    ][mirror]
    for q in range(start, stop, step):
      yield (q, r)

def _print_board_iter(board, width: int, side_length: int = BOARD_SIZE):
  height = 2 * side_length - 1
  for row in range(height):
    row_length = height - abs(side_length - row - 1)
    print(" "  * ((height - row_length) * width), end="")
    for col in range(row_length):
      print(next(board), end="")
    print()

def _gex_axial_to_index_map(board_iter):
  ax_to_index = {}
  index_to_ax = {}
  for i, (q, r) in enumerate(board_iter):
    ax_to_index[(q, r)] = i
    index_to_ax[i] = (q, r)
  return ax_to_index, index_to_ax

ax_to_ix, ix_to_ax = _gex_axial_to_index_map(generate_board())

ix_flip_map = [
  ax_to_ix[qr]
  for qr in generate_board(flip=True, mirror=True)
]

def adjacent_idxs(idx: int, player_idx: int):
    q, r = ix_to_ax[idx]
    adj_coords = [
        ((q + 1, r), True),
        ((q - 1, r), True),
        ((q, r + 1), player_idx == 1),
        ((q, r - 1), player_idx == 0),
        ((q + 1, r + 1), player_idx == 1),
        ((q - 1, r - 1), player_idx == 0),
    ]
    res = []
    for coord, valid in adj_coords:
        if not valid:
            continue
        idx = ax_to_ix.get(coord)
        if idx is None:
            continue
        if idx == PLAYER_IDX_CITY_IDXS[player_idx]:
            continue
        res.append(idx)
    return res

def flood_fill(board: np.ndarray[int, int], player_idx: int, start_idx: int, count: int):
    valid_moves = []
    # Players can't move coins into their own city
    visited = set([PLAYER_IDX_CITY_IDXS[player_idx]])
    queue = [(start_idx, count, True)]
    while queue:
        idx, count, is_first = queue.pop()
        if idx in visited:
            continue
        visited.add(idx)

        if board[idx] <= 0:
            # If the tile is empty or owned by the opponent, it's a valid move
            valid_moves.append(idx)

        # Continue moving if:
        # - count > 0, and
        # - the current tile is empty
        if count > 0 and (is_first or board[idx] == 0):
            queue.extend(
                (adj_idx, count - 1, False)
                for adj_idx in adjacent_idxs(idx, player_idx)
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

"""
Board layout:
                  ( 0  4) ( 1  4) ( 2  4) ( 3  4) ( 4  4)
                    0/60    1/59    2/58    3/57    4/56

              (-1  3) ( 0  3) ( 1  3) ( 2  3) ( 3  3) ( 4  3)
                5/55    6/54    7/53    8/52    9/51   10/50

          (-2  2) (-1  2) ( 0  2) ( 1  2) ( 2  2) ( 3  2) ( 4  2)
           11/49   12/48   13/47   14/46   15/45   16/44   17/43

      (-3  1) (-2  1) (-1  1) ( 0  1) ( 1  1) ( 2  1) ( 3  1) ( 4  1)
       18/42   19/41   20/40   21/39   22/38   23/37   24/36   25/35

  (-4  0) (-3  0) (-2  0) (-1  0) ( 0  0) ( 1  0) ( 2  0) ( 3  0) ( 4  0)
   26/34   27/33   28/32   29/31   30/30   31/29   32/28   33/27   34/26

      (-4 -1) (-3 -1) (-2 -1) (-1 -1) ( 0 -1) ( 1 -1) ( 2 -1) ( 3 -1)
       35/25   36/24   37/23   38/22   39/21   40/20   41/19   42/18

          (-4 -2) (-3 -2) (-2 -2) (-1 -2) ( 0 -2) ( 1 -2) ( 2 -2)
           43/17   44/16   45/15   46/14   47/13   48/12   49/11

              (-4 -3) (-3 -3) (-2 -3) (-1 -3) ( 0 -3) ( 1 -3)
               50/10   51/ 9   52/ 8   53/ 7   54/ 6   55/ 5

                  (-4 -4) (-3 -4) (-2 -4) (-1 -4) ( 0 -4)
                   56/ 4   57/ 3   58/ 2   59/ 1   60/ 0
"""

# Board is a 2D array of [coin_count, ..., player0_coin_count, player1_coin_count]
# where the coin_count is positive for player 1, and negative for player -1
_board_len = len(ix_to_ax) + 2

PLAYER_CITY_IDXS = {
    1: ax_to_ix[(2, 4)],
    -1: ax_to_ix[(-2, -4)],
}

PLAYER_IDX_CITY_IDXS = {
    0: PLAYER_CITY_IDXS[1],
    1: PLAYER_CITY_IDXS[-1],
}

_player_starting_idxs = parse_idxs("""
                ( 0  4) ( 1  4) (     ) ( 3  4) ( 4  4)
            (-1  3) ( 0  3) ( 1  3) ( 2  3) ( 3  3) ( 4  3)
        (-2  2) (-1  2) ( 0  2) ( 1  2) ( 2  2) ( 3  2) ( 4  2)
""")
# For south player, not used
#   parse_idxs("""
#           (-4 -2) (-3 -2) (-2 -2) (-1 -2) ( 0 -2) ( 1 -2) ( 2 -2)
#               (-4 -3) (-3 -3) (-2 -3) (-1 -3) ( 0 -3) ( 1 -3)
#                   (-4 -4) (-3 -4) (     ) (-1 -4) ( 0 -4)
#   """)
player_starting_location_mask = np.zeros(_board_len, dtype=bool)
player_starting_location_mask[_player_starting_idxs] = True

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
       int(count)
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

_player_idx = {
    1: 0,
    -1: 1,
}

def assert_sign_eq(a: int, b: int):
    assert (a < 0) == (b < 0), f"assert_sign_eq: {a}, {b}"

_counter = 0

class Board:
    arr: np.ndarray[int, int]

    def __init__(self, board_arr: np.ndarray[int, int]):
        self.arr = board_arr

    @classmethod
    def get_arr(cls) -> np.ndarray[int, int]:
        return np.zeros(_board_len, dtype=np.int8)

    @classmethod
    def get_initial_arr(cls) -> np.ndarray[int, int]:
        global _counter
        arr = cls.get_arr()
        _counter += 1
        arr[-2:] = [15, 16]
        return arr

    def coins_to_add(self, player: int) -> int:
        return self.arr[-2 + _player_idx[player]]

    def coins_to_add_deduct(self, player: int, count: int):
        self.arr[-2 + _player_idx[player]] -= count
        assert self.arr[-2 + _player_idx[player]] >= 0,\
            f"coins_remaining_deduct: {player=}, {count=}, {self.arr[-2 + _player_idx[player]]=}"

    def player_available_starting_idxs(self, player: int) -> np.ndarray[int, int]:
        assert player == 1, f"player_available_starting_idxs: {player}"
        return np.where((self.arr == 0) & player_starting_location_mask)[0]

    def player_coin_idxs(self, player: int) -> np.ndarray[int, int]:
        arr = self.arr[:-2]
        return np.where((arr > 0) if player > 0 else (arr < 0))[0]

    def coins_at_idx(self, *, player: int, idx: int) -> int:
        """ Returns the normalized coin count at `idx` for `player`.

        count > 0: coins are owned by `player`
        count == 0: No coins
        count < 0: coins are owned by opponent
        """
        if player == 1:
            return self.arr[idx]
        return -self.arr[idx]

    def src_idx_player_to_idx(self, player: int, src_idx_player: int) -> int:
        assert player == 1, f"src_idx_player_to_idx: {player}"
        arr = self.arr[:-2]
        player_coins = (arr > 0)
        player_coin_idxs = np.where(player_coins)[0]
        return player_coin_idxs[src_idx_player]

    def coins_deduct(self, player: int, idx: int, count: int):
        """ Deduct `count` coins from the stack owned by `player` at `idx` """
        assert (self.arr[idx] < 0) == (player < 0), f"coins_deduct: {player=}, {idx=}, {count=}"
        new_amount = abs(self.arr[idx])
        new_amount -= count
        if new_amount < 0:
            print(f"ERROR: coins_deduct: {player=}, {idx=}, {count=}, {new_amount=}, {self.arr[idx]=}")
            self.display()
            new_amount = 0 # hack: it's unclear why this is happening... hopefully it's not a problem :|
        self.arr[idx] = new_amount * player

    def coins_add(self, player: int, idx: int, count: int):
        """ Add `count` coins to the board location `idx`, where `player` is the
        owner of the coins being moved. """
        current_count = self.arr[idx]

        if (current_count < 0) == (player < 0):
            # Coins are being added to the player's existing stack
            self.arr[idx] += (count * player)
        else:
            # Coins are being added to an empty stack, or capturing an
            # opponent's stack.
            self.arr[idx] = count * player

    def coins_on_board(self, player: int) -> int:
        arr = self.arr[:-2]
        return np.abs(arr[(arr > 0) if player > 0 else (arr < 0)]).sum()

    def canonicalize_idx(self, player: int, idx: int) -> int:
        if player == 1:
            return idx
        return ix_flip_map[idx]

    def canonicalize_arr(self, player: int) -> np.ndarray[int, int]:
        if player == 1:
            return self.arr
        res = self.arr.copy()
        res[:-2] = res[ix_flip_map] * -1
        res[-2], res[-1] = res[-1], res[-2]
        return res

    def display(self):
        board_bits = iter([
            f"{Color.RED}{v}{Color.OFF} " if v > 0 else
            f"{Color.BLUE}{abs(v)}{Color.OFF} " if v < 0 else
            "_ "
            for v in self.arr
        ])
        _print_board_iter(
            board_bits,
            width=1,
        )
        print("Player 1:", self.coins_to_add(1))
        print("Player 2:", self.coins_to_add(-1))

class GameWin(Exception):
    def __init__(self, action: int):
        self.action = action

class JGGame(Game):
    def getInitBoard(self):
        # return initial board (numpy board)
        return Board.get_initial_arr()

    def getBoardSize(self):
        return Board.get_arr().shape + (1,)

    def getActionSize(self):
        return 2 ** (4 + 6 + 4)

    def getNextState(self, board_arr: np.ndarray[int, int], player: int, action: int):
        #print("BEFORE getNextState")
        #print("Action:", action_unpack(action))
        #Board(board_arr).display()
        res = self._getNextState(board_arr, player, action)
        #print("AFTER getNextState")
        #Board(res[0]).display()
        return res

    def _getNextState(self, board_arr: np.ndarray[int, int], player: int, action: int):
        if action == 0b11111111111111:
            return board_arr, -player

        skip, src_idx_player, dst_idx, count = action_unpack(action)

        board = Board(board_arr.copy())
        remaining_to_add = board.coins_to_add(player)
        if remaining_to_add > 0:
            board.coins_to_add_deduct(player, count)
        else:
            src_idx = board.src_idx_player_to_idx(player, src_idx_player)
            board.coins_deduct(player, src_idx, count)

        board.coins_add(player, dst_idx, count)

        player_remaining = board.coins_to_add(player)
        opponent_remaining = board.coins_to_add(-player)
        if player_remaining or opponent_remaining:
            # First player plays all their coins first, then the opponent plays all their coins
            res = board.arr, (
                player # We have coins remaining to add
                if player_remaining
                else -player # We've played all our coins, it's the opponent's turn
            )
            return res
        return board.arr, -player

    def getValidMoves(self, board_arr: np.ndarray[int, int], player: int) -> np.ndarray[bool]:
        try:
            return self._getValidMoves(board_arr, player)
        except GameWin as e:
            #print("Found game win")
            res = np.zeros(self.getActionSize(), dtype=bool)
            res[e.action] = True
            return res

    def _getValidMoves(self, board_arr: np.ndarray[int, int], player: int) -> np.ndarray[bool]:
        actions: list[int] = []
        board = Board(board_arr)

        def add_action(skip: bool, src_idx_player: int, dst_idx: int, count: int):
            if src_idx_player > 0b1111:
                # Hack to prevent invalid moves from being added
                # This should hardly ever happen, but it's a hack
                return
            action = action_pack(skip, src_idx_player, dst_idx, count)
            #self.getNextState(board_arr, player, action)
            if dst_idx == PLAYER_CITY_IDXS[-player]:
                raise GameWin(action)
            #check_board, _ = self.getNextState(board, player, action)
            #if np.any(check_board < 0):
            #    breakpoint()
            actions.append(action)

        coins_to_add = board.coins_to_add(player)
        if coins_to_add > 0:
            # Placing coins on the board
            available_idxs = board.player_available_starting_idxs(player)
            max_coins_to_add = min(0b1111, coins_to_add)
            for available_idx in available_idxs:
                for move_coin_count in range(1, max_coins_to_add + 1):
                    add_action(
                        False,
                        0,
                        available_idx,
                        move_coin_count,
                    )
        else:
            # Moving coins on the board
            # 1. Moves
            player_coin_idxs = board.player_coin_idxs(player)
            for src_idx_player, src_idx in enumerate(player_coin_idxs):
                move_coin_count = board.coins_at_idx(player=player, idx=src_idx)
                assert move_coin_count > 0, f"coin_count: {move_coin_count}"
                valid_move_idxs = flood_fill(
                    board.arr,
                    _player_idx[player],
                    src_idx,
                    move_coin_count,
                )
                for dst_idx in valid_move_idxs:
                    add_action(
                        False,
                        src_idx_player,
                        dst_idx,
                        move_coin_count,
                    )

            # 2. Splits
            for src_idx_player, src_idx in enumerate(player_coin_idxs):
                for adj_idx in adjacent_idxs(src_idx, _player_idx[player]):
                    adj_idx_coins = board.coins_at_idx(player=player, idx=adj_idx)
                    if adj_idx_coins < 0:
                        # Can't split into an opponent's stack
                        continue
                    max_split_count = min(
                        0b1111 - adj_idx_coins,
                        board.coins_at_idx(player=player, idx=src_idx),
                    )
                    for split_count in range(1, max_split_count + 1):
                        add_action(
                            False,
                            src_idx_player,
                            adj_idx,
                            split_count,
                        )

        if not actions:
            breakpoint()
        res = np.zeros(self.getActionSize(), dtype=bool)
        res[actions] = True
        return res

    def getGameEnded(self, board_arr: np.ndarray[int, int], player: int):
        board = Board(board_arr)

        if board.coins_on_board(player) + board.coins_to_add(player) == 0:
            # Player has no coins remaining
            #  print("Game win: player has no coins remaining")
            #  board.display()
            return -1

        if board.coins_on_board(-player) + board.coins_to_add(-player) == 0:
            # Opponent has no coins remaining
            #  print("Game win: opponent has no coins remaining")
            #  print("Player 1:", board.coins_to_add(1), board.coins_on_board(1))
            #  print("Player 2:", board.coins_to_add(-1), board.coins_on_board(-1))
            #  board.display()
            #breakpoint()
            return 1

        if board.coins_at_idx(player=player, idx=PLAYER_CITY_IDXS[player]):
            # Player's city has coins on it
            # print("Game win: player's city has coins on it")
            # board.display()
            return -1

        if board.coins_at_idx(player=player, idx=PLAYER_CITY_IDXS[-player]):
            # Opponent's city has coins on it
            #print("Game win: opponent's city has coins on it")
            #board.display()
            return 1

        return 0

    def getCanonicalForm(self, board_arr: np.ndarray[int, int], player: int):
        return Board(board_arr).canonicalize_arr(player)

    def getSymmetries(self, board: np.ndarray[int, int], pi: np.ndarray[float, int]):
        return [
            (board, pi),
        ]

    def stringRepresentation(self, board_arr: np.ndarray[int, int]):
        return board_arr.tobytes()

    @staticmethod
    def display(board_arr: np.ndarray[int, int]):
        Board(board_arr).display()
