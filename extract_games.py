import pickle
import json
import numpy as np
import os
import sys

# Add current directory to path so we can import JGGame
sys.path.append(os.getcwd())

from JGGame import JGGame, Board, action_pack
from Coach import save_examples
import JGGame as JGGameModule
from JGGame import ix_mirror_map

# Database connection logic
import subprocess


def run_query(query):
    # Escape double quotes in query for shell passing if simple
    # But using subprocess list is safer
    # We wrap query in a way that output is json lines
    full_query = f"SELECT row_to_json(t) FROM ({query}) t"
    cmd = [
        "psql",
        "postgresql://jg_admin:jg_admin@localhost:6753/jg",
        "-t",
        "-A",
        "-c",
        full_query,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Query failed: {result.stderr}")

    rows = []
    for line in result.stdout.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def parse_coord(coord_list):
    # Log uses [q, r]
    # We determined q needs to be negated for JGGame Global Frame
    q, r = coord_list
    return (-q, r)


def log_coord_to_canonical_idx(coord, jg_player):
    # 1. Parse Log -> Global JG Coord
    global_coord = parse_coord(coord)

    # 2. Global JG Index
    global_idx = JGGameModule.ax_to_ix.get(global_coord)
    if global_idx is None:
        raise ValueError(f"Invalid global coordinate {global_coord} from {coord}")

    # 3. Canonicalize
    if jg_player == 1:
        # P1 (Top): Canonical is Global
        return global_idx
    else:
        # P2 (Bottom): Canonical is Mirrored
        return ix_mirror_map[global_idx]


def get_action_int_canonical(game, canonical_board_arr, jg_player, move):
    # reconstruct action integer from log move, relative to canonical board (player 1 perspective)
    args = move["payload"]["args"]
    move_type = move["payload"]["type"]

    board = Board(canonical_board_arr)

    # JGGame expects actions to be executed by Player 1 (the canonical player)
    # The canonical board has Current Player's coins at Top (Stack 0..15).

    if move_type == "addCoins":
        # args: [[q, r], {"count": n}]
        coord = args[0]
        count = args[1]["count"]

        dst_idx = log_coord_to_canonical_idx(coord, jg_player)

        # addCoins always comes from 'coins to add' which is virtual src index 0?
        # In JGGame.getValidMoves: add_action(False, 0, available_idx, count)
        # So src_idx_player is 0.

        action = action_pack(False, 0, dst_idx, count)
        return action

    elif move_type == "moveTile":
        # args: [{"src": [q, r], "dst": [q, r]}]
        if isinstance(args, list):
            args = args[0]

        src_coord = args["src"]
        dst_coord = args["dst"]

        src_idx = log_coord_to_canonical_idx(src_coord, jg_player)
        dst_idx = log_coord_to_canonical_idx(dst_coord, jg_player)

        # We need src_idx_player (index into player's coin stacks)
        # In canonical board, player IS 1.
        player_coin_idxs = board.player_coin_idxs(1)

        try:
            src_idx_player = np.where(player_coin_idxs == src_idx)[0][0]
        except IndexError:
            # Debug info
            print("\nError finding src_idx in player stacks")
            print(f"JG Player: {jg_player}")
            print(
                f"Log Src: {src_coord} -> Global: {parse_coord(src_coord)} -> Canon Idx: {src_idx}"
            )
            print(f"Canonical Board P1 Stacks: {player_coin_idxs}")
            board.display()
            raise ValueError(
                f"Player {jg_player} has no coins at {src_coord} (canon idx {src_idx})"
            )

        count = board.coins_at_idx(player=1, idx=src_idx)

        action = action_pack(False, src_idx_player, dst_idx, count)
        return action

    return None


def process_game(game_row):
    log = game_row["log"]
    gameover = game_row["gameover"]
    initial_state = game_row["initialState"]

    winner_str = gameover.get("winner")
    if winner_str == "0":
        # "0" is You. You = -1 in JGGame (Bottom)
        winner_jg = -1
    elif winner_str == "1":
        # "1" is Computer. Computer = 1 in JGGame (Top)
        winner_jg = 1
    else:
        return []

    # Parse Bidding from Log
    p1_coins = 18
    p2_coins = 18  # p0 coins

    p1_bids = []
    p0_bids = []
    moves = []

    if isinstance(log, str):
        log = json.loads(log)

    for entry in log:
        if "action" not in entry:
            continue
        action = entry["action"]
        if action["type"] != "MAKE_MOVE":
            continue

        payload = action["payload"]
        ptype = payload["type"]
        player_id = payload["playerID"]

        if ptype == "makeOpeningBid":
            bid = payload["args"][0]
            if player_id == "1":
                p1_bids.append(bid)
            else:
                p0_bids.append(bid)
        elif ptype in ["addCoins", "moveTile"]:
            moves.append(entry)

    max_p1 = max(p1_bids) if p1_bids else 0
    max_p0 = max(p0_bids) if p0_bids else 0

    # Bidding logic: Highest bid wins and goes first.
    first_player = 0
    if max_p1 > max_p0:
        p1_coins -= max_p1
        first_player = 1
    elif max_p0 > max_p1:
        p2_coins -= max_p0
        first_player = -1
    else:
        # Tie break - check first move
        if moves:
            first_move_player_id = moves[0]["action"]["payload"]["playerID"]
            if first_move_player_id == "1":
                p1_coins -= max_p1
                first_player = 1
            else:
                p2_coins -= max_p0
                first_player = -1
        else:
            return []

    # Initialize JGGame
    g = JGGame()

    # Global Board Array
    global_board_arr = g.getInitBoard()
    global_board_arr[-2] = p1_coins
    global_board_arr[-1] = p2_coins

    current_player = first_player

    history = []

    for entry in moves:
        payload = entry["action"]["payload"]
        player_id = payload["playerID"]
        jg_player = 1 if player_id == "1" else -1

        # Check turn sync
        # If jg_player != current_player, we might have an issue if the engine enforces turns strictly
        # But here we just follow the log. We assume the log is valid.

        # 1. Canonicalize for current player
        canonical_board = g.getCanonicalForm(global_board_arr, jg_player)

        # 2. Get Action
        try:
            action = get_action_int_canonical(
                g, canonical_board, jg_player, entry["action"]
            )
        except ValueError as e:
            print(f"Skipping game due to parse error: {e}")
            return []

        # 3. Store
        # (board, pi, v) - v filled later
        pi = np.zeros(g.getActionSize())
        pi[action] = 1.0
        history.append({"board": canonical_board, "pi": pi, "player": jg_player})

        # 4. Apply Action
        # getNextState must be called with player=1 (canonical)
        next_canon_board, next_canon_player = g.getNextState(canonical_board, 1, action)

        # 5. Restore Global Board
        # Inverse of getCanonicalForm is itself for player=-1 and 1
        global_board_arr = g.getCanonicalForm(next_canon_board, jg_player)

        # Update current_player?
        # Actually next_canon_player tells us if turn changed relative to canonical?
        # If next_canon_player is -1, then next turn is opponent.
        # But we are driven by Log, so we don't strictly need to track current_player for validation
        # unless we want to assert validity.

    # Assign Rewards
    train_examples = []
    for item in history:
        # v = 1 if this player won
        # winner_jg is the winner.
        v = 1 if item["player"] == winner_jg else -1
        train_examples.append((item["board"], item["pi"], v))

    return train_examples


def main():
    query = """
        SELECT "gameover", "log", "initialState"
        FROM "Games"
        WHERE "gameover" IS NOT NULL
          AND ("gameover"::json->>'winner') = '0'
    """
    rows = run_query(query)
    print(f"Found {len(rows)} games")

    all_examples = []
    for row in rows:
        examples = process_game(row)
        if examples:
            print(f"Processed game: {len(examples)} examples")
            all_examples.extend(examples)

    print(f"Total examples: {len(all_examples)}")

    if all_examples:
        out_file = "examples/human-examples.pkl"
        os.makedirs("examples", exist_ok=True)
        save_examples(all_examples, out_file)
        print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
