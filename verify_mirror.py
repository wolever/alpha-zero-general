#!/usr/bin/env python3
"""Verification script for board mirror symmetry.

This script creates various test boards and displays them alongside their
mirrored versions to verify that the mirror symmetry is implemented correctly.
"""

import numpy as np
from JGGame import JGGame, Board, ix_mirror_map, ix_to_ax, ax_to_ix


def print_separator():
    print("\n" + "=" * 80 + "\n")


def verify_mirror_map():
    """Verify that the mirror map is correct (mirroring twice should return to original)."""
    print("Verifying mirror map...")
    errors = []
    for idx in range(len(ix_mirror_map)):
        mirrored_idx = ix_mirror_map[idx]
        double_mirrored_idx = ix_mirror_map[mirrored_idx]
        if double_mirrored_idx != idx:
            errors.append(
                f"Position {idx} -> {mirrored_idx} -> {double_mirrored_idx} (expected {idx})"
            )
    
    if errors:
        print(f"ERROR: Found {len(errors)} mirror map errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        return False
    else:
        print("[OK] Mirror map is correct (mirroring twice returns to original)")
        return True


def test_board_mirroring():
    """Test board mirroring with various configurations."""
    game = JGGame()
    
    print_separator()
    print("TEST 1: Empty board")
    print_separator()
    board_arr = Board.get_arr()
    board = Board(board_arr)
    print("Original board:")
    board.display()
    
    mirrored_arr = game._mirror_board(board_arr)
    mirrored_board = Board(mirrored_arr)
    print("\nMirrored board:")
    mirrored_board.display()
    
    # Verify they're the same (empty board should mirror to itself)
    if np.array_equal(board_arr, mirrored_arr):
        print("\n[OK] Empty board correctly mirrors to itself")
    else:
        print("\n[ERROR] ERROR: Empty board should mirror to itself")
    
    print_separator()
    print("TEST 2: Initial board")
    print_separator()
    board_arr = Board.get_initial_arr()
    board = Board(board_arr)
    print("Original board:")
    board.display()
    
    mirrored_arr = game._mirror_board(board_arr)
    mirrored_board = Board(mirrored_arr)
    print("\nMirrored board:")
    mirrored_board.display()
    
    print_separator()
    print("TEST 3: Board with coins at specific positions")
    print_separator()
    board_arr = Board.get_arr()
    # Place some coins at known positions
    # Position 0 is (0, 4) - top center
    # Position 4 is (4, 4) - top right
    # Position 30 is (0, 0) - center
    board_arr[0] = 3   # Red coins at top center
    board_arr[4] = 2   # Red coins at top right
    board_arr[30] = 5  # Red coins at center
    board_arr[34] = -4  # Blue coins at (4, 0) - right of center
    
    board = Board(board_arr)
    print("Original board:")
    board.display()
    
    # Show coordinate mapping
    print("\nPosition mapping:")
    for idx in [0, 4, 30, 34]:
        q, r = ix_to_ax[idx]
        mirrored_idx = ix_mirror_map[idx]
        mq, mr = ix_to_ax[mirrored_idx]
        print(f"  Position {idx} ({q:3d}, {r:3d}) -> {mirrored_idx} ({mq:3d}, {mr:3d})")
    
    mirrored_arr = game._mirror_board(board_arr)
    mirrored_board = Board(mirrored_arr)
    print("\nMirrored board:")
    mirrored_board.display()
    
    # Verify specific positions
    print("\nVerifying specific positions:")
    checks = [
        (0, 4, "Top left -> Top right (should swap)"),
        (4, 0, "Top right -> Top left (should swap)"),
        (30, 30, "Center -> Center (should stay)"),
        (34, 26, "Right of center -> Left of center"),
    ]
    all_correct = True
    for orig_idx, expected_mirrored_idx, description in checks:
        actual_mirrored_idx = ix_mirror_map[orig_idx]
        if actual_mirrored_idx == expected_mirrored_idx:
            print(f"  [OK] {description}: {orig_idx} -> {actual_mirrored_idx}")
        else:
            print(f"  [ERROR] {description}: {orig_idx} -> {actual_mirrored_idx} (expected {expected_mirrored_idx})")
            all_correct = False
    
    if all_correct:
        print("\n[OK] All position checks passed")
    else:
        print("\n[ERROR] Some position checks failed")
    
    print_separator()
    print("TEST 4: Board with coins on both sides")
    print_separator()
    board_arr = Board.get_arr()
    # Create an asymmetric pattern
    board_arr[1] = 3   # Red at (1, 4)
    board_arr[5] = 2   # Red at (-1, 3)
    board_arr[31] = 4  # Red at (1, 0)
    board_arr[35] = -3  # Blue at (-4, -1)
    board_arr[39] = -2  # Blue at (0, -1)
    
    board = Board(board_arr)
    print("Original board:")
    board.display()
    
    mirrored_arr = game._mirror_board(board_arr)
    mirrored_board = Board(mirrored_arr)
    print("\nMirrored board:")
    mirrored_board.display()
    
    # Verify mirroring twice returns to original
    double_mirrored = game._mirror_board(mirrored_arr)
    if np.array_equal(board_arr, double_mirrored):
        print("\n[OK] Double mirroring returns to original")
    else:
        print("\n[ERROR] ERROR: Double mirroring should return to original")
        print("Differences:")
        diff_indices = np.where(board_arr != double_mirrored)[0]
        for idx in diff_indices[:10]:
            print(f"  Position {idx}: {board_arr[idx]} != {double_mirrored[idx]}")


def test_policy_mirroring():
    """Test policy mirroring with comprehensive test cases."""
    game = JGGame()
    from JGGame import action_pack, action_unpack, ix_mirror_map
    
    print_separator()
    print("TEST 5: Policy mirroring - Basic test")
    print_separator()
    
    board_arr = Board.get_arr()
    board_arr[1] = 3   # Red at (1, 4)
    board_arr[5] = 2   # Red at (-1, 3)
    board_arr[30] = 5  # Red at center
    
    board = Board(board_arr)
    print("Board:")
    board.display()
    
    # Create a simple policy with a few actions
    pi = np.zeros(game.getActionSize(), dtype=np.float32)
    
    # Find valid actions
    valid_moves = game.getValidMoves(board_arr, 1)
    valid_action_indices = np.where(valid_moves)[0]
    
    if len(valid_action_indices) > 0:
        # Set some random probabilities for valid actions
        np.random.seed(42)
        for action_idx in valid_action_indices[:5]:  # Take first 5 valid actions
            pi[action_idx] = np.random.random()
        
        # Normalize
        pi = pi / pi.sum()
        
        print(f"\nOriginal policy has {len(valid_action_indices)} valid actions")
        print("Top 5 actions in original policy:")
        top_actions = np.argsort(pi)[::-1][:5]
        for i, action_idx in enumerate(top_actions):
            if pi[action_idx] > 0:
                skip, src_idx, dst_idx, count = action_unpack(action_idx)
                q, r = ix_to_ax[dst_idx]
                print(f"  {i+1}. Action {action_idx}: prob={pi[action_idx]:.4f}, "
                      f"skip={skip}, src={src_idx}, dst={dst_idx} ({q}, {r}), count={count}")
        
        mirrored_pi = game._mirror_policy(board_arr, pi)
        
        print("\nTop 5 actions in mirrored policy:")
        top_mirrored = np.argsort(mirrored_pi)[::-1][:5]
        for i, action_idx in enumerate(top_mirrored):
            if mirrored_pi[action_idx] > 0:
                skip, src_idx, dst_idx, count = action_unpack(action_idx)
                mq, mr = ix_to_ax[dst_idx]
                print(f"  {i+1}. Action {action_idx}: prob={mirrored_pi[action_idx]:.4f}, "
                      f"skip={skip}, src={src_idx}, dst={dst_idx} ({mq}, {mr}), count={count}")
        
        # Verify probabilities sum to approximately 1
        orig_sum = pi.sum()
        mirrored_sum = mirrored_pi.sum()
        print(f"\nOriginal policy sum: {orig_sum:.6f}")
        print(f"Mirrored policy sum: {mirrored_sum:.6f}")
        
        if abs(orig_sum - 1.0) < 0.001 and abs(mirrored_sum - 1.0) < 0.001:
            print("[OK] Policy probabilities are correctly normalized")
        else:
            print("[ERROR] ERROR: Policy probabilities should sum to 1.0")
    else:
        print("No valid moves found for this board")
    
    # Test 5b: Detailed action-by-action verification
    print_separator()
    print("TEST 5b: Policy mirroring - Action-by-action verification")
    print_separator()
    
    board_arr = Board.get_arr()
    # Create an asymmetric board with coins at specific positions that will have many valid moves
    board_arr[1] = 3   # Red at (1, 4)
    board_arr[6] = 2   # Red at (0, 3)
    board_arr[13] = 4  # Red at (0, 2)
    board_arr[21] = 3  # Red at (0, 1)
    board_arr[30] = 5  # Red at (0, 0) - center
    
    board = Board(board_arr)
    print("Board:")
    board.display()
    
    player_coin_idxs = board.player_coin_idxs(1)
    print(f"\nPlayer coin positions: {player_coin_idxs}")
    print("Player coin coordinates:")
    for i, pos in enumerate(player_coin_idxs):
        q, r = ix_to_ax[pos]
        print(f"  src_idx_player={i}: position {pos} = ({q}, {r})")
    
    # Create a policy with specific test actions
    pi = np.zeros(game.getActionSize(), dtype=np.float32)
    
    # Get valid moves to find some real actions
    valid_moves = game.getValidMoves(board_arr, 1)
    valid_action_indices = np.where(valid_moves)[0]
    
    print(f"\nFound {len(valid_action_indices)} valid actions")
    
    if len(valid_action_indices) >= 3:
        # Select a few specific actions to test
        test_actions = valid_action_indices[:min(10, len(valid_action_indices))]
        
        # Assign probabilities
        for i, action_idx in enumerate(test_actions):
            pi[action_idx] = 0.1 + (i * 0.05)  # Varying probabilities
        
        # Normalize
        pi = pi / pi.sum()
        
        print(f"\nTesting {len(test_actions)} actions:")
        print("\nOriginal actions:")
        for action_idx in test_actions:
            if pi[action_idx] > 0:
                skip, src_idx_player, dst_idx, count = action_unpack(action_idx)
                dst_q, dst_r = ix_to_ax[dst_idx]
                if src_idx_player < len(player_coin_idxs):
                    src_pos = player_coin_idxs[src_idx_player]
                    src_q, src_r = ix_to_ax[src_pos]
                    print(f"  Action {action_idx}: prob={pi[action_idx]:.4f}, "
                          f"src={src_idx_player} (pos {src_pos} = ({src_q}, {src_r})), "
                          f"dst={dst_idx} ({dst_q}, {dst_r}), count={count}")
                else:
                    print(f"  Action {action_idx}: prob={pi[action_idx]:.4f}, "
                          f"src={src_idx_player} (new coin), "
                          f"dst={dst_idx} ({dst_q}, {dst_r}), count={count}")
        
        mirrored_pi = game._mirror_policy(board_arr, pi)
        mirrored_board_arr = game._mirror_board(board_arr)
        mirrored_board = Board(mirrored_board_arr)
        mirrored_player_coin_idxs = mirrored_board.player_coin_idxs(1)
        
        print(f"\nMirrored board coin positions: {mirrored_player_coin_idxs}")
        print("Mirrored board coin coordinates:")
        for i, pos in enumerate(mirrored_player_coin_idxs):
            q, r = ix_to_ax[pos]
            print(f"  src_idx_player={i}: position {pos} = ({q}, {r})")
        
        print("\nMirrored actions:")
        mirrored_action_indices = np.where(mirrored_pi > 0)[0]
        for action_idx in sorted(mirrored_action_indices, key=lambda x: -mirrored_pi[x])[:len(test_actions)]:
            if mirrored_pi[action_idx] > 0:
                skip, src_idx_player, dst_idx, count = action_unpack(action_idx)
                dst_q, dst_r = ix_to_ax[dst_idx]
                if src_idx_player < len(mirrored_player_coin_idxs):
                    src_pos = mirrored_player_coin_idxs[src_idx_player]
                    src_q, src_r = ix_to_ax[src_pos]
                    print(f"  Action {action_idx}: prob={mirrored_pi[action_idx]:.4f}, "
                          f"src={src_idx_player} (pos {src_pos} = ({src_q}, {src_r})), "
                          f"dst={dst_idx} ({dst_q}, {dst_r}), count={count}")
                else:
                    print(f"  Action {action_idx}: prob={mirrored_pi[action_idx]:.4f}, "
                          f"src={src_idx_player} (new coin), "
                          f"dst={dst_idx} ({dst_q}, {dst_r}), count={count}")
        
        # Verify: for each original action, find its mirrored counterpart
        print("\nVerifying action mirroring:")
        all_correct = True
        for orig_action_idx in test_actions:
            if pi[orig_action_idx] == 0:
                continue
            
            skip, src_idx_player, dst_idx, count = action_unpack(orig_action_idx)
            
            # Find the mirrored action
            expected_mirrored_dst = ix_mirror_map[dst_idx]
            
            # Find expected mirrored source
            if src_idx_player < len(player_coin_idxs):
                orig_src_pos = player_coin_idxs[src_idx_player]
                expected_mirrored_src_pos = ix_mirror_map[orig_src_pos]
                # Find src_idx_player in mirrored board
                try:
                    expected_mirrored_src_idx = np.where(mirrored_player_coin_idxs == expected_mirrored_src_pos)[0][0]
                except IndexError:
                    print(f"  [ERROR] Action {orig_action_idx}: mirrored source position {expected_mirrored_src_pos} not found in mirrored board")
                    all_correct = False
                    continue
            else:
                expected_mirrored_src_idx = src_idx_player
            
            expected_mirrored_action = action_pack(skip, expected_mirrored_src_idx, expected_mirrored_dst, count)
            
            # Check if this action exists in mirrored policy with correct probability
            if mirrored_pi[expected_mirrored_action] > 0:
                prob_match = abs(mirrored_pi[expected_mirrored_action] - pi[orig_action_idx]) < 0.0001
                if prob_match:
                    print(f"  [OK] Action {orig_action_idx} -> {expected_mirrored_action} (prob preserved: {pi[orig_action_idx]:.4f})")
                else:
                    print(f"  [ERROR] Action {orig_action_idx} -> {expected_mirrored_action} (prob mismatch: {pi[orig_action_idx]:.4f} vs {mirrored_pi[expected_mirrored_action]:.4f})")
                    all_correct = False
            else:
                print(f"  [ERROR] Action {orig_action_idx}: expected mirrored action {expected_mirrored_action} not found or has zero probability")
                all_correct = False
        
        if all_correct:
            print("\n[OK] All action mirroring checks passed")
        else:
            print("\n[ERROR] Some action mirroring checks failed")
        
        # Verify probability preservation
        orig_sum = pi.sum()
        mirrored_sum = mirrored_pi.sum()
        if abs(orig_sum - mirrored_sum) < 0.0001:
            print(f"[OK] Probability sum preserved: {orig_sum:.6f} == {mirrored_sum:.6f}")
        else:
            print(f"[ERROR] Probability sum mismatch: {orig_sum:.6f} != {mirrored_sum:.6f}")
    
    # Test 5c: Test with initial board (placing new coins)
    print_separator()
    print("TEST 5c: Policy mirroring - Initial board (placing new coins)")
    print_separator()
    
    board_arr = Board.get_initial_arr()
    board = Board(board_arr)
    print("Initial board:")
    board.display()
    
    pi = np.zeros(game.getActionSize(), dtype=np.float32)
    valid_moves = game.getValidMoves(board_arr, 1)
    valid_action_indices = np.where(valid_moves)[0]
    
    if len(valid_action_indices) > 0:
        # Select actions that place new coins (src_idx_player should be 0)
        placing_actions = [a for a in valid_action_indices[:10] if action_unpack(a)[1] == 0]
        
        if len(placing_actions) > 0:
            for i, action_idx in enumerate(placing_actions[:5]):
                pi[action_idx] = 0.2  # Equal probability
            
            pi = pi / pi.sum()
            
            print(f"\nTesting {len([a for a in placing_actions[:5] if pi[a] > 0])} actions that place new coins:")
            for action_idx in placing_actions[:5]:
                if pi[action_idx] > 0:
                    skip, src_idx_player, dst_idx, count = action_unpack(action_idx)
                    dst_q, dst_r = ix_to_ax[dst_idx]
                    print(f"  Action {action_idx}: prob={pi[action_idx]:.4f}, "
                          f"src={src_idx_player} (new coin), dst={dst_idx} ({dst_q}, {dst_r}), count={count}")
            
            mirrored_pi = game._mirror_policy(board_arr, pi)
            mirrored_board_arr = game._mirror_board(board_arr)
            
            print("\nMirrored actions:")
            mirrored_action_indices = np.where(mirrored_pi > 0)[0]
            for action_idx in sorted(mirrored_action_indices, key=lambda x: -mirrored_pi[x])[:5]:
                if mirrored_pi[action_idx] > 0:
                    skip, src_idx_player, dst_idx, count = action_unpack(action_idx)
                    dst_q, dst_r = ix_to_ax[dst_idx]
                    print(f"  Action {action_idx}: prob={mirrored_pi[action_idx]:.4f}, "
                          f"src={src_idx_player} (new coin), dst={dst_idx} ({dst_q}, {dst_r}), count={count}")
            
            # Verify: new coin actions should mirror destinations correctly
            print("\nVerifying new coin placement mirroring:")
            all_correct = True
            for orig_action_idx in placing_actions[:5]:
                if pi[orig_action_idx] == 0:
                    continue
                
                skip, src_idx_player, dst_idx, count = action_unpack(orig_action_idx)
                expected_mirrored_dst = ix_mirror_map[dst_idx]
                expected_mirrored_action = action_pack(skip, src_idx_player, expected_mirrored_dst, count)
                
                if mirrored_pi[expected_mirrored_action] > 0:
                    prob_match = abs(mirrored_pi[expected_mirrored_action] - pi[orig_action_idx]) < 0.0001
                    if prob_match:
                        orig_q, orig_r = ix_to_ax[dst_idx]
                        mir_q, mir_r = ix_to_ax[expected_mirrored_dst]
                        print(f"  [OK] Action {orig_action_idx} (dst {dst_idx} ({orig_q}, {orig_r})) -> "
                              f"{expected_mirrored_action} (dst {expected_mirrored_dst} ({mir_q}, {mir_r}))")
                    else:
                        print(f"  [ERROR] Action {orig_action_idx} -> {expected_mirrored_action} (prob mismatch)")
                        all_correct = False
                else:
                    print(f"  [ERROR] Action {orig_action_idx}: expected mirrored action {expected_mirrored_action} not found")
                    all_correct = False
            
            if all_correct:
                print("[OK] All new coin placement mirroring checks passed")
            else:
                print("[ERROR] Some new coin placement mirroring checks failed")
        else:
            print("No actions that place new coins found")
    else:
        print("No valid moves found for initial board")


def test_symmetries():
    """Test the getSymmetries method."""
    game = JGGame()
    
    print_separator()
    print("TEST 6: getSymmetries method")
    print_separator()
    
    board_arr = Board.get_arr()
    board_arr[1] = 3
    board_arr[30] = 5
    
    board = Board(board_arr)
    print("Original board:")
    board.display()
    
    # Create a simple policy
    pi = np.zeros(game.getActionSize(), dtype=np.float32)
    valid_moves = game.getValidMoves(board_arr, 1)
    if valid_moves.sum() > 0:
        pi[valid_moves] = 1.0 / valid_moves.sum()
    
    symmetries = game.getSymmetries(board_arr, pi)
    print(f"\nNumber of symmetries: {len(symmetries)}")
    
    for i, (sym_board, sym_pi) in enumerate(symmetries):
        print(f"\nSymmetry {i+1}:")
        Board(sym_board).display()
        print(f"Policy sum: {sym_pi.sum():.6f}")


def main():
    print("=" * 80)
    print("BOARD MIRROR SYMMETRY VERIFICATION")
    print("=" * 80)
    
    # Verify mirror map correctness
    if not verify_mirror_map():
        print("\n[ERROR] Mirror map verification failed. Aborting further tests.")
        return
    
    # Test board mirroring
    test_board_mirroring()
    
    # Test policy mirroring
    test_policy_mirroring()
    
    # Test symmetries
    test_symmetries()
    
    print_separator()
    print("Verification complete!")
    print_separator()


if __name__ == "__main__":
    main()

