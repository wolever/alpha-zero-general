import torch
from JGGame import JGGame, ix_to_ax
from JGNet import JGNNet, nn_args

def verify_transformation():
    game = JGGame()
    net = JGNNet(game, nn_args)
    net.eval()

    # Create a dummy input: batch size 1, 63 elements
    # We'll set specific indices to specific values to verify mapping
    input_board = torch.zeros(1, 63)

    # Set a value at a known coordinate
    # (0, 0) -> should map to (4, 4) in the 9x9 grid
    # Find index for (0, 0)
    center_idx = -1
    for idx, coord in ix_to_ax.items():
        if coord == (0, 0):
            center_idx = idx
            break

    assert center_idx != -1
    input_board[0, center_idx] = 10.0 # Arbitrary value

    # Set coin counts
    input_board[0, -2] = 5.0 # P1 coins
    input_board[0, -1] = 7.0 # P2 coins

    print(f"Testing with input value 10.0 at (0,0) [idx {center_idx}]")

    # We need to hook into the forward pass or just copy the logic to verify
    # Let's copy the logic for verification to be sure what's happening inside

    s = input_board
    batch_size = s.size(0)

    # 1. Transform
    x = torch.zeros(batch_size, 3, 9, 9)
    board_vals = s[:, :61]

    coords = net.idx_to_xy[:61]
    xs = coords[:, 0]
    ys = coords[:, 1]
    coords_flat = xs * 9 + ys

    x_c0_flat = x[:, 0, :, :].view(batch_size, -1)
    index = coords_flat.unsqueeze(0).expand(batch_size, -1)
    x_c0_flat.scatter_(1, index, board_vals)

    x[:, 1, :, :] = s[:, -2].view(batch_size, 1, 1, 1)
    x[:, 2, :, :] = s[:, -1].view(batch_size, 1, 1, 1)

    # Verify Center
    center_val = x[0, 0, 4, 4].item()
    print(f"Value at grid (4, 4): {center_val}")
    assert center_val == 10.0, f"Expected 10.0, got {center_val}"

    # Verify Coins
    p1_val = x[0, 1, 0, 0].item() # Should be constant
    print(f"P1 coin plane value: {p1_val}")
    assert p1_val == 5.0

    p2_val = x[0, 2, 8, 8].item()
    print(f"P2 coin plane value: {p2_val}")
    assert p2_val == 7.0

    # Run actual forward pass to check for shape errors
    print("Running forward pass...")
    pi, v = net(input_board)
    print(f"Output shapes: pi={pi.shape}, v={v.shape}")
    print("Verification passed!")

if __name__ == "__main__":
    verify_transformation()
