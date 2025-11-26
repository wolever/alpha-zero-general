import os
import time
import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from NeuralNet import NeuralNet
from JGGame import JGGame
from tqdm import tqdm

from utils import dotdict, AverageMeter

if TYPE_CHECKING:
    from main import TrainingArgs

log = logging.getLogger(__name__)


nn_args = dotdict(
    {
        "lr": 0.001,
        "dropout": 0.3,
        "epochs": 10,
        "batch_size": 64,
        "device": (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        ),
        "num_channels": 1024,
    }
)


class JGNNet(nn.Module):
    def __init__(self, game: JGGame, nn_args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.nn_args = nn_args

        super(JGNNet, self).__init__()

        # Pre-compute the mapping from 1D index to 2D (x, y) coordinates
        # Board size is 9x9 (q, r in [-4, 4])
        # x = q + 4
        # y = r + 4
        self.grid_size = 9
        self.register_buffer("idx_to_xy", self._create_idx_mapping())

        # Input is 3 channels:
        # 0: Board state
        # 1: Player 1 coins to add (constant plane)
        # 2: Player 2 coins to add (constant plane)
        self.conv1 = nn.Conv2d(3, nn_args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            nn_args.num_channels, nn_args.num_channels, 3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            nn_args.num_channels, nn_args.num_channels, 3, stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            nn_args.num_channels, nn_args.num_channels, 3, stride=1, padding=1
        )

        self.bn1 = nn.BatchNorm2d(nn_args.num_channels)
        self.bn2 = nn.BatchNorm2d(nn_args.num_channels)
        self.bn3 = nn.BatchNorm2d(nn_args.num_channels)
        self.bn4 = nn.BatchNorm2d(nn_args.num_channels)

        self.fc1 = nn.Linear(
            nn_args.num_channels * self.grid_size * self.grid_size, 2048
        )
        self.fc_bn1 = nn.BatchNorm1d(2048)

        self.fc2 = nn.Linear(2048, 1024)
        self.fc_bn2 = nn.BatchNorm1d(1024)

        self.fc3 = nn.Linear(1024, self.action_size)
        self.fc4 = nn.Linear(1024, 1)

    def _create_idx_mapping(self):
        # Import here to avoid circular dependency issues if any
        from JGGame import ix_to_ax

        # Mapping tensor: [63, 2] where each row is [x, y] for that index
        # Initialize with -1 to catch errors
        mapping = torch.full((63, 2), -1, dtype=torch.long)

        # Iterate through all valid board indices
        # Note: The last two indices are coin counts, handled separately
        for idx, (q, r) in ix_to_ax.items():
            x = q + 4
            y = r + 4
            assert 0 <= x < 9 and 0 <= y < 9
            mapping[idx] = torch.tensor([x, y])

        return mapping

    def forward(self, s):
        # s: batch_size x 63 (or 63x1)
        batch_size = s.size(0)
        s = s.view(batch_size, -1)  # Ensure flattened

        # 1. Transform 1D input to 3-channel 9x9 2D input
        # Initialize grid: batch x 3 x 9 x 9
        x = torch.zeros(batch_size, 3, self.grid_size, self.grid_size, device=s.device)

        # Channel 0: Board state
        # We need to scatter the values from s into the grid positions
        # s[:, :61] contains the board cells
        board_vals = s[:, :61]  # batch x 61

        # We use the pre-computed mapping.
        # idx_to_xy is 61x2 (we only need the first 61 entries for the board)
        # We can't easily use scatter_nd with batch dimension in this specific way without some reshaping
        # So we'll do it by indexing.

        # Get x and y coordinates for the 61 board positions
        # mapping is 63x2, take first 61
        coords = self.idx_to_xy[:61]  # 61 x 2
        xs = coords[:, 0]
        ys = coords[:, 1]

        # Assign values.
        # We want x[:, 0, xs, ys] = board_vals
        # But we have a batch dimension.
        # x[:, 0, xs, ys] will select (batch, 61) elements if we broadcast correctly?
        # Actually, standard advanced indexing works if we align dimensions.

        # Let's use a flat view for assignment to be safe and efficient
        # Grid flat size is 81.
        # coords_flat = xs * 9 + ys
        coords_flat = xs * self.grid_size + ys  # 61

        # Create a flat view of the first channel: batch x 81
        x_c0_flat = x[:, 0, :, :].view(batch_size, -1)

        # Assign values
        # We need to expand coords_flat to match batch size if we use scatter
        # or just loop if batch size is small (but it's not).
        # scatter is best.
        # dim=1, index=coords_flat expanded, src=board_vals

        index = coords_flat.unsqueeze(0).expand(batch_size, -1)  # batch x 61
        x_c0_flat.scatter_(1, index, board_vals)

        # Reshape back (done automatically since it's a view)

        # Channel 1: P1 Coins (index -2)
        p1_coins = s[:, -2].view(batch_size, 1, 1)
        x[:, 1, :, :] = p1_coins

        # Channel 2: P2 Coins (index -1)
        p2_coins = s[:, -1].view(batch_size, 1, 1)
        x[:, 2, :, :] = p2_coins

        # 2. Pass through CNN
        s = F.relu(self.bn1(self.conv1(x)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))

        # Flatten
        s = s.view(batch_size, -1)

        s = F.dropout(
            F.relu(self.fc_bn1(self.fc1(s))),
            p=self.nn_args.dropout,
            training=self.training,
        )
        s = F.dropout(
            F.relu(self.fc_bn2(self.fc2(s))),
            p=self.nn_args.dropout,
            training=self.training,
        )

        pi = self.fc3(s)
        v = self.fc4(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)


class NNetWrapper(NeuralNet):
    nnet: JGNNet
    args: "TrainingArgs"

    def __init__(self, game, args):
        self.args = args
        self.nnet = JGNNet(game, nn_args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if nn_args.device:
            self.nnet.to(nn_args.device)

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())
        num_examples = len(examples)
        print(f"Training net on {num_examples} examples...")

        for epoch in range(nn_args.epochs):
            epoch += 1
            with self.args.time("train/epoch") as data:
                self.nnet.train()
                pi_losses = AverageMeter()
                v_losses = AverageMeter()

                batch_count = int(num_examples / nn_args.batch_size)

                t = tqdm(range(batch_count), desc=f"Epoch {epoch:2d}")
                for _ in t:
                    sample_ids = np.random.randint(
                        num_examples, size=nn_args.batch_size
                    )
                    boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                    boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                    target_pis = torch.FloatTensor(np.array(pis))
                    target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                    # predict
                    if nn_args.device:
                        boards, target_pis, target_vs = (
                            boards.contiguous().to(nn_args.device),
                            target_pis.contiguous().to(nn_args.device),
                            target_vs.contiguous().to(nn_args.device),
                        )
                    else:
                        raise AssertionError("No GPU device!")

                    # compute output
                    out_pi, out_v = self.nnet(boards)
                    l_pi = self.loss_pi(target_pis, out_pi)
                    l_v = self.loss_v(target_vs, out_v)
                    total_loss = l_pi + l_v

                    # record loss
                    pi_losses.update(l_pi.item(), boards.size(0))
                    v_losses.update(l_v.item(), boards.size(0))
                    t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                data.update(
                    {
                        "train/epoch/epoch": epoch,
                        "train/epoch/pi_loss": pi_losses.avg,
                        "train/epoch/v_loss": v_losses.avg,
                        "train/epoch/total_loss": pi_losses.avg + v_losses.avg,
                        "train/epoch/num_examples": num_examples,
                    }
                )

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        board = board.contiguous().to(nn_args.device)
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, filename):
        torch.save({"state_dict": self.nnet.state_dict()}, filename)

    def load_checkpoint(self, filename):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        if not os.path.exists(filename):
            log.warning(f"Previous model not found: {filename}")
            return
        map_location = nn_args.device
        checkpoint = torch.load(filename, map_location=map_location)
        self.nnet.load_state_dict(checkpoint["state_dict"])
