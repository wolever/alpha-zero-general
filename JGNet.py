import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from NeuralNet import NeuralNet
from tqdm import tqdm

from utils import dotdict, AverageMeter

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'mps': torch.backends.mps.is_available(),
    'num_channels': 1024,
})

class JGNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(JGNNet, self).__init__()

        # Increase channels for more representational capacity
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)

        # Adapt for the unusual board shape
        # Since your board is only 2 cells high, need to preserve height
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=(0, 1))
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=(0, 1))

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        # Calculate the output size from the convolutions
        with torch.no_grad():
            # Create a dummy input with your board dimensions
            dummy_input = torch.zeros(1, 1, self.board_x, self.board_y)

            # Pass through convolutions
            x = F.relu(self.bn1(self.conv1(dummy_input)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))

        # Get flattened size
        conv_output_size = x.numel()

        self.fc1 = nn.Linear(conv_output_size, 2048)

        self.fc_bn1 = nn.BatchNorm1d(2048)

        self.fc2 = nn.Linear(2048, 1024)
        self.fc_bn2 = nn.BatchNorm1d(1024)

        # Output layer for policy needs to handle the large action space
        self.fc3 = nn.Linear(1024, self.action_size)

        # Value head remains the same
        self.fc4 = nn.Linear(1024, 1)

    def forward(self, s):
        # s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)  # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))  # batch_size x num_channels x board_x x board_y

        # Custom padding to maintain width but allow height reduction
        s = F.relu(self.bn3(self.conv3(s)))  # batch_size x num_channels x reduced_height x board_x
        s = F.relu(self.bn4(self.conv4(s)))  # batch_size x num_channels x further_reduced_height x board_x

        # Flatten for fully connected layers
        s = s.view(s.size(0), -1)

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 2048
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024

        pi = self.fc3(s)  # batch_size x action_size
        v = self.fc4(s)   # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = JGNNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()
        elif args.mps:
            self.nnet.to('mps')

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())
        print(f"Training net on {len(examples)} examples...")

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
                elif args.mps:
                    boards, target_pis, target_vs = boards.contiguous().to('mps'), target_pis.contiguous().to('mps'), target_vs.contiguous().to('mps')

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

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda: board = board.contiguous().cuda()
        elif args.mps: board = board.contiguous().to('mps')
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
        torch.save({ 'state_dict': self.nnet.state_dict() }, filename)

    def load_checkpoint(self, filename):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        if not os.path.exists(filename):
            raise ValueError("No model in path {}".format(filename))
        map_location = None if (args.cuda or args.mps) else 'cpu'
        checkpoint = torch.load(filename, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
