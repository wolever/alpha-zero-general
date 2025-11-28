import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from JGGame import JGGame, action_unpack


if TYPE_CHECKING:
    from main import TrainingArgs

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS:
    """
    This class handles the MCTS tree.
    """

    game: JGGame
    args: "TrainingArgs"

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [
            self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
            for a in range(self.game.getActionSize())
        ]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        # Use log-space calculations to prevent overflow when temp is small
        # Convert to numpy array for easier computation
        counts = np.array(counts, dtype=np.float64)

        # Only add EPS to valid actions to avoid giving non-zero probability to invalid actions
        counts = np.maximum(counts, EPS)

        # Compute log(counts) * (1.0 / temp) in log space
        log_counts = np.log(counts) / temp

        # Subtract max to prevent overflow when exponentiating
        log_counts_max = np.max(log_counts)
        exp_counts = np.exp(log_counts - log_counts_max)

        # Ensure only valid actions are considered
        valids = self.game.getValidMoves(canonicalBoard, 1)
        exp_counts = np.where(valids, exp_counts, 0)

        # Normalize to get probabilities
        counts_sum = np.sum(exp_counts)

        if counts_sum == 0:
            # No actions were explored (e.g., all paths hit terminal states or depth limits)
            # Fall back to uniform distribution over valid moves
            log.warning(
                "No actions explored in MCTS, using uniform distribution over valid moves"
            )
            valids = self.game.getValidMoves(canonicalBoard, 1)
            probs = valids / np.sum(valids)
            return probs.tolist()

        probs = (exp_counts / counts_sum).tolist()
        return probs

    def search(self, canonicalBoard, depth=0):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if depth > self.args.MCTSDepth:
            _, v = self.nnet.predict(canonicalBoard)
            return -v

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)

        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                from JGGame import Board

                Board(canonicalBoard).display()
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            if depth == 0 and self.args.dirichletEpsilon > 0:
                # Add Dirichlet noise to the root node to encourage exploration
                valids_idxs = np.where(valids)[0]
                noise = np.random.dirichlet(
                    [self.args.dirichletAlpha] * len(valids_idxs)
                )

                # Mix noise with the prior probabilities
                # self.Ps[s] is already masked and normalized to valid moves
                self.Ps[s][valids_idxs] = (1 - self.args.dirichletEpsilon) * self.Ps[s][
                    valids_idxs
                ] + self.args.dirichletEpsilon * noise

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float("inf")
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in np.where(valids)[0]:
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(
                    self.Ns[s]
                ) / (1 + self.Nsa[(s, a)])
            else:
                u = (
                    self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
                )  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s, depth + 1)
        if next_player == 1:
            v = -v

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (
                self.Nsa[(s, a)] + 1
            )
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
