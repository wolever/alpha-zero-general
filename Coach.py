import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from JGGame import JGGame
from MCTS import MCTS

log = logging.getLogger(__name__)

def relink(src: str, dst: str):
    if os.path.exists(dst):
        os.unlink(dst)
    os.link(src, dst)

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    game: JGGame

    def __init__(self, game: JGGame, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        from JGGame import Board, action_unpack

        board_stack = [None] * 10

        while True:

            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)

            # DW note: this appears to be a bug; it should be using player=1 and the canonical board
            #board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
            board, self.curPlayer = self.game.getNextState(canonicalBoard, 1, action)
            board_stack[episodeStep % len(board_stack)] = (episodeStep, board)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

            if episodeStep > 250:
                from JGGame import action_unpack
                print("STUCK IN LOOP")
                print(self.curPlayer)
                for _, past_board in sorted(board_stack, key=lambda x: x[0]):
                    Board(past_board).display()
                Board(board).display()
                print(action, '=', action_unpack(action))
                return []

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            temp_checkpoint = os.path.join(self.args.checkpoint, f'temp.pth.tar')
            self.nnet.save_checkpoint(temp_checkpoint)
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)

            while len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            if not os.path.exists(self.args.checkpoint):
                print("Checkpoint Directory does not exist! Making directory {}".format(self.args.checkpoint))
                os.mkdir(self.args.checkpoint)

            #print("Saving previous network...")
            #temp_file = os.path.join(self.args.checkpoint, 'temp.pth.tar')
            #self.nnet.save_checkpoint(temp_file)

            # Previous network
            #self.pnet.load_checkpoint(temp_file)
            pmcts = MCTS(self.game, self.pnet, self.args)

            # Train the new network
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            # Play against the previous network
            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                is_best = False
            else:
                log.info('ACCEPTING NEW MODEL')
                is_best = True

            if is_best:
                best_file = os.path.join(self.args.checkpoint, 'best.pth.tar')
                self.nnet.save_checkpoint(best_file)
                self.saveTrainExamples(best_file)
                self.pnet.load_checkpoint(best_file)
            else:
                self.nnet.load_checkpoint(temp_checkpoint)


    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, checkpoint_file):
        filename = checkpoint_file + ".examples"
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.checkpoint, self.args.load_folder_file)
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
