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
        curPlayer = 1
        canonicalBoard = self.game.getInitBoard()
        episodeStep = 0

        from JGGame import Board, action_unpack
        verbose = False

        while True:
            episodeStep += 1
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)

            # DW note: this appears to be a bug; it should be using player=1 and the canonical board
            #board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
            nextBoard, nextPlayer = self.game.getNextState(canonicalBoard, 1, action)
            #print("Next player:", nextPlayer)
            #print("Next board:")
            #Board(nextBoard).display()

            if nextPlayer != 1:
                curPlayer = nextPlayer
                canonicalBoard = self.game.getCanonicalForm(nextBoard, nextPlayer)
            else:
                canonicalBoard = nextBoard

            r = self.game.getGameEnded(canonicalBoard, 1)

            if r != 0:
                # Scale reward based on number of turns

                result = []
                if verbose:
                    print("Ending player:", curPlayer)
                    print("Ending result:", r)
                    print("Ending board:")
                    Board(canonicalBoard).display()

                for x in trainExamples:
                    player_perspective = r * x[1]
                    is_win = player_perspective > 0
                    min_turns = 10
                    max_turns = 75 if is_win else 20
                    min_scale = 0.2

                    # Calculate the reward scaling factor (from 1.0 to min_scale)
                    if episodeStep <= min_turns:
                        scale = 1.0
                    elif episodeStep >= max_turns:
                        scale = min_scale
                    else:
                        scale = 1.0 - (1.0 - min_scale) * (episodeStep - min_turns) / (max_turns - min_turns)

                    # Process each example with the appropriate scaled reward
                    # Determine if player won or lost

                    # Scale the reward according to the number of turns
                    scaled_reward = player_perspective * scale * (0.75 if is_win else 1)
                    if verbose:
                        print(f"Board reward: {scaled_reward}")
                        Board(x[0]).display()

                    result.append((x[0], x[2], scaled_reward))

                #print("Done!")
                #breakpoint()
                return result

            if episodeStep > 250:
                from JGGame import action_unpack
                print("STUCK IN LOOP")
                print(curPlayer)
                Board(self.game.getCanonicalForm(canonicalBoard, curPlayer)).display()
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
            temp_checkpoint = os.path.join(self.args.checkpoint, 'temp.pth.tar')
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

            log.info('PRV/NEW WINS : %d / %d ; DRAWS : %d' % (pwins, nwins, draws))
            is_new_better = float(nwins) / ((pwins + nwins) or 1) > self.args.updateThreshold
            if not is_new_better:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(temp_checkpoint)
            else:
                log.info('ACCEPTING NEW MODEL')
                best_file = os.path.join(self.args.checkpoint, 'best.pth.tar')
                self.nnet.save_checkpoint(best_file)
                self.saveTrainExamples(best_file)
                self.pnet.load_checkpoint(best_file)

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
