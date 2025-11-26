import logging
import os
import sys
import traceback
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from Arena import Arena
from JGGame import JGGame
from MCTS import MCTS

if TYPE_CHECKING:
    from main import TrainingArgs

log = logging.getLogger(__name__)


def relink(src: str, dst: str):
    if os.path.exists(dst):
        os.unlink(dst)
    os.link(src, dst)


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    game: JGGame
    args: "TrainingArgs"

    def __init__(self, game: JGGame, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        # If we successfully load past examples, we'll skip the first self-play
        # round in learn() and immediately start training from them.
        self.skip_first_self_play = False

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
            # board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
            nextBoard, nextPlayer = self.game.getNextState(canonicalBoard, 1, action)
            # print("Next player:", nextPlayer)
            # print("Next board:")
            # Board(nextBoard).display()

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
                    # Simple reward scaling - 1 for win, -1 for loss
                    reward = r * x[1]

                    # Complex reward scaling
                    # player_perspective = r * x[1]
                    # is_win = player_perspective > 0
                    # min_turns = 20 if is_win else 7
                    # max_turns = 75 if is_win else 20
                    # min_scale = 0.2

                    ## Calculate the reward scaling factor (from 1.0 to min_scale)
                    # if episodeStep <= min_turns:
                    #    scale = 1.0
                    # elif episodeStep >= max_turns:
                    #    scale = min_scale
                    # else:
                    #    scale = 1.0 - (1.0 - min_scale) * (episodeStep - min_turns) / (max_turns - min_turns)

                    ## Process each example with the appropriate scaled reward
                    ## Determine if player won or lost

                    ## Scale the reward according to the number of turns
                    # reward = player_perspective * scale * (0.75 if is_win else 1)
                    # if verbose:
                    #    print(f"Board reward: {reward}")
                    #    Board(x[0]).display()

                    result.append((x[0], x[2], reward))

                return result

            if episodeStep > self.args.maxTurnsInGame:
                from JGGame import action_unpack

                print("STUCK IN LOOP")
                print(curPlayer)
                Board(self.game.getCanonicalForm(canonicalBoard, curPlayer)).display()
                print(action, "=", action_unpack(action))
                return []

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        error_count = 0

        for i in range(1, self.args.numIters + 1):
            with self.args.time("iteration") as data:
                data["iteration"] = i
                try:
                    self.runIteration(i)
                    error_count = 0
                except Exception as e:
                    error_count += 1
                    giving_up = error_count > 10
                    data.update(
                        {
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                            "giving_up": giving_up,
                            "error_count": error_count,
                        }
                    )
                    log.error(f"Error in iteration {i}: {e}")
                    if giving_up:
                        log.error(
                            f"Giving up on iteration {i} after {error_count} errors"
                        )
                        raise e
                    log.info("Trying again...")

    def runIteration(self, i: int):
        log.info(f"Starting Iter #{i} ...")

        # Optionally skip the first self-play iteration if we already have
        # examples loaded from disk.
        skip_self_play = self.skip_first_self_play and i == 1
        if skip_self_play:
            log.info(
                "Skipping self-play for first iteration because past examples were loaded."
            )
            new_examples = []
        else:
            with self.args.time("self_play") as data:
                new_examples = self.runSelfPlay()
                data["num_examples"] = len(new_examples)
            log.info(f"Collected {len(new_examples)} examples.")

        exdir = os.path.join(self.args.dataDirectory, "examples")
        os.makedirs(exdir, exist_ok=True)
        ex_file = f"{self.args.runId}-{i}-{len(new_examples)}.pkl"
        save_examples(new_examples, os.path.join(exdir, ex_file))
        self.args.write_log(
            "examples_collected",
            {"iteration": i, "num_examples": len(new_examples), "file": ex_file},
        )

        if new_examples:
            self.trainExamplesHistory.append(new_examples)

        while (
            len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory
        ):
            log.warning(
                f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}"
            )
            self.trainExamplesHistory.pop(0)

        # shuffle examples before training
        trainExamples = []
        for e in self.trainExamplesHistory:
            trainExamples.extend(e)
        shuffle(trainExamples)

        # Train the new network
        with self.args.time("train") as data:
            self.nnet.train(trainExamples)
            data["num_examples"] = len(trainExamples)

        pmcts = MCTS(self.game, self.pnet, self.args)
        nmcts = MCTS(self.game, self.nnet, self.args)

        # Play against the previous network
        log.info("PITTING AGAINST PREVIOUS VERSION")
        with self.args.time("arena") as data:
            arena = Arena(
                self.args,
                lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
                self.game,
            )
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
            is_new_better = (
                float(nwins) / ((pwins + nwins) or 1) > self.args.updateThreshold
            )
            data.update(
                {
                    "prev_wins": pwins,
                    "new_wins": nwins,
                    "draws": draws,
                    "is_new_better": is_new_better,
                }
            )

        log.info("PRV/NEW WINS : %d / %d ; DRAWS : %d" % (pwins, nwins, draws))
        if not is_new_better:
            log.info("REJECTING NEW MODEL")
            self.nnet = self.pnet
        else:
            log.info("ACCEPTING NEW MODEL")
            with self.args.time("save_best") as data:
                best_file = os.path.join(self.args.dataDirectory, "best.pth.tar")
                self.nnet.save_checkpoint(best_file)
                self.saveTrainExamples(best_file)
                best_file_link = f"{self.args.runId}-best-{i}.pth.tar"
                os.link(
                    best_file, os.path.join(self.args.dataDirectory, best_file_link)
                )
                data["file"] = best_file_link
                self.pnet.load_checkpoint(best_file)

    def saveTrainExamples(self, checkpoint_file):
        filename = checkpoint_file + ".examples"
        save_examples(self.trainExamplesHistory, filename)

    def loadTrainExamples(self):
        """
        Load all previously saved training examples from the examples/
        directory under args.dataDirectory. This replaces the old behavior
        of loading from a single checkpoint-specific .examples file.

        If one or more example files are successfully loaded, we mark
        skip_first_self_play so that the first self-play round is skipped
        and training starts directly from the loaded data.
        """

        exdir = os.path.join(self.args.dataDirectory, "examples")
        if not os.path.isdir(exdir):
            log.warning(f'Directory "{exdir}" with trainExamples not found!')
            return

        files = sorted(
            f for f in os.listdir(exdir) if f.endswith(".pkl") and os.path.isfile(os.path.join(exdir, f))
        )
        if not files:
            log.warning(f'No example files found in "{exdir}".')
            return

        log.info(f"Loading trainExamples from {len(files)} files in {exdir}...")

        self.trainExamplesHistory = []
        total_examples = 0

        for fname in files:
            path = os.path.join(exdir, fname)
            try:
                examples = load_examples(path)
            except Exception as e:
                log.error(f"Failed to load examples from {path}: {e}")
                continue

            self.trainExamplesHistory.append(examples)
            try:
                num = len(examples)
            except TypeError:
                num = 0
            total_examples += num
            log.info(f"Loaded {num} examples from {fname}")

        if not self.trainExamplesHistory:
            log.warning(f"Finished loading examples from {exdir}, but none were usable.")
            return

        self.skip_first_self_play = True
        log.info(
            f"Finished loading trainExamples. "
            f"Total files: {len(self.trainExamplesHistory)}, total examples: {total_examples}. "
            "First self-play iteration will be skipped."
        )

    def runSelfPlay(self):
        iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
        for _ in tqdm(range(self.args.numEps), desc="Self Play"):
            self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
            iterationTrainExamples += self.executeEpisode()
        return iterationTrainExamples


def save_examples(examples, filename):
    with open(filename, "wb+") as f:
        Pickler(f).dump(examples)
    f.closed


def load_examples(filename):
    with open(filename, "rb") as f:
        return Unpickler(f).load()
