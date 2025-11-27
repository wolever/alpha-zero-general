from JGNet import NNetWrapper
from JGNet import JGNNet
import logging
import os
import traceback
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from typing import TYPE_CHECKING
import multiprocessing as mp

import numpy as np
from tqdm import tqdm
import wandb

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
    nnet: NNetWrapper
    pnet: NNetWrapper

    def __init__(self, game: JGGame, nnet: NNetWrapper, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game, args)  # the competitor network
        self.pnet.nnet.load_state_dict(self.nnet.nnet.state_dict())
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        # If we successfully load past examples, we'll skip the first self-play
        # round in learn() and immediately start training from them.
        self.skip_first_self_play = False

    @staticmethod
    def executeEpisode(mcts, game, args):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Args:
            mcts: MCTS instance to use for action selection
            game: Game instance
            args: Training arguments

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        playerAbs = 1
        boardCanonical = game.getInitBoard()
        episodeStep = 0

        from JGGame import Board, action_unpack

        while True:
            episodeStep += 1
            if episodeStep > args.maxTurnsInGame:
                print("\nSTUCK IN LOOP")
                print("Player:", playerAbs)
                Board(game.getCanonicalForm(boardCanonical, playerAbs)).display()
                return ([], episodeStep, 0)

            # Temperature decay: Keep high exploration early, then decay gradually
            # This prevents premature convergence on trivial opening patterns
            if episodeStep < args.tempThreshold // 2:
                # Keep temp=1.0 for first half of tempThreshold (first 10 turns)
                temp = 1.0
            elif episodeStep < args.tempThreshold:
                # Decay from 1.0 to 0.5 over turns 10-20
                progress = (episodeStep - args.tempThreshold // 2) / (
                    args.tempThreshold // 2
                )
                temp = 1.0 - 0.5 * progress
            else:
                # Decay from 0.5 to 0.1 over turns 20-40
                progress = min(
                    1.0,
                    (episodeStep - args.tempThreshold) / args.tempThreshold,
                )
                temp = 0.5 - 0.4 * progress
            pi = mcts.getActionProb(boardCanonical, temp=temp)
            sym = game.getSymmetries(boardCanonical, pi)
            for b, p in sym:
                trainExamples.append([playerAbs, b, p])

            action = np.random.choice(len(pi), p=pi)
            nextBoard, nextPlayer = game.getNextState(boardCanonical, 1, action)

            winnerCanonical = game.getGameEnded(nextBoard, 1)
            if not winnerCanonical:
                if nextPlayer != 1:
                    playerAbs = -playerAbs
                    boardCanonical = game.getCanonicalForm(nextBoard, -1)
                else:
                    boardCanonical = nextBoard
                continue

            result = []
            winnerAbs = winnerCanonical * playerAbs
            for trainPlayerAbs, trainBoard, trainPi in trainExamples:
                # Simple reward scaling - 1 for win, -1 for loss
                reward = winnerAbs * trainPlayerAbs
                result.append((trainBoard, trainPi, reward))

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
                # result.append((board, pi, reward))

            return (
                result,  # list[(board, pi, reward)]
                episodeStep,
                winnerAbs,
            )

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
                    data["iteration/result"] = "ok"
                except Exception as e:
                    error_count += 1
                    giving_up = error_count > 10
                    data.update(
                        {
                            "iteration/result": "error",
                            "iteration/error": repr(e),
                            "iteration/traceback": traceback.format_exc(),
                            "iteration/error_count": error_count,
                            "iteration/aborted": giving_up,
                        }
                    )
                    log.exception(f"Error in iteration {i}: {e}")
                    if giving_up:
                        break
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
            with self.args.time("self-play") as data:
                new_examples, winners = self.runSelfPlay()
                data.update(
                    {
                        "self-play/examples": len(new_examples),
                        "self-play/first_player_win_rate": (
                            winners.count(1) / (len(winners) or 1)
                        ),
                        "self-play/draws": winners.count(0),
                    }
                )
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
            data["train/examples"] = len(trainExamples)

        pmcts = MCTS(self.game, self.pnet, self.args)
        nmcts = MCTS(self.game, self.nnet, self.args)

        # Play against the previous network
        log.info("PITTING AGAINST PREVIOUS VERSION")
        with self.args.time("arena-result") as data:
            arena = Arena(
                self.args,
                lambda x: np.argmax(
                    pmcts.getActionProb(x, temp=0.2)
                ),  # Small temp for robustness testing
                lambda x: np.argmax(nmcts.getActionProb(x, temp=0.2)),
                self.game,
            )
            pwins, nwins, draws, results_in_position = arena.playGames(
                self.args.arenaCompare
            )
            is_new_better = (
                float(nwins) / ((pwins + nwins) or 1) > self.args.updateThreshold
            )
            total_played = pwins + nwins + draws
            data.update(
                {
                    "arena-result/old_wins": pwins,
                    "arena-result/new_wins": nwins,
                    "arena-result/draws": draws,
                    "arena-result/total_games": total_played,
                    "arena-result/new_win_rate": (
                        float(nwins) / total_played if total_played else 0.0
                    ),
                    "arena-result/is_new_better": float(is_new_better),
                    "arena-result/first_player_win_rate": (
                        (results_in_position[0][2] + results_in_position[1][1])
                        / (np.array(results_in_position).sum() or 1)
                    ),
                    "arena-result/new_first/draws": results_in_position[0][0],
                    "arena-result/new_first/old_wins": results_in_position[0][1],
                    "arena-result/new_first/new_wins": results_in_position[0][2],
                    "arena-result/new_second/draws": results_in_position[1][0],
                    "arena-result/new_second/old_wins": results_in_position[1][1],
                    "arena-result/new_second/new_wins": results_in_position[1][2],
                }
            )

        log.info("OLD/NEW WINS : %d / %d ; DRAWS : %d" % (pwins, nwins, draws))
        if not is_new_better:
            log.info("REJECTING NEW MODEL")
            self.nnet.nnet.load_state_dict(self.pnet.nnet.state_dict())
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
            f
            for f in os.listdir(exdir)
            if f.endswith(".pkl") and os.path.isfile(os.path.join(exdir, f))
        )
        if not files:
            log.warning(f'No example files found in "{exdir}".')
            return

        files.reverse()
        files = files[: self.args.numItersForTrainExamplesHistory]

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
            log.warning(
                f"Finished loading examples from {exdir}, but none were usable."
            )
            return

        self.skip_first_self_play = True
        log.info(
            f"Finished loading trainExamples. "
            f"Total files: {len(self.trainExamplesHistory)}, total examples: {total_examples}. "
            "First self-play iteration will be skipped."
        )

    def runSelfPlay(self):
        """
        Execute self-play games, optionally in parallel.

        Uses multiprocessing to run multiple episodes concurrently if
        args.numParallelSelfPlay > 1.

        Returns:
            Tuple of (iterationTrainExamples, winners)
        """
        from parallel_worker import execute_episode_worker

        winners = []
        iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

        num_workers = getattr(self.args, "numParallelSelfPlay", 1)
        num_episodes = self.args.numEps

        if num_workers > 1:
            # Parallel execution using multiprocessing
            log.info(f"Running self-play with {num_workers} parallel workers")

            # Get the neural network state dict and move to CPU for sharing
            nnet_state_dict = {
                k: v.cpu() for k, v in self.nnet.nnet.state_dict().items()
            }

            # Create pickleable args
            args_dict = self.args.model_dump()
            from main import TrainingArgs

            pickleable_args = TrainingArgs(**args_dict)

            # Import the initializer
            from parallel_worker import init_worker, execute_episode_worker

            # Create process pool with initializer
            # Workers load the network once during initialization
            with mp.Pool(
                processes=num_workers,
                initializer=init_worker,
                initargs=(pickleable_args, nnet_state_dict, self.game.__class__),
            ) as pool:
                with tqdm(total=num_episodes, desc="Self Play") as pbar:
                    # Workers are already initialized, just send episode IDs
                    for result in pool.imap_unordered(
                        execute_episode_worker, range(num_episodes)
                    ):
                        # Log each game (includes duration from worker)
                        trainExamples, episodeStep, winnerAbs, duration = result
                        iterationTrainExamples += trainExamples
                        winners.append(winnerAbs)
                        with self.args.time("self-play-game") as data:
                            data.update(
                                {
                                    "duration": duration,  # Override with worker-measured time
                                    "self-play-game/result": (
                                        "success" if trainExamples else "failed"
                                    ),
                                    "self-play-game/winner": winnerAbs,
                                    "self-play-game/turns": episodeStep,
                                    "self-play-game/examples": len(trainExamples),
                                }
                            )

                        pbar.update(1)

        else:
            # Sequential execution (original implementation)
            log.info("Running self-play sequentially")
            for _ in tqdm(range(num_episodes), desc="Self Play"):
                with self.args.time("self-play-game") as data:
                    try:
                        self.mcts = MCTS(
                            self.game, self.nnet, self.args
                        )  # reset search tree
                        trainExamples, episodeStep, winnerAbs = Coach.executeEpisode(
                            self.mcts, self.game, self.args
                        )
                        iterationTrainExamples += trainExamples
                        data.update(
                            {
                                "self-play-game/result": "success",
                                "self-play-game/winner": winnerAbs,
                                "self-play-game/turns": episodeStep,
                                "self-play-game/examples": len(trainExamples),
                            }
                        )
                        winners.append(winnerAbs)
                    except Exception as e:
                        log.exception(f"Failed to execute episode: {e}")
                        data.update(
                            {
                                "self-play-game/result": "failed",
                                "self-play-game/error": repr(e),
                            }
                        )

        return iterationTrainExamples, winners


def save_examples(examples, filename):
    if not examples:
        return
    with open(filename, "wb+") as f:
        Pickler(f).dump(examples)


def load_examples(filename):
    with open(filename, "rb") as f:
        return Unpickler(f).load()
