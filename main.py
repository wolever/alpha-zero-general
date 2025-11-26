import io
import logging
import time
import os
import sys
import json
from contextlib import contextmanager

import coloredlogs
from Coach import Coach
from pydantic import BaseModel, PrivateAttr
from datetime import datetime

# from othello.OthelloGame import OthelloGame as Game
from JGGame import JGGame as Game
from JGNet import NNetWrapper as nn

from utils import dotdict

log = logging.getLogger(__name__)

coloredlogs.install(level="INFO")  # Change this to DEBUG to see more info.


class TrainingArgs(BaseModel):
    runId: str = datetime.now().strftime("%Y%m%d%H%M%S")
    numIters: int = 1000
    numEps: int = 200  # Number of self-play games per iteration
    tempThreshold: int = 7  # The first N moves are random, then the rest are greedy
    updateThreshold: float = 0.6  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    numMCTSSims: int = 50  # Number of games moves for MCTS to simulate.
    MCTSDepth: int = 9  # Depth of the MCTS tree.
    arenaCompare: int = 24  # Number of games to play during arena play to determine if new net will be accepted.
    cpuct: int = 1  # Exploration constant
    dataDirectory: str = (
        f"./checkpoints-{Game.__name__}-v0"  # Directory to save the checkpoints
    )
    load_model: bool = False  # Whether to load the model from the checkpoint
    load_folder_file: str = "best.pth.tar"  # Name of the checkpoint file

    maxTurnsInGame: int = (
        75  # Maximum number of turns in a game before it's considered a draw.
    )

    maxlenOfQueue: int = 10_000  # Number of game examples to train the neural networks.
    numItersForTrainExamplesHistory: int = (
        10  # Number of iterations to store the train examples
    )

    _outf: io.TextIOWrapper = PrivateAttr(default=None)

    def write_log(self, event: str, data: dict):
        if self._outf is None:
            outdir = os.path.join(self.dataDirectory, "logs")
            os.makedirs(outdir, exist_ok=True)
            self._outf = open(os.path.join(outdir, f"{self.runId}.jsonl"), "w")
        json.dump(
            {"ts": datetime.now().isoformat(), "event": event, **data}, self._outf
        )
        self._outf.write("\n")
        self._outf.flush()

    _timeDataStack: list[tuple[str, float]] = PrivateAttr(default_factory=list)

    @contextmanager
    def time(self, step: str):
        start = time.time()
        data = {}
        self._timeDataStack.append(data)
        yield data
        self._timeDataStack.remove(data)

        res_data = {}
        for d in self._timeDataStack:
            res_data.update(d)
        res_data.update(data)
        self.write_log(step, {"duration": time.time() - start, **data})

    def close(self):
        outf = getattr(self, "_outf", None)
        if outf is not None:
            outf.close()


sys.setrecursionlimit(10_000)

args = TrainingArgs()


def main():
    log.info("Starting run %s...", args.runId)
    log.info("Loading %s...", Game.__name__)
    g = Game()

    log.info("Loading %s...", nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info(
            'Loading checkpoint "%s/%s"...', args.dataDirectory, args.load_folder_file
        )
        nnet.load_checkpoint(os.path.join(args.dataDirectory, args.load_folder_file))
    else:
        log.warning("Not loading a checkpoint!")

    log.info("Loading the Coach...")
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info("Starting the learning process 🎉")
    try:
        args.write_log("starting", args.model_dump())
        c.learn()
    finally:
        args.close()


if __name__ == "__main__":
    main()
