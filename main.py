import logging
import os
import sys

import coloredlogs
from Coach import Coach

#from othello.OthelloGame import OthelloGame as Game
from JGGame import JGGame as Game
from JGNet import NNetWrapper as nn

from utils import dotdict

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 1,                # Number of complete self-play games to simulate during a new iteration.
    #'numIters': 1000,
    #'numEps': 1,               # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    #'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'numMCTSSims': 5,          # Number of games moves for MCTS to simulate.
    'arenaCompare':  20,        # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': f'./checkpoints-{Game.__name__}',
    'load_model': False,
    'load_folder_file': 'best.pth.tar',
    'numItersForTrainExamplesHistory': 20,

})

sys.setrecursionlimit(10_000)

def main():
    log.info('Loading %s...', Game.__name__)
    g = Game()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.checkpoint, args.load_folder_file)
        nnet.load_checkpoint(os.path.join(args.checkpoint, args.load_folder_file))
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
