import logging
import os
import sys
import time
import requests
import pickle
import io
from collections import deque

import coloredlogs
from Coach import Coach
from JGGame import JGGame as Game
from JGNet import NNetWrapper as nn
from utils import dotdict

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

# Configuration
args = dotdict({
    'numIters': 1000,
    'numEps': 1,               # Number of games per batch to upload
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 100,
    'MCTSDepth': 10,
    'arenaCompare': 20,
    'cpuct': 1,
    'checkpoint': './checkpoints-JGGame-worker/',
    'load_model': False,
    'load_folder_file': 'best.pth.tar',
    'numItersForTrainExamplesHistory': 20,
    'trainer_url': 'http://localhost:8000',
})

def fetch_model(nnet):
    """Downloads the latest model from the trainer."""
    url = f"{args.trainer_url}/model"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            if not os.path.exists(args.checkpoint):
                os.makedirs(args.checkpoint)
            filepath = os.path.join(args.checkpoint, 'best.pth.tar')
            with open(filepath, 'wb') as f:
                f.write(response.content)
            nnet.load_checkpoint(filepath)
            log.info("Loaded latest model from trainer.")
            return True
        else:
            log.error(f"Failed to fetch model: {response.status_code}")
            return False
    except Exception as e:
        log.error(f"Error fetching model: {e}")
        return False

def upload_examples(examples):
    """Uploads training examples to the trainer."""
    url = f"{args.trainer_url}/upload_examples"
    try:
        # Pickle the examples
        data = pickle.dumps(examples)
        response = requests.post(url, data=data)
        if response.status_code == 200:
            log.info(f"Uploaded {len(examples)} examples.")
            return True
        else:
            log.error(f"Failed to upload examples: {response.status_code}")
            return False
    except Exception as e:
        log.error(f"Error uploading examples: {e}")
        return False

def main():
    if len(sys.argv) > 1:
        args.trainer_url = sys.argv[1]

    g = Game()
    nnet = nn(g)
    c = Coach(g, nnet, args)

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    log.info(f"Worker started, connecting to {args.trainer_url}")

    while True:
        # 1. Fetch latest model
        if fetch_model(nnet):
            # 2. Generate examples
            examples = []
            for _ in range(args.numEps):
                c.mcts = None # Reset MCTS for new game/model?
                # Coach.executeEpisode resets MCTS internally if we look at the code?
                # No, Coach.executeEpisode uses self.mcts.
                # In Coach.learn loop: self.mcts = MCTS(self.game, self.nnet, self.args)
                # So we should reset it here.
                from MCTS import MCTS
                c.mcts = MCTS(g, nnet, args)
                examples += c.executeEpisode()

            # 3. Upload
            upload_examples(examples)
        else:
            time.sleep(5)

if __name__ == "__main__":
    main()
