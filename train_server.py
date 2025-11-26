import logging
import os
import sys
import threading
import time
import io
from collections import deque
from pickle import Pickler, Unpickler

import coloredlogs
import uvicorn
import fastapi
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Any

from Coach import Coach
from JGGame import JGGame as Game
from JGNet import NNetWrapper as nn
from utils import dotdict

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

# Configuration
args = dotdict({
    'numIters': 1000,
    'numEps': 10,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 100,
    'MCTSDepth': 10,
    'arenaCompare': 20,
    'cpuct': 1,
    'checkpoint': './checkpoints-JGGame-distributed/',
    'load_model': False,
    'load_folder_file': 'best.pth.tar',
    'numItersForTrainExamplesHistory': 20,
    'port': 8000,
})

app = FastAPI()
example_queue = deque()
latest_model_lock = threading.Lock()

@app.get("/model")
def get_model():
    """Returns the latest model checkpoint file."""
    checkpoint_path = os.path.join(args.checkpoint, 'best.pth.tar')
    if not os.path.exists(checkpoint_path):
        # If no best model yet, try temp
        checkpoint_path = os.path.join(args.checkpoint, 'temp.pth.tar')

    if not os.path.exists(checkpoint_path):
        raise HTTPException(status_code=404, detail="Model not found")

    return FileResponse(checkpoint_path, media_type='application/octet-stream', filename='best.pth.tar')

@app.post("/upload_examples")
async def upload_examples(request: fastapi.Request):
    body = await request.body()
    try:
        examples = Unpickler(io.BytesIO(body)).load()
        example_queue.extend(examples)
        return {"status": "ok", "count": len(examples)}
    except Exception as e:
        log.error(f"Error receiving examples: {e}")
        raise HTTPException(status_code=400, detail=str(e))

class DistributedCoach(Coach):
    def runSelfPlay(self):
        """
        Overrides Coach.runSelfPlay to fetch examples from the network queue.
        """
        iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

        # Estimate turns per game to decide how many examples we need
        # In original main.py, numEps=10.
        # Let's assume we want roughly numEps * 40 turns = 400 examples per iteration.
        required_examples = self.args.numEps * 40

        log.info(f"Waiting for {required_examples} examples from workers...")

        while len(example_queue) < required_examples:
            time.sleep(2)
            log.info(f"Queue size: {len(example_queue)} / {required_examples}")

        # Drain the queue up to a reasonable limit or all of it
        # We take everything available to maximize data usage
        count = 0
        while len(example_queue) > 0:
            iterationTrainExamples.append(example_queue.popleft())
            count += 1

        log.info(f"Collected {count} examples.")
        return iterationTrainExamples

def start_server():
    uvicorn.run(app, host="0.0.0.0", port=args.port)

def main():
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    g = Game()
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(os.path.join(args.checkpoint, args.load_folder_file))
    else:
        # Save initial random model so workers have something to fetch
        nnet.save_checkpoint(os.path.join(args.checkpoint, 'best.pth.tar'))

    c = DistributedCoach(g, nnet, args)

    # Start API server in background
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    log.info(f"Trainer server started on port {args.port}")
    log.info("Starting distributed training loop...")
    c.learn()

if __name__ == "__main__":
    main()
