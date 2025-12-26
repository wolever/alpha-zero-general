from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import json
from typing import List, Dict, Union, Optional
import random

# Import the necessary modules from AlphaZero implementation
import sys
import os

# Add the AlphaZero directory to the path to import the modules
sys.path.append(os.path.join(os.path.dirname(__file__), "engine/ml/alpha-zero-general"))

from JGGame import Board, JGGame, action_unpack, action_pack
from JGNet import NNetWrapper
from main import TrainingArgs
from MCTS import MCTS
from utils import dotdict

# Create a FastAPI app
app = FastAPI(title="JG Game MCTS API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize the game, neural network, and MCTS
game = JGGame()

# Configuration for MCTS
args = TrainingArgs()

# Load the neural network
_model: Optional[MCTS] = None


def download_model_from_gcs(bucket_name: str, env: str, dest_path: str):
    try:
        from google.cloud import storage

        print(
            f"Downloading model from gs://{bucket_name}/{env}/best.pth.tar to {dest_path}..."
        )
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(f"{env}/best.pth.tar")
        blob.download_to_filename(dest_path)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download model: {e}")
        # We might want to re-raise if we strictly need the remote model
        # raise e


def get_model() -> MCTS:
    global _model
    if _model is None:
        model_env = os.environ.get("JG_ENV", "dev")  # default to dev or local
        bucket_name = os.environ.get("MODEL_BUCKET")

        checkpoint_dir = args.dataDirectory
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        load_file = args.load_folder_file
        load_path = os.path.join(checkpoint_dir, load_file)

        # If running in cloud (bucket specified), try to fetch the latest model
        if bucket_name:
            download_model_from_gcs(bucket_name, model_env, load_path)

        print(f"Loading model from {load_path}...")
        # Check if file exists before loading
        if not os.path.exists(load_path):
            print(f"WARNING: Model file {load_path} not found!")
            if bucket_name:
                raise FileNotFoundError(
                    f"Model file {load_path} not found after download attempt."
                )
            # If local dev, maybe fine? But probably not.

        nnet = NNetWrapper(game, args)
        nnet.load_checkpoint(load_path)
        _model = MCTS(game, nnet, args)
        print("Model loaded.")
    return _model


# Models for request and response
class BoardRequest(BaseModel):
    phase: str
    player_idx: int
    board: List[int]


class MoveResponse(BaseModel):
    type: str  # "move", "split", or "skip"
    src_idx: Optional[int]
    dst_idx: int
    count: int
    weight: float


class MovesResponse(BaseModel):
    moves: List[MoveResponse]


@app.post("/get-moves", response_model=MovesResponse)
def get_moves(request: BoardRequest) -> MovesResponse:
    """
    Get possible moves and their probabilities using MCTS.

    Args:
        request: JSON object containing the board state

    Returns:
        JSON object with a list of moves and their probabilities
    """
    # Handle bidding phase: return random bid between 1 and 5
    if request.phase == "bidding":
        random_bid = random.randint(1, 5)
        return {
            "moves": [
                {
                    "type": "bid",
                    "src_idx": None,
                    "dst_idx": 0,
                    "count": random_bid,
                    "weight": 1.0,
                }
            ]
        }

    mcts = get_model()
    board_arr = np.array(request.board, dtype=np.int8)
    board = Board(board_arr)

    player = {
        0: 1,
        1: -1,
    }[request.player_idx]

    print("Player:", player)
    board.display()

    board_canonical = Board(game.getCanonicalForm(board_arr, player))
    board_canonical.display()

    a_probs = np.array(mcts.getActionProb(board_canonical.arr))

    # Get the indices of non-zero probabilities
    a_where = np.where(a_probs > 0)[0].astype(np.int32)

    def fix_idx(idx: int) -> int:
        if player == 1:
            return int(idx)
        return int(board.canonicalize_idx(player, idx))

    moves = []
    for action in a_where:
        skip, src_idx_player, dst_idx, count = action_unpack(action)

        is_add = board_canonical.coins_to_add(1) > 0
        move_type = "add" if is_add else "skip" if skip else "move"

        if not is_add:
            print("src_idx_player:", src_idx_player)
            print(
                "board_idx:", board_canonical.src_idx_player_to_idx(1, src_idx_player)
            )
            print(
                "fix_idx:",
                fix_idx(board_canonical.src_idx_player_to_idx(1, src_idx_player)),
            )

        moves.append(
            {
                "type": move_type,
                "src_idx": (
                    None
                    if is_add
                    else fix_idx(
                        board_canonical.src_idx_player_to_idx(1, src_idx_player)
                    )
                ),
                "dst_idx": fix_idx(dst_idx),
                "count": int(count),
                "weight": float(a_probs[action]),
            }
        )

    return {"moves": moves}


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8189))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
