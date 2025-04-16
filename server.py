from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import json
from typing import List, Dict, Union, Optional

# Import the necessary modules from AlphaZero implementation
import sys
import os

# Add the AlphaZero directory to the path to import the modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'engine/ml/alpha-zero-general'))

from JGGame import Board, JGGame, action_unpack, action_pack
from JGNet import NNetWrapper
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
args = dotdict({
    'numMCTSSims': 25,          # Number of simulations for MCTS
    'cpuct': 1,                 # Exploration constant
    'numItersForTrainExamplesHistory': 20,
    'checkpoint': './checkpoints-JGGame/',
    'load_model': True,
    'load_folder_file': 'apr15-good.pth.tar',
})

# Load the neural network
_model: Optional[MCTS] = None

def get_model() -> MCTS:
    global _model
    if _model is None:
        print("Loading model...")
        nnet = NNetWrapper(game)
        nnet.load_checkpoint(os.path.join(args.checkpoint, args.load_folder_file))
        _model = MCTS(game, nnet, args)
        print("Model loaded.")
    return _model

# Models for request and response
class BoardRequest(BaseModel):
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
def get_moves(request: BoardRequest):
    """
    Get possible moves and their probabilities using MCTS.

    Args:
        request: JSON object containing the board state

    Returns:
        JSON object with a list of moves and their probabilities
    """
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
        move_type = (
            "add" if is_add else
            "skip" if skip else
            "move"
        )

        if not is_add:
            print("src_idx_player:", src_idx_player)
            print("board_idx:", board_canonical.src_idx_player_to_idx(1, src_idx_player))
            print("fix_idx:", fix_idx(board_canonical.src_idx_player_to_idx(1, src_idx_player)))

        moves.append({
            "type": move_type,
            "src_idx": (
                None if is_add else
                fix_idx(board_canonical.src_idx_player_to_idx(1, src_idx_player))
            ),
            "dst_idx": fix_idx(dst_idx),
            "count": int(count),
            "weight": float(a_probs[action])
        })

    print(moves)
    return {"moves": moves}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8189, reload=True)
