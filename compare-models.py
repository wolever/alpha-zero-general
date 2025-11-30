import sys
import os
import numpy as np
import logging

# Add current directory to sys.path to ensure imports work
sys.path.append(os.getcwd())

from JGGame import JGGame
from JGNet import NNetWrapper as nn
from MCTS import MCTS
from Arena import Arena
from main import TrainingArgs

import click

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


@click.command()
@click.argument("old_model", type=click.Path(exists=True))
@click.argument("new_model", type=click.Path(exists=True))
@click.option("--games", default=20, help="Number of games to play")
@click.option("--mcts-depth", default=25, help="MCTS depth")
@click.option("--mcts-sims", default=50, help="Number of MCTS simulations per move")
def main(old_model, new_model, games, mcts_depth, mcts_sims):
    print(f"Comparing models:\nOld: {old_model}\nNew: {new_model}")
    print(
        f"Playing {games} games with MCTS depth {mcts_depth} and {mcts_sims} simulations..."
    )

    # Initialize game and args
    g = JGGame()
    args = TrainingArgs()

    # Adjust args for comparison
    args.numMCTSSims = mcts_sims
    args.MCTSDepth = mcts_depth

    # Load Model 1
    n1 = nn(g, args)
    n1.load_checkpoint(old_model)
    mcts1 = MCTS(g, n1, args)

    # Load Model 2
    n2 = nn(g, args)
    n2.load_checkpoint(new_model)
    mcts2 = MCTS(g, n2, args)

    # Define players
    def player1(x):
        return np.argmax(mcts1.getActionProb(x, temp=0.1))

    def player2(x):
        return np.argmax(mcts2.getActionProb(x, temp=0.1))

    player1.name = f"Old ({os.path.basename(old_model)})"
    player2.name = f"New ({os.path.basename(new_model)})"

    # Create Arena
    arena = Arena(args, player1, player2, g)

    # Play games
    oneWon, twoWon, draws, results_in_position = arena.playGames(games, verbose=True)

    print("\nResults:")
    print(f"Old wins: {oneWon}")
    print(f"New wins: {twoWon}")
    print(f"Draws: {draws}")

    if oneWon > twoWon:
        print("Winner: Old")
    elif twoWon > oneWon:
        print("Winner: New")
    else:
        print("Tie")


if __name__ == "__main__":
    main()
