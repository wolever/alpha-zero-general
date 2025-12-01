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
@click.option("--mcts-depth", default=25, help="Default MCTS depth for both models")
@click.option(
    "--mcts-sims",
    default=50,
    help="Default number of MCTS simulations per move for both models",
)
@click.option(
    "--old-mcts-depth",
    default=None,
    type=int,
    help="MCTS depth for old model (overrides --mcts-depth)",
)
@click.option(
    "--old-mcts-sims",
    default=None,
    type=int,
    help="Number of MCTS simulations for old model (overrides --mcts-sims)",
)
@click.option(
    "--new-mcts-depth",
    default=None,
    type=int,
    help="MCTS depth for new model (overrides --mcts-depth)",
)
@click.option(
    "--new-mcts-sims",
    default=None,
    type=int,
    help="Number of MCTS simulations for new model (overrides --mcts-sims)",
)
def main(
    old_model,
    new_model,
    games,
    mcts_depth,
    mcts_sims,
    old_mcts_depth,
    old_mcts_sims,
    new_mcts_depth,
    new_mcts_sims,
):
    # Determine actual values for each model
    old_depth = old_mcts_depth if old_mcts_depth is not None else mcts_depth
    old_sims = old_mcts_sims if old_mcts_sims is not None else mcts_sims
    new_depth = new_mcts_depth if new_mcts_depth is not None else mcts_depth
    new_sims = new_mcts_sims if new_mcts_sims is not None else mcts_sims

    print(f"Comparing models:\nOld: {old_model}\nNew: {new_model}")
    print(f"Playing {games} games")
    print(f"Old model: MCTS depth {old_depth}, {old_sims} simulations")
    print(f"New model: MCTS depth {new_depth}, {new_sims} simulations")

    # Initialize game
    g = JGGame()

    # Create separate args for each model
    old_args = TrainingArgs()
    old_args.numMCTSSims = old_sims
    old_args.MCTSDepth = old_depth

    new_args = TrainingArgs()
    new_args.numMCTSSims = new_sims
    new_args.MCTSDepth = new_depth

    # Load Old Model
    n1 = nn(g, old_args)
    n1.load_checkpoint(old_model)
    mcts1 = MCTS(g, n1, old_args)

    # Load New Model
    n2 = nn(g, new_args)
    n2.load_checkpoint(new_model)
    mcts2 = MCTS(g, n2, new_args)

    # Define players
    def player1(x):
        return np.argmax(mcts1.getActionProb(x, temp=0.1))

    def player2(x):
        return np.argmax(mcts2.getActionProb(x, temp=0.1))

    player1.name = f"Old ({os.path.basename(old_model)})"
    player2.name = f"New ({os.path.basename(new_model)})"

    # Create Arena (use old_args for arena settings like maxTurnsInGame)
    arena = Arena(old_args, player1, player2, g)

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
