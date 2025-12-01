import sys
import os
import numpy as np
import logging
from itertools import product
from collections import defaultdict

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
@click.argument("model", type=click.Path(exists=True))
@click.option(
    "--sims", default="10,50,100", help="Comma-separated list of MCTS simulation counts"
)
@click.option("--depths", default="5,15,25", help="Comma-separated list of MCTS depths")
@click.option(
    "--games",
    default=4,
    help="Number of games for each configuration matchup",
)
def main(model, sims, depths, games):
    # Parse sims and depths
    sims_list = [int(s.strip()) for s in sims.split(",")]
    depths_list = [int(d.strip()) for d in depths.split(",")]

    # Create all configurations
    configs = list(product(sims_list, depths_list))

    print(f"Model: {model}")
    print(f"Testing {len(configs)} configurations:")
    for i, (s, d) in enumerate(configs):
        print(f"  Config {i}: sims={s}, depth={d}")
    print(f"\nGames per matchup: {games}")
    print(f"Total matchups: {len(configs) * (len(configs) - 1) // 2}")
    print(f"Total games: {len(configs) * (len(configs) - 1) // 2 * games}\n")

    # Initialize game
    g = JGGame()

    # Track results: wins[config_idx] = number of wins
    wins = defaultdict(int)
    draws = defaultdict(int)
    total_games = defaultdict(int)
    head_to_head = defaultdict(
        lambda: defaultdict(int)
    )  # head_to_head[i][j] = wins of i vs j

    # Run round-robin tournament
    for i, config1 in enumerate(configs):
        sims1, depth1 = config1

        for j, config2 in enumerate(configs):
            if i >= j:  # Skip self-play and avoid duplicates
                continue

            sims2, depth2 = config2

            print(f"\n{'=' * 60}")
            print(
                f"Matchup: Config {i} (s={sims1}, d={depth1}) vs Config {j} (s={sims2}, d={depth2})"
            )
            print(f"{'=' * 60}")

            # Create args for each config
            args1 = TrainingArgs()
            args1.numMCTSSims = sims1
            args1.MCTSDepth = depth1

            args2 = TrainingArgs()
            args2.numMCTSSims = sims2
            args2.MCTSDepth = depth2

            # Load models
            n1 = nn(g, args1)
            n1.load_checkpoint(model)
            mcts1 = MCTS(g, n1, args1)

            n2 = nn(g, args2)
            n2.load_checkpoint(model)
            mcts2 = MCTS(g, n2, args2)

            # Define players
            def player1(x):
                return np.argmax(mcts1.getActionProb(x, temp=0.1))

            def player2(x):
                return np.argmax(mcts2.getActionProb(x, temp=0.1))

            player1.name = f"Config {i}"
            player2.name = f"Config {j}"

            # Create arena
            arena = Arena(args1, player1, player2, g)

            # Play games
            oneWon, twoWon, matchDraws, _ = arena.playGames(games, verbose=False)

            # Update results
            wins[i] += oneWon
            wins[j] += twoWon
            draws[i] += matchDraws // 2  # Split draws
            draws[j] += matchDraws - matchDraws // 2
            total_games[i] += games
            total_games[j] += games

            head_to_head[i][j] = oneWon
            head_to_head[j][i] = twoWon

            print(
                f"Results: Config {i} won {oneWon}, Config {j} won {twoWon}, Draws: {matchDraws}"
            )

    # Display final results
    print(f"\n\n{'=' * 80}")
    print("FINAL RESULTS")
    print(f"{'=' * 80}\n")

    # Sort configs by wins
    sorted_configs = sorted(range(len(configs)), key=lambda i: wins[i], reverse=True)

    print(
        f"{'Rank':<6}{'Config':<8}{'Sims':<8}{'Depth':<8}{'Wins':<8}{'Draws':<8}{'Games':<8}{'Win %':<8}"
    )
    print("-" * 80)
    for rank, idx in enumerate(sorted_configs, 1):
        sims, depth = configs[idx]
        win_pct = (wins[idx] / total_games[idx] * 100) if total_games[idx] > 0 else 0
        print(
            f"{rank:<6}{idx:<8}{sims:<8}{depth:<8}{wins[idx]:<8}{draws[idx]:<8}{total_games[idx]:<8}{win_pct:<7.1f}%"
        )

    # Display head-to-head matrix
    print(f"\n\nHead-to-Head Matrix (rows vs columns):")
    print(f"{'Config':<10}", end="")
    for j in range(len(configs)):
        print(f"{j:<6}", end="")
    print()
    print("-" * (10 + 6 * len(configs)))

    for i in range(len(configs)):
        print(f"{i:<10}", end="")
        for j in range(len(configs)):
            if i == j:
                print(f"{'--':<6}", end="")
            else:
                print(f"{head_to_head[i][j]:<6}", end="")
        sims, depth = configs[i]
        print(f"  (s={sims}, d={depth})")

    print(f"\n\nBest configuration: Config {sorted_configs[0]}")
    best_sims, best_depth = configs[sorted_configs[0]]
    print(f"  Sims: {best_sims}, Depth: {best_depth}")
    print(
        f"  Win rate: {wins[sorted_configs[0]] / total_games[sorted_configs[0]] * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
