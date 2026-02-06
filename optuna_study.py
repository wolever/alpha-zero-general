"""
Optuna hyperparameter optimization for Alpha Zero training.

This script finds optimal training parameters by running short training trials
and measuring game characteristics (length, draw rate, balance).

Usage:
    uv run python optuna_study.py                    # Run optimization
    uv run python optuna_study.py --n-trials 50     # Run 50 trials
    uv run python optuna_study.py --show-best       # Show best parameters found

The study persists to SQLite, so you can stop and restart at any time.
"""

import argparse
import logging
import os
import shutil
import sys
import multiprocessing as mp

# Set multiprocessing start method before any other imports
mp.set_start_method("spawn", force=True)

import optuna
import coloredlogs
import wandb

from datetime import datetime
from JGGame import JGGame as Game
from JGNet import NNetWrapper as nn
from Coach import Coach
from main import TrainingArgs

log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")

# Suppress verbose optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Optimization target: games should average this many turns
TARGET_TURNS = 25

# Number of training iterations per trial (trade-off: signal vs speed)
ITERATIONS_PER_TRIAL = 5

# Number of self-play games per iteration during trials
# Matches production config (250) to ensure learning dynamics are representative
GAMES_PER_ITERATION = 250

# Weights for the composite objective (lower is better)
WEIGHT_TURN_DEVIATION = 1.0      # Penalize deviation from TARGET_TURNS
WEIGHT_DRAW_RATE = 10.0          # Heavily penalize draws
WEIGHT_P1_IMBALANCE = 5.0        # Penalize P1 vs P2 imbalance
WEIGHT_TURN_VARIANCE = 0.1       # Slightly penalize high variance


def compute_objective(metrics: dict) -> float:
    """
    Compute composite objective score from trial metrics.

    Lower is better.
    """
    turn_deviation = abs(metrics["mean_turns"] - TARGET_TURNS)
    draw_rate = metrics["draw_rate"]
    p1_imbalance = abs(metrics["p1_win_rate"] - 0.5)
    turn_variance = metrics["stddev_turns"]

    score = (
        WEIGHT_TURN_DEVIATION * turn_deviation +
        WEIGHT_DRAW_RATE * draw_rate +
        WEIGHT_P1_IMBALANCE * p1_imbalance +
        WEIGHT_TURN_VARIANCE * turn_variance
    )

    return score


def create_trial_args(trial: optuna.Trial, study_name: str) -> TrainingArgs:
    """
    Sample hyperparameters from the trial and create TrainingArgs.
    """
    # Sample parameters
    params = {
        "tempThreshold": trial.suggest_int("tempThreshold", 10, 40),
        "updateThreshold": trial.suggest_float("updateThreshold", 0.5, 0.75),
        "numMCTSSims": trial.suggest_int("numMCTSSims", 200, 800, step=50),
        "MCTSDepth": trial.suggest_int("MCTSDepth", 15, 40),
        "arenaCompare": trial.suggest_int("arenaCompare", 12, 24, step=2),
        "cpuct": trial.suggest_float("cpuct", 0.3, 1.5, log=True),
        "dirichletAlpha": trial.suggest_float("dirichletAlpha", 0.3, 1.5),
        "dirichletEpsilon": trial.suggest_float("dirichletEpsilon", 0.05, 0.35),
        "drawPenalty": trial.suggest_float("drawPenalty", -0.9, -0.1),
    }

    # Create args with trial-specific settings
    run_id = f"optuna-{study_name}-trial-{trial.number}"
    trial_checkpoint_dir = f"./checkpoints-optuna-{study_name}/trial-{trial.number}"

    args = TrainingArgs(
        runId=run_id,
        numIters=ITERATIONS_PER_TRIAL,
        numEps=GAMES_PER_ITERATION,
        dataDirectory=trial_checkpoint_dir,
        load_model=True,  # Start from existing best model
        load_examples=False,  # Don't load old examples - we want fresh games
        load_folder_file="best.pth.tar",
        # Use fewer parallel workers to reduce overhead
        numParallelSelfPlay=2,
        # Sampled parameters
        **params,
    )

    return args


def objective(trial: optuna.Trial, study_name: str, base_checkpoint: str) -> float:
    """
    Optuna objective function.

    Runs a short training session and returns a score based on game metrics.
    """
    args = create_trial_args(trial, study_name)

    # Create trial checkpoint directory
    os.makedirs(args.dataDirectory, exist_ok=True)

    # Copy the base checkpoint to the trial directory
    base_model_path = os.path.join(base_checkpoint, "best.pth.tar")
    trial_model_path = os.path.join(args.dataDirectory, "best.pth.tar")

    if os.path.exists(base_model_path):
        shutil.copy(base_model_path, trial_model_path)
        log.info(f"Copied base model from {base_model_path}")
    else:
        log.warning(f"No base model found at {base_model_path}, starting from scratch")
        args.load_model = False

    # Initialize wandb for this trial
    wandb.init(
        entity="wolever",
        project="alpha-zero-jg",
        name=args.runId,
        group=f"optuna-{study_name}",
        config={
            "trial_number": trial.number,
            "study_name": study_name,
            **args.model_dump(),
        },
        reinit=True,
    )

    try:
        # Create game and neural network
        game = Game()
        nnet = nn(game, args)

        if args.load_model:
            nnet.load_checkpoint(trial_model_path)

        # Create coach and run iterations
        coach = Coach(game, nnet, args)
        metrics = coach.run_iterations(ITERATIONS_PER_TRIAL)

        # Compute objective
        score = compute_objective(metrics)

        # Log metrics to trial
        trial.set_user_attr("mean_turns", metrics["mean_turns"])
        trial.set_user_attr("stddev_turns", metrics["stddev_turns"])
        trial.set_user_attr("draw_rate", metrics["draw_rate"])
        trial.set_user_attr("p1_win_rate", metrics["p1_win_rate"])
        trial.set_user_attr("num_games", metrics["num_games"])

        # Log to wandb
        wandb.log({
            "optuna/score": score,
            "optuna/mean_turns": metrics["mean_turns"],
            "optuna/draw_rate": metrics["draw_rate"],
            "optuna/p1_win_rate": metrics["p1_win_rate"],
        })

        log.info(
            f"Trial {trial.number} complete: score={score:.3f}, "
            f"mean_turns={metrics['mean_turns']:.1f}, "
            f"draw_rate={metrics['draw_rate']:.2%}, "
            f"p1_win_rate={metrics['p1_win_rate']:.2%}"
        )

        return score

    finally:
        wandb.finish()
        args.close()

        # Clean up trial checkpoint directory to save disk space
        # Keep only if this is the best trial so far
        try:
            if trial.study.best_trial.number != trial.number:
                shutil.rmtree(args.dataDirectory, ignore_errors=True)
                log.info(f"Cleaned up trial directory: {args.dataDirectory}")
        except ValueError:
            # No best trial yet (this is the first completed trial)
            pass


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization")
    parser.add_argument(
        "--study-name",
        default=datetime.now().strftime("%Y%m%d"),
        help="Name for the study (used for persistence and wandb grouping)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of trials to run",
    )
    parser.add_argument(
        "--base-checkpoint",
        default="./checkpoints-JGGame-v1",
        help="Directory containing the base model to start from",
    )
    parser.add_argument(
        "--show-best",
        action="store_true",
        help="Show best parameters and exit",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL (default: sqlite:///optuna-{study_name}.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a single quick trial to verify setup works",
    )
    args = parser.parse_args()

    # Set up storage in the data directory
    os.makedirs(args.base_checkpoint, exist_ok=True)
    storage = args.storage or f"sqlite:///{args.base_checkpoint}/optuna-{args.study_name}.db"

    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )

    if args.show_best:
        if len(study.trials) == 0:
            print("No trials completed yet.")
            return

        print("\n=== BEST TRIAL ===")
        best = study.best_trial
        print(f"Trial {best.number}: score = {best.value:.4f}")
        print("\nParameters:")
        for key, value in best.params.items():
            print(f"  {key}: {value}")
        print("\nMetrics:")
        for key, value in best.user_attrs.items():
            print(f"  {key}: {value}")

        print("\n=== TOP 5 TRIALS ===")
        trials_df = study.trials_dataframe()
        if len(trials_df) > 0:
            top5 = trials_df.nsmallest(5, "value")
            print(top5[["number", "value", "params_cpuct", "params_drawPenalty", "params_dirichletEpsilon"]].to_string())

        return

    # Dry run mode - run a quick single trial with minimal games
    if args.dry_run:
        log.info("=== DRY RUN MODE ===")
        log.info("Running a single quick trial to verify setup...")

        # Override constants for fast execution
        global ITERATIONS_PER_TRIAL, GAMES_PER_ITERATION
        ITERATIONS_PER_TRIAL = 1
        GAMES_PER_ITERATION = 10

        args.n_trials = 1
        args.study_name = f"dry-run-{datetime.now().strftime('%H%M%S')}"
        storage = f"sqlite:///{args.base_checkpoint}/optuna-{args.study_name}.db"

        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            direction="minimize",
        )

    # Print startup banner
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = args.n_trials - completed
    has_base_model = os.path.exists(os.path.join(args.base_checkpoint, "best.pth.tar"))

    print("\n" + "=" * 60)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"  Study name:      {args.study_name}")
    print(f"  Storage:         {storage}")
    print(f"  Base model:      {'Found' if has_base_model else 'Not found (training from scratch)'}")
    print(f"  Trials:          {completed} completed, {remaining} remaining")
    print(f"  Games/trial:     {ITERATIONS_PER_TRIAL} iterations x {GAMES_PER_ITERATION} games = {ITERATIONS_PER_TRIAL * GAMES_PER_ITERATION}")
    print(f"  wandb:           https://wandb.ai/wolever/alpha-zero-jg (group: optuna-{args.study_name})")
    print("=" * 60)

    if remaining <= 0:
        print("\nAll trials complete! Use --show-best to see results.")
        print(f"Or increase --n-trials to run more (currently {args.n_trials}).")
        return

    print(f"\nStarting optimization ({remaining} trials)...")
    print("Press Ctrl+C to stop (progress is saved automatically)\n")

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args.study_name, args.base_checkpoint),
        n_trials=args.n_trials,
        catch=(Exception,),  # Catch and log exceptions, don't crash
    )

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    if len(study.trials) > 0:
        best = study.best_trial
        print(f"\nBest trial: {best.number}")
        print(f"Best score: {best.value:.4f}")
        print("\nBest parameters:")
        for key, value in best.params.items():
            print(f"  {key}: {value}")

        # Save best params to file
        import json
        best_params_file = f"best-params-{args.study_name}.json"
        with open(best_params_file, "w") as f:
            json.dump({
                "score": best.value,
                "params": best.params,
                "user_attrs": best.user_attrs,
            }, f, indent=2)
        print(f"\nBest parameters saved to: {best_params_file}")


if __name__ == "__main__":
    main()
