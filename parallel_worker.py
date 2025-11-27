"""
Worker function for parallel self-play execution.
This needs to be at module level for multiprocessing to pickle it.
"""

from MCTS import MCTS
from JGNet import NNetWrapper


def execute_episode_worker(args_tuple):
    """
    Worker function to execute a single episode in a separate process.

    Args:
        args_tuple: Tuple of (worker_id, args, nnet_state_dict, game_class)

    Returns:
        Tuple of (trainExamples, episodeStep, winnerAbs)
    """
    worker_id, args, nnet_state_dict, game_class = args_tuple

    # Create a fresh neural network and load the weights
    game = game_class()
    nnet = NNetWrapper(game, args)
    nnet.nnet.load_state_dict(nnet_state_dict)
    nnet.nnet.eval()  # Set to eval mode for inference

    # Create MCTS for this episode
    mcts = MCTS(game, nnet, args)

    # Use the shared executeEpisode implementation from Coach
    from Coach import Coach

    return Coach.executeEpisode(mcts, game, args)
