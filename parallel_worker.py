"""
Worker function for parallel self-play execution with persistent state.
Workers initialize once with the neural network, then run multiple episodes.
"""

from MCTS import MCTS
from JGNet import NNetWrapper

# Global worker state (initialized once per worker process)
_worker_game = None
_worker_nnet = None
_worker_args = None


def init_worker(args, nnet_state_dict, game_class):
    """
    Initialize worker process with neural network and game.

    Called once when worker starts, before any episodes are run.
    This avoids reloading the network for every episode.

    Args:
        args: Training arguments
        nnet_state_dict: Neural network weights
        game_class: Game class to instantiate
    """
    global _worker_game, _worker_nnet, _worker_args

    # Create game instance (lightweight)
    _worker_game = game_class()

    # Create and load neural network (expensive, done once per worker)
    _worker_nnet = NNetWrapper(_worker_game, args)
    _worker_nnet.nnet.load_state_dict(nnet_state_dict)
    _worker_nnet.nnet.eval()

    _worker_args = args


def execute_episode_worker(worker_id):
    """
    Execute a single episode using the pre-loaded network.

    Args:
        worker_id: Identifier for this episode

    Returns:
        Tuple of (trainExamples, episodeStep, winnerAbs, duration)
    """
    import time

    global _worker_game, _worker_nnet, _worker_args

    start_time = time.time()

    # Create fresh MCTS for this episode (uses pre-loaded network)
    mcts = MCTS(_worker_game, _worker_nnet, _worker_args)

    # Use shared executeEpisode implementation
    from Coach import Coach

    result = Coach.executeEpisode(mcts, _worker_game, _worker_args)

    duration = time.time() - start_time

    # Return original result + duration
    return (*result, duration)
