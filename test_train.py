import os
from main import TrainingArgs, Coach, Game, nn

def test_train():
    # Minimal args for fast testing
    args = TrainingArgs(
        runId="test_run",
        numIters=2,
        numEps=2,
        numMCTSSims=10, # Very low for speed
        load_model=False,
        load_examples=False,
        arenaCompare=2,
    )

    print("Initializing Game...")
    g = Game()
    print("Initializing Net...")
    nnet = nn(g, args)
    print("Initializing Coach...")
    c = Coach(g, nnet, args)

    print("Starting Learn...")
    c.learn()
    print("Test Complete!")

if __name__ == "__main__":
    test_train()
