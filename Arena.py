import logging
from typing import TYPE_CHECKING


import JGGame

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from main import TrainingArgs


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    game: JGGame.JGGame
    args: "TrainingArgs"

    def __init__(self, args, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.args = args
        self.player1 = player1
        self.player1.name = "old"
        self.player2 = player2
        self.player2.name = "new"
        self.game = game
        self.display = display or self.game.display

    def playGame(self, player1, player2, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [None, player1, player2]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0

        verbose = False

        for player in players[0], players[2]:
            if hasattr(player, "startGame"):
                player.startGame()

        while not self.game.getGameEnded(board, curPlayer):
            it += 1
            if verbose:
                assert self.display
                print(
                    f"Starting turn {it} for player {players[curPlayer].name} ({curPlayer})"
                )
                self.display(self.game.getCanonicalForm(board, curPlayer))

            action = players[curPlayer](board)
            valids = self.game.getValidMoves(board, 1)

            if valids[action] == 0:
                log.error(f"Action {action} is not valid!")
                log.info(f"curPlayer = {players[curPlayer].name} ({curPlayer})")
                log.info(f"action = {action}")
                log.info(f"valids = {valids}")
                log.info(f"board = {board}")
                self.game.display(self.game.getCanonicalForm(board, curPlayer))
                break

            # Notifying the opponent for the move
            opponent = players[-curPlayer]
            if hasattr(opponent, "notify"):
                opponent.notify(board, action)

            newBoard, newPlayer = self.game.getNextState(board, 1, action)
            if newPlayer == -1:
                curPlayer = -curPlayer
                board = self.game.getCanonicalForm(newBoard, -1)
            else:
                board = newBoard

            if it > self.args.maxTurnsInGame:
                print("STUCK IN LOOP")
                print(curPlayer)
                print(board)
                print(action, "=", JGGame.action_unpack(action))
                break

        winner = curPlayer * self.game.getGameEnded(board, 1)
        self.game.display(self.game.getCanonicalForm(board, curPlayer))

        for player in players[0], players[2]:
            if hasattr(player, "endGame"):
                player.endGame()

        return winner, it, curPlayer

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)

        class draw:
            name = "draw"

        players = [draw, self.player1, self.player2]
        wins = [0, 0, 0]  # [draw, old, new]
        results_in_position = [
            [0, 0, 0],  # new is first, [draw, old, new]
            [0, 0, 0],  # new is second, [draw, old, new]
        ]
        game_index = 0

        # First half: player1 starts.
        for round_num, r in enumerate([1, -1]):
            round_num += 1
            for _ in range(num):
                with self.args.time("arena/game") as game_data:
                    game_index += 1
                    gameResult, turns, last_player = self.playGame(
                        players[r], players[-r], verbose=verbose
                    )
                    wins[gameResult * r] += 1
                    results_in_position[-r][gameResult * r] += 1
                    print(
                        f"Round {round_num} game {game_index} result: {players[gameResult * r].name} ({gameResult * r}) wins "
                        f"in {turns} turns; "
                        f"total: old={wins[1]}, new={wins[2]}, draw={wins[0]}"
                    )

                    # Log each arena game outcome to Weights & Biases, if active.
                    game_data.update(
                        {
                            "arena/game/index": game_index,
                            "arena/game/phase": round_num,
                            "arena/game/starting_player": r,
                            "arena/game/starting_player_name": players[r].name,
                            "arena/game/game_result": gameResult * r,
                            "arena/game/turns": turns,
                        }
                    )

        return wins[1], wins[2], wins[0], results_in_position
