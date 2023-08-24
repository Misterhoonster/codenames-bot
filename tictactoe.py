import time
from playgroundrl.client import *
from util import parse_arguments

class TestTicTacToe(PlaygroundClient):
    def __init__(
        self, auth_file: str, render: bool = False, self_training: bool = False
    ):
        super().__init__(
            game=GameType.TICTACTOE,
            model_name="tutorial-tictactoe",
            auth_file=auth_file,
            render_gameplay=render,
        )
        self.self_training = self_training

    def callback(self, state: TicTacToeState, reward):
        if state.player_moving_id not in self.player_ids:
            return None

        if self.self_training:
            time.sleep(1)

        board = state.board

        player = state.player_moving_id
        opposing_player = (state.player_moving_id + 1) % 2

        for i in range(0, 3):
            for j in range(0, 3):
                if board[i][j] == -1:
                    # Choose first open state
                    return i * 3 + j

    def gameover_callback(self):
        pass

if __name__ == "__main__":
    args = parse_arguments("tictactoe")
    t = TestTicTacToe(args.authfile, args.render, args.self_training)
    t.run(
        pool=Pool(args.pool),
        num_games=args.num_games,
        self_training=args.self_training,
        maximum_messages=500000,
        game_parameters=json.loads(args.params),
    )