from playgroundrl.client import *
from playgroundrl.actions import *
from playgroundrl.args import get_arguments
from word2vec import *
# from util import parse_arguments

BOARD_SIZE = 25

class TestCodenames(PlaygroundClient):
    def __init__(self, **kwargs):
        super().__init__(
            GameType.CODENAMES,
            model_name="bob-v3",
            **kwargs
        )

    def get_open_squares(self, state: CodenamesState):
        open_squares = []
        for i in range(BOARD_SIZE):
            if state.guessed[i] == "UNKNOWN":
                open_squares.append((i, state.words[i]))
        return open_squares

    def get_spymaster_squares(self, state: CodenamesState):
        team_squares = []
        opp_squares = []
        for i in range(BOARD_SIZE):
            if state.guessed[i] == "UNKNOWN":
                if state.actual[i] == state.color:
                    team_squares.append(state.words[i])
                else:
                    opp_squares.append(state.words[i])
        return team_squares, opp_squares

    def callback(self, state: CodenamesState, reward):
        if state.player_moving_id not in self.player_ids:
            return None

        if state.role == "GIVER":
            team_squares, opp_squares = self.get_spymaster_squares(state)
            word, count = generate_clue(team_squares, opp_squares)
            return CodenamesSpymasterAction(word, count)
        elif state.role == "GUESSER":
            open_squares = self.get_open_squares(state)
            return CodenamesGuesserAction(
                guesses=generate_guess(open_squares, state.clue, state.count),
            )

    def gameover_callback(self):
        pass

if __name__ == "__main__":
    # args = parse_arguments("codenames")
    init_args, run_args = get_arguments()
    t = TestCodenames(**init_args)
    t.run(**run_args)