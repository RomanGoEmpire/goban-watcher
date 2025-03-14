import json
import subprocess

import numpy as np

from src import GRID_SIZE, KATAGO_PATH, KOMI
from src.utils.game import Cell
from src.utils.sgf_helper import (
    convert_cell_to_player_color,
    convert_move_to_coordinate,
    convert_coordinate_to_move,
    opponent,
)


def send_position_into_analysis(process, data: dict) -> None:
    """Sends game position to katago."""
    # must be one line string
    data_string = json.dumps(data).replace("\n", "") + "\n"
    process.stdin.write(data_string)
    process.stdin.flush()


def start_katago_process() -> subprocess.Popen[str]:
    model_path = str(KATAGO_PATH.joinpath("models", "b28c512nbt.bin.gz"))
    config_path = str(KATAGO_PATH.joinpath("configs", "analysis_example.cfg"))

    # This might not work on your system, if it can't find katago
    # Change "katago" to absolute path of katago executable
    process = subprocess.Popen(
        ["katago", "analysis", "-config", config_path, "-model", model_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    return process


def get_best_variation(
    process,
    board: list[list[Cell]],
    moves: list[tuple[Cell, tuple[int, int]]],
    current_player_cell: Cell,
) -> list[tuple[Cell, tuple[int, int]]]:
    initial_stones = convert_board_to_initial_stones(board)
    sgf_moves = convert_to_sgf_moves(moves)
    current_player = convert_cell_to_player_color(current_player_cell)
    variations = possible_variations(sgf_moves, current_player)

    amount_moves = len(moves)
    analyse_turns = [i for i in range(amount_moves)]

    for i, moves in enumerate(variations):
        data = {
            "id": str(i),  # has to be string
            "initialStones": initial_stones,
            "moves": moves,
            "rules": "tromp-taylor",
            "komi": KOMI,
            "boardXSize": GRID_SIZE,
            "boardYSize": GRID_SIZE,
            "analyzeTurns": analyse_turns,
        }
        send_position_into_analysis(process, data)

    results = {str(i): [] for i in range(len(variations))}
    for i in range(amount_moves * len(variations)):
        while True:
            new_data = process.stdout.readline().strip()
            if new_data:
                result = json.loads(new_data)
                results[result["id"]].append(result)
                break

    diffs = [
        (
            k,
            abs(
                np.array([m["rootInfo"]["scoreLead"] for m in v])
                - np.array(
                    [v[max(0, i - 1)]["rootInfo"]["scoreLead"] for i in range(len(v))]
                )
            ).mean(),
        )
        for k, v in results.items()
    ]
    diffs.sort(key=lambda r: r[1])

    best_variation = variations[int(diffs[0][0])]
    best_sequence = []
    for color, sgf_position in best_variation:
        cell = Cell.BLACK if color == "b" else Cell.WHITE
        position = convert_coordinate_to_move(sgf_position)
        best_sequence.append((cell, position))
    return best_sequence


def convert_to_sgf_moves(
    moves: list[tuple[Cell, tuple[int, int]]],
) -> list[tuple[str, str]]:
    sgf_moves = []
    for cell, position in moves:
        player_color = convert_cell_to_player_color(cell)
        sgf_position = convert_move_to_coordinate(*position)
        sgf_moves.append((player_color, sgf_position))
    return sgf_moves


def convert_board_to_initial_stones(board: list[list[Cell]]) -> list[tuple[str, str]]:
    initial_stones = []
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            cell = board[y][x]
            if cell == Cell.EMPTY:
                continue

            player_color = convert_cell_to_player_color(cell)
            sgf_position = convert_move_to_coordinate(x, y)
            initial_stones.append((player_color, sgf_position))
    return initial_stones


def possible_variations(
    moves: list[tuple[str, str]], current_player: str, branch: list = []
) -> list:
    """Create a list of all possible variations with the moves provided."""
    if not moves:
        return [branch]

    variations = []
    player_moves = [move for move in moves if move[0] == current_player]

    for pm in player_moves:
        new_moves = [m for m in moves if m != pm]
        new_branch = branch + [pm]
        variations += possible_variations(
            new_moves, opponent(current_player), new_branch
        )

    return variations
