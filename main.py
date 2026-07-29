import argparse
import json
import os
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike
from sgfmill import sgf

from src import (
    BACKUP_PATH,
    CELL_SIZE,
    CORNER_INDEXES,
    GRID_SIZE,
    HALF_CELL_SIZE,
    HOSHIS,
    RECORDING_PATH,
    SCREEN_SIZE,
)
from src.stone_classification import load_rf
from src.utils.colors import Color
from src.utils.custom_logger import get_color_logger
from src.utils.cv2_helper import (
    add_grid,
    blur_and_sharpen,
    convert_to_top_down,
    default_corners,
)
from src.utils.game import Cell, Game
from src.utils.katago_helper import get_best_variation, start_katago_process

RF_PATH = "weights/random_forest_model.pkl"

MAX_DEPTH = 9
AMOUNT_IDENTICAL_IMAGES = 15


def default_mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        global default_mouse
        default_mouse = [x, y]


def setup_corners(cap: cv2.VideoCapture) -> list[list[int]]:
    ret, frame = cap.read()

    shape = frame.shape
    y, x, _ = shape
    corners = default_corners(shape)

    selected_corner = None

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 10):  # Enter/Return
            break
        if key == ord("r"):
            logger.debug("Reset corners")
            corners = default_corners(shape)
            continue
        # TODO(2025-09-12 15:09:30): Maybe add "l" to load corners instead of passing a cli flag?
        if key == ord("s"):
            save_corners_to_file(corners)
            continue

        ret, frame = cap.read()
        display_img = deepcopy(frame)

        # Handling connecting issues with Iphone
        if not ret:
            time.sleep(2)
            continue

        display_img = cv2.resize(display_img, (x, y))

        for index in CORNER_INDEXES:
            if key == ord(str(index + 1)):
                selected_corner = index if selected_corner != index else None

        if selected_corner is not None:
            corners[selected_corner] = default_mouse

        for index, corner in enumerate(corners):
            cv2.line(display_img, corner, corners[index - 1], Color.GREEN.value)

        transformed_frame = convert_to_top_down(frame, corners)
        display_transformed_frame = add_grid(transformed_frame)

        cv2.imshow("Default", display_img)
        cv2.imshow("Transformed", display_transformed_frame)
    cv2.destroyAllWindows()
    return corners


def save_corners_to_file(corners: list[list[int]]) -> None:
    with open(Path.joinpath(BACKUP_PATH, "corners.json"), "w") as f:
        json.dump(corners, f, indent=4)
    logger.info(f"Saved corners to: {BACKUP_PATH}")


def try_to_load_corners_from_file() -> list[list[int]] | None:
    if not Path.exists(corners_path):
        logger.warning(
            f"No corner backup found in: {corners_path}. Manual setup required"
        )
        return None
    with open(corners_path) as f:
        return json.load(f)


def classify_all_cells(model, frame: MatLike) -> list[Cell]:
    low_offsets = np.arange(0, GRID_SIZE) * CELL_SIZE - HALF_CELL_SIZE
    high_offsets = np.arange(1, GRID_SIZE + 1) * CELL_SIZE + HALF_CELL_SIZE

    low_offsets = np.clip(low_offsets, 0, frame.shape[0])
    high_offsets = np.clip(high_offsets, 0, frame.shape[1])

    cells = np.array(
        [
            cv2.resize(
                frame[
                    low_offsets[y] : high_offsets[y], low_offsets[x] : high_offsets[x]
                ],
                (CELL_SIZE, CELL_SIZE),
            ).flatten()
            for y in range(GRID_SIZE)
            for x in range(GRID_SIZE)
        ]
    )

    results = model.predict(cells)
    results = [Cell(r) for r in results]

    return results


def base_visual_board() -> MatLike:
    frame = np.full((SCREEN_SIZE, SCREEN_SIZE, 3), Color.BROWN.value, dtype=np.uint8)

    # draw lines
    for i in range(GRID_SIZE):
        start = HALF_CELL_SIZE
        end = SCREEN_SIZE - HALF_CELL_SIZE
        height = i * CELL_SIZE + HALF_CELL_SIZE
        frame = cv2.line(frame, (start, height), (end, height), Color.BLACK.value, 1)
        frame = cv2.line(frame, (height, start), (height, end), Color.BLACK.value, 1)

    # draw hoshis
    for x in HOSHIS:
        for y in HOSHIS:
            frame = cv2.circle(
                frame,
                (HALF_CELL_SIZE + x * CELL_SIZE, HALF_CELL_SIZE + y * CELL_SIZE),
                int(CELL_SIZE * 0.1),
                Color.BLACK.value,
                -1,
            )
    return frame


def add_stones_to_visual(frame, board: list[list[Cell]]) -> None:
    for col in range(GRID_SIZE):
        for row in range(GRID_SIZE):
            cell = board[col][row].value
            if cell == 1:
                color = Color.BLACK.value
            elif cell == 2:
                color = Color.WHITE.value
            else:
                color = None

            if color:
                frame = cv2.circle(
                    frame,
                    (
                        HALF_CELL_SIZE + CELL_SIZE * row,
                        HALF_CELL_SIZE + CELL_SIZE * col,
                    ),
                    HALF_CELL_SIZE,
                    color,
                    -1,
                )
    return frame


def diff_between_boards(current_board: list[list[Cell]], new_board: list[Cell]) -> list:
    diff = []
    for i in range(len(new_board)):
        col = i // GRID_SIZE
        row = i % GRID_SIZE
        current_cell = current_board[col][row]
        new_cell = new_board[i]

        if current_cell != new_cell:
            diff.append(
                {"position": (row, col), "current": current_cell, "new": new_cell}
            )
    return diff


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example CLI with argparse")
    parser.add_argument(
        "-c", "--camera", type=int, default=0, help="Camera index. Starts at 0"
    )
    parser.add_argument(
        "--enable-katago",
        action="store_true",
        default=False,
        dest="use_katago",
        help="Activates KataGo for move reconstruction",
    )
    parser.add_argument(
        "--identical-frames",
        type=int,
        default=AMOUNT_IDENTICAL_IMAGES,
        help="Amount of identical frames required to trigger stone evaluation. Default is 15 for 30fps -> 0.5s",
    )
    parser.add_argument(
        "--use-saved-corners",
        action="store_true",
        default=False,
        dest="use_saved_corners",  # optional: name the positive concept
        help='Uses saved corners instead of manual setup. Press "s" after manual setup is done to save corners',
    )
    return parser.parse_args()


def convert_to_sgf_position(position):
    return (GRID_SIZE - 1 - position[1], position[0])


def add_move_to_sgf(
    child: sgf.Tree_node, color: str, position: tuple[int, int]
) -> sgf.Tree_node:
    sgf_position = convert_to_sgf_position(position)
    new_child = child.new_child()
    new_child.set_move(color.lower(), sgf_position)
    return new_child


def remove_last_move_from_sgf(node: sgf.Tree_node) -> sgf.Tree_node:
    parent = node.parent
    if parent is None:
        raise ValueError("Cannot remove the root node of the SGF tree")

    node.delete()
    return parent


def save_sgf_to_file(sgf_game) -> None:
    logger.debug("Saving SGF")
    with open((f"{RECORDING_PATH}/{timestamp}.sgf"), "wb") as f:
        f.write(sgf_game.serialise())


def add_chaos_to_sgf(child: sgf.Tree_node, moves: list[dict]) -> sgf.Tree_node:
    new_child = child.new_child()

    # in sgf add move via AE|AB|AW must include all moves so they must be group by Cell
    # e.g AB[dd][de] => add two black stones next to each other in "edit" mode
    black_moves = [m for m in moves if m["new"] == Cell.BLACK.value]
    white_moves = [m for m in moves if m["new"] == Cell.WHITE.value]
    empty_moves = [m for m in moves if m["new"] == Cell.EMPTY.value]

    groups = [black_moves, white_moves, empty_moves]
    for group in groups:
        if len(group) == 0:
            continue
        color = str(group[0]["new"])
        positions = [convert_to_sgf_position(move["position"]) for move in group]
        new_child.set(f"A{color.upper()}", positions)
    return new_child


def main() -> None:
    args = get_args()

    os.makedirs(BACKUP_PATH, exist_ok=True)
    os.makedirs(RECORDING_PATH, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    logger.info(f"Using Camera {args.camera}")

    if not cap.isOpened():
        logger.error("Camera could not be openend")
        exit(1)

    # TODO(2025-09-12 23:09:85): merge default and transformed into a single one
    # Mouse movement will only be allowed up to border of default one
    cv2.namedWindow("Default")
    cv2.setMouseCallback("Default", default_mouse_callback)
    cv2.namedWindow("Transformed")

    corners = None
    if args.use_saved_corners:
        corners = try_to_load_corners_from_file()

    if corners:
        logger.debug("Corner backup found and loaded")
    else:
        corners = setup_corners(cap)

    model = load_rf(RF_PATH)

    katago_process = None
    if args.use_katago:
        logger.debug("Katago enabled. Trying to start it ...")
        katago_process = start_katago_process()

    logger.debug("Katago started successfully")

    visual_board = base_visual_board()

    last_results = []

    # handles game logic
    game = Game()

    # handles game recording
    sgf_game = sgf.Sgf_game(size=19)
    sgf_root = sgf_game.get_root()
    sgf_child = sgf_root.new_child()

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in [ord("q"), 27]:  # 27 = ESC
            break

        _, frame = cap.read()
        image = convert_to_top_down(frame, corners)
        image = blur_and_sharpen(image)
        classified_cells = classify_all_cells(model, image)

        last_results.append(classified_cells)

        canvas = add_stones_to_visual(visual_board.copy(), game.board)
        complete = np.hstack((image, canvas))  # type: ignore #
        cv2.imshow("Complete", complete)

        # last_results are empty at the start and have to be filled up at the start
        # this initialisation process is ignored and the recording only starts
        # when enough frames exists to be compared
        if len(last_results) < args.identical_frames:
            continue
        # remove oldest result
        last_results.pop(0)

        current_player, opponent_player = game.current_and_opponent_color()

        # not all the same => movement exists
        if not all(r == classified_cells for r in last_results):
            continue

        changes = diff_between_boards(game.board, classified_cells)

        if len(changes) == 0:
            continue

        # single new move
        if (
            len(changes) == 1
            and changes[0]["current"] == Cell.EMPTY.value
            and changes[0]["new"] == current_player
        ):
            position = changes[0]["position"]
            if game.is_empty(position):
                logger.info(f"New Move: {changes[0]['new']} - {position}")
                game.add_move(*position)
                sgf_child = add_move_to_sgf(sgf_child, str(changes[0]["new"]), position)
                save_sgf_to_file(sgf_game)
                continue
            else:
                logger.fatal(f"Illegal Move: {changes[0]['new']} - {position}")
                exit(1)

        # two new moves
        if (
            len(changes) == 2
            and all(d["current"] == 0 for d in changes)  # only additions
            # one move of each color
            and set([changes[0]["new"], changes[1]["new"]])
            == set([Cell.BLACK.value, Cell.WHITE.value])
        ):
            logger.info("Two Moves")
            ordered_changes = (
                changes if changes[0]["new"] == current_player else changes[::-1]
            )
            for move in ordered_changes:
                logger.info(f"Move: {move['new']} - {move['position']}")
                game.add_move(*move["position"])
                sgf_child = add_move_to_sgf(
                    sgf_child, str(move["new"]), move["position"]
                )
            save_sgf_to_file(sgf_game)
            continue

        # capture happended. One new move of current_player and 1+ removals of opponent
        moves_added = [c for c in changes if c["current"] == 0]
        moves_removed = [c for c in changes if c["new"] == 0]
        removed_colors = set([rm["current"] for rm in moves_removed])
        if (
            len(moves_added) == 1
            and moves_added[0]["new"] == current_player
            and len(removed_colors) == 1
            and moves_removed[0]["current"] == opponent_player
        ):
            move = moves_added[0]
            logger.info(f"Move (Capture): {move['new']} - {move['position']}")
            game.add_move(*move["position"])
            sgf_child = add_move_to_sgf(sgf_child, str(move["new"]), move["position"])
            save_sgf_to_file(sgf_game)
            continue

        # multiple moves added
        if katago_process:
            valid_addition = all(c["current"] == Cell.EMPTY for c in changes)
            amount_player_stones = len(
                [c["new"] for c in changes if c["new"] == current_player]
            )
            amount_opponent_stones = len(
                [c["new"] for c in changes if c["new"] == opponent_player]
            )
            # New moves must be evenly distributed between both players
            valid_offset = [0, 1]
            valid_amounts = (
                amount_player_stones - amount_opponent_stones in valid_offset
            )

            if (
                valid_addition
                and valid_amounts
                # TODO(2025-09-12 18:09:41): add it as parameter. Add hint that the amount of variations increases drastically.
                and 1 < len(moves_added) < MAX_DEPTH
                and len(moves_removed) == 0
            ):
                logger.info(
                    "Multiple new moves. Using katago to guess the correct sequence"
                )
                new_moves = []
                for move in moves_added:
                    new_moves.append((move["new"], move["position"]))

                sequence = get_best_variation(
                    katago_process,
                    game.board,
                    new_moves,
                    current_player,
                )
                for move in sequence:
                    color, position = move
                    logger.info(f"Move: {color} - {position}")
                    game.add_move(*position)
                    sgf_child = add_move_to_sgf(sgf_child, str(color), position)
                save_sgf_to_file(sgf_game)
                continue

        # rollback: takeback or single missdetection
        if len(moves_removed) == 1:
            move = moves_removed[0]
            x, y = move["position"]

            if (
                game.board[y][x] == move["current"]
                and game.board_history[-2][y][x] == move["new"]
            ):
                game.remove_move()
                sgf_child = remove_last_move_from_sgf(sgf_child)
            logger.info(f"Move (Takeback): {move['position']}")
            save_sgf_to_file(sgf_game)
            continue

        if len(moves_added) > 0 or len(moves_removed) > 0:
            # Chaos.
            # Possible reasons:
            #   A -> capture and moves after
            #   B -> capture and stone was moved or wrongly recognized at the start
            #   C -> illegal moves

            logger.warning(
                "Game State changed drastically! Unable to extract sequence and therefore updating the board as whole to the new state"
            )
            game.board = [
                classified_cells[i * GRID_SIZE : i * GRID_SIZE + GRID_SIZE]
                for i in range(GRID_SIZE)
            ]
            sgf_child = add_chaos_to_sgf(sgf_child, moves_added + moves_removed)
            save_sgf_to_file(sgf_game)

    cap.release()

    save_sgf_to_file(sgf_game)

    if katago_process:
        katago_process.terminate()


if __name__ == "__main__":
    default_mouse = [0, 0]
    timestamp = str(datetime.now().strftime("%d%m%Y%H%M%S"))
    corners_path = Path.joinpath(BACKUP_PATH, "corners.json")
    logger = get_color_logger()
    main()
