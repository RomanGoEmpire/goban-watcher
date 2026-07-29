from copy import deepcopy
from enum import IntEnum, auto
from pathlib import Path

from sgfmill import sgf

from src import GRID_SIZE


class Cell(IntEnum):
    EMPTY = 0
    BLACK = auto()
    WHITE = auto()

    def __str__(self) -> str:
        return {"EMPTY": "e", "BLACK": "b", "WHITE": "w"}[self.name]


class Game:
    def __init__(self):
        self.move: int = 0
        self.captured_black: int = 0
        self.captured_white: int = 0
        self.board: list[list[Cell]] = self._initialize_board()
        self.board_history: list = [self.board]
        self.neighbors = self.get_neighbors()

    def reset(self) -> None:
        self.move = 0
        self.captured_black = 0
        self.captured_white = 0
        self.board = self._initialize_board()
        self.board_history = [self.board]

    @staticmethod
    def _initialize_board() -> list[list[Cell]]:
        return [[Cell.EMPTY for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    def current_and_opponent_color(self) -> tuple[Cell, Cell]:
        """Returns the color of the current player and the opponent based on self.move"""
        current_color, opponent_color = (
            (Cell.BLACK, Cell.WHITE),
            (Cell.WHITE, Cell.BLACK),
        )[self.move & 1]
        return current_color, opponent_color

    def add_move(self, x: int, y: int) -> None:
        """Adds Cell at position x, y for the player whose turn it is."""
        assert self.is_empty((x, y)), "Position occupied"

        current_color, opponent_color = self.current_and_opponent_color()

        # create copy since move might be invalid and should not change the current boards
        board_after_capture = deepcopy(self.board)

        # add move
        board_after_capture[y][x] = current_color

        # iterate over each neighbor and check if it has liberties
        neighbors = self.neighbors[y][x]
        captures = 0
        for neighbor_x, neighbor_y in neighbors:
            group, liberties = self.get_group_and_liberties(
                neighbor_x, neighbor_y, board_after_capture
            )

            if liberties > 0:
                continue

            captures += len(group)
            for cell in group:
                cell_x, cell_y = cell
                board_after_capture[cell_y][cell_x] = Cell.EMPTY

        # stone does not capture any stones and has no liberties afterwards
        _, liberties = self.get_group_and_liberties(x, y, board_after_capture)

        assert liberties > 0, "Move would lead to suicide"

        if len(self.board_history) > 2:
            # board position repeats => ko rule
            assert board_after_capture != self.board_history[-2], (
                "Move would lead to invalid repetition (ko)"
            )

        if opponent_color == Cell.WHITE:
            self.captured_white += captures
        elif opponent_color == Cell.BLACK:
            self.captured_black += captures

        self.move += 1
        self.board = board_after_capture
        self.board_history.append(board_after_capture)

    def remove_move(self) -> None:
        self.board = self.board_history[-2]
        self.board_history.pop()
        self.move -= 1

    def get_group_and_liberties(
        self, x: int, y: int, board
    ) -> tuple[list[tuple[int, int]], int]:
        color = self.get_color((x, y), board)

        queue = set()
        for neighbor in self.neighbors[y][x]:
            queue.add(neighbor)

        group = {(x, y)}
        liberties = 0

        while queue:
            cell = queue.pop()

            if cell in group:
                continue

            if self.is_empty((cell), board):
                liberties += 1

            if self.get_color(cell, board) == color:
                group.add(cell)
                cell_x, cell_y = cell
                queue.update(self.neighbors[cell_y][cell_x])
        return list(group), liberties

    @staticmethod
    def get_neighbors() -> list:
        """Returns lookup table of neighbors for each x, y."""
        lookup = []
        for y in range(GRID_SIZE):
            neighbor_row = []
            for x in range(GRID_SIZE):
                neighbors = []
                # left
                if x > 0:
                    neighbors.append((x - 1, y))
                # right
                if x < GRID_SIZE - 1:
                    neighbors.append((x + 1, y))
                # top
                if y > 0:
                    neighbors.append((x, y - 1))
                # bottom
                if y < GRID_SIZE - 1:
                    neighbors.append((x, y + 1))
                neighbor_row.append(neighbors)
            lookup.append(neighbor_row)
        return lookup

    def get_color(self, coordinates: tuple[int, int], board=None) -> Cell:
        board = self.board if board is None else board
        x, y = coordinates
        return board[y][x]

    def is_empty(self, coordinates: tuple[int, int], board=None) -> bool:
        board = self.board if board is None else board
        x, y = coordinates
        return board[y][x] == Cell.EMPTY.value

    def add_sgf(self, filename: Path) -> None:
        """Plays out complete sgf by adding each move."""
        with open(filename, "rb") as f:
            main_sequence = sgf.Sgf_game.from_bytes(f.read()).get_main_sequence()

        for node in main_sequence:
            _, move = node.get_move()
            if move:
                x, y = move
                x, y = y, GRID_SIZE - 1 - x  # change due sgf coordinate order
                self.add_move(x, y)
