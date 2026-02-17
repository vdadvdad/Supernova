import os
import multiprocessing as mp
import time
import tkinter as tk
from tkinter import messagebox

from gravity_chess import GameState, Move, WHITE, BLACK, iterative_deepening_best_move_with_score

BOARD_SIZE = 8
SQUARE_SIZE = 64

PIECE_IMAGE_FILES = {
    "P": "white_pawn.png",
    "N": "white_knight.png",
    "B": "white_bishop.png",
    "R": "white_rook.png",
    "Q": "white_queen.png",
    "K": "white_king.png",
    "p": "black_pawn.png",
    "n": "black_knight.png",
    "b": "black_bishop.png",
    "r": "black_rook.png",
    "q": "black_queen.png",
    "k": "black_king.png",
}


class GravityChessApp:
    def __init__(self, root: tk.Tk, depth: int = 5) -> None:
        self.root = root
        self.depth = depth
        self.state = GameState.starting()
        self.user_color = WHITE
        self.selected: tuple[int, int] | None = None
        self.engine_thinking = False
        self.engine_queue: mp.Queue = mp.Queue()
        self.engine_process: mp.Process | None = None
        self.engine_generation = 0

        self.canvas = tk.Canvas(
            root,
            width=BOARD_SIZE * SQUARE_SIZE,
            height=BOARD_SIZE * SQUARE_SIZE,
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, columnspan=2)
        self.canvas.bind("<Button-1>", self.on_click)

        self.status = tk.Label(root, text="Your move", anchor="w")
        self.status.grid(row=1, column=0, sticky="we")

        self.reset_button = tk.Button(root, text="New Game", command=self.reset)
        self.reset_button.grid(row=1, column=1, sticky="e")

        self.images = self._load_images()
        self.draw_board()
        self.root.after(50, self._poll_engine_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _load_images(self) -> dict[str, tk.PhotoImage]:
        images: dict[str, tk.PhotoImage] = {}
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pieces")
        target_w = int(SQUARE_SIZE * 0.8)
        target_h = int(SQUARE_SIZE * 0.8)
        for piece, filename in PIECE_IMAGE_FILES.items():
            path = os.path.join(base, filename)
            image = tk.PhotoImage(file=path)
            image = self._fit_image(image, target_w, target_h)
            images[piece] = image
        return images

    def _fit_image(
        self, image: tk.PhotoImage, target_w: int, target_h: int
    ) -> tk.PhotoImage:
        width = image.width()
        height = image.height()
        if width == target_w and height == target_h:
            return image
        scale_down = max(width // target_w, height // target_h)
        if scale_down > 1:
            return image.subsample(scale_down, scale_down)
        scale_up = min(
            max(1, target_w // max(1, width)),
            max(1, target_h // max(1, height)),
        )
        if scale_up > 1:
            return image.zoom(scale_up, scale_up)
        return image

    def reset(self) -> None:
        if self.engine_thinking:
            self._stop_engine_process()
        self.state = GameState.starting()
        self.selected = None
        self.status.config(text="Your move")
        self.draw_board()

    def draw_board(self) -> None:
        self.canvas.delete("all")
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x1 = c * SQUARE_SIZE
                y1 = (BOARD_SIZE - 1 - r) * SQUARE_SIZE
                x2 = x1 + SQUARE_SIZE
                y2 = y1 + SQUARE_SIZE
                is_light = (r + c) % 2 == 0
                color = "#f0d9b5" if is_light else "#b58863"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)
                if self.selected == (r, c):
                    self.canvas.create_rectangle(
                        x1 + 2, y1 + 2, x2 - 2, y2 - 2, outline="#3f8efc", width=3
                    )
                piece = self.state.piece_at(r, c)
                if piece:
                    self.canvas.create_image(
                        x1 + SQUARE_SIZE // 2,
                        y1 + SQUARE_SIZE // 2,
                        image=self.images[piece],
                    )

    def on_click(self, event: tk.Event) -> None:
        if self.engine_thinking:
            return
        c = event.x // SQUARE_SIZE
        r = BOARD_SIZE - 1 - (event.y // SQUARE_SIZE)
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            return
        if self.state.side_to_move != self.user_color:
            return

        if self.selected is None:
            piece = self.state.piece_at(r, c)
            if piece and self.state.side_of(piece) == self.user_color:
                self.selected = (r, c)
                self.draw_board()
            return

        if self.selected == (r, c):
            self.selected = None
            self.draw_board()
            return

        self.try_player_move(self.selected, (r, c))

    def try_player_move(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        legal = self._find_legal_move(start, end)
        if not legal:
            self.selected = None
            self.draw_board()
            return
        self.selected = None
        self.state = self.state.apply_move(legal)
        self.draw_board()
        winner = self.state.is_terminal()
        if winner:
            self.status.config(text=f"Winner: {winner}")
            messagebox.showinfo("Game Over", f"Winner: {winner}")
            return
        self.start_engine_move()

    def _find_legal_move(self, start: tuple[int, int], end: tuple[int, int]) -> Move | None:
        for move in self.state.generate_moves():
            if move.start == start and move.end == end:
                return move
        return None

    def start_engine_move(self) -> None:
        self.engine_thinking = True
        self.status.config(text="Engine thinking...")
        self.engine_generation += 1
        generation = self.engine_generation
        if self.engine_process and self.engine_process.is_alive():
            return
        state_copy = self.state.copy()
        ctx = mp.get_context("spawn")
        self.engine_process = ctx.Process(
            target=_engine_worker,
            args=(state_copy, self.depth, generation, self.engine_queue),
        )
        self.engine_process.start()

    def _poll_engine_queue(self) -> None:
        try:
            generation, move, score, nodes, elapsed = self.engine_queue.get_nowait()
        except Exception:
            self.root.after(50, self._poll_engine_queue)
            return
        if generation != self.engine_generation:
            self.root.after(50, self._poll_engine_queue)
            return
        if self.engine_process and not self.engine_process.is_alive():
            self.engine_process = None
        self.finish_engine_move(move, score, nodes, elapsed)
        self.root.after(50, self._poll_engine_queue)

    def _stop_engine_process(self) -> None:
        self.engine_generation += 1
        if self.engine_process and self.engine_process.is_alive():
            self.engine_process.terminate()
            self.engine_process.join(timeout=0.2)
        self.engine_process = None
        self.engine_thinking = False

    def on_close(self) -> None:
        self._stop_engine_process()
        self.root.destroy()

    def finish_engine_move(
        self,
        move: Move | None,
        score: float,
        nodes: int,
        elapsed: float,
    ) -> None:
        self.engine_thinking = False
        if move is None:
            self.status.config(text="Engine has no moves.")
            messagebox.showinfo("Game Over", "Engine has no moves.")
            return
        self.state = self.state.apply_move(move)
        self.draw_board()
        winner = self.state.is_terminal()
        if winner:
            self.status.config(text=f"Winner: {winner}")
            messagebox.showinfo("Game Over", f"Winner: {winner}")
            return
        self.status.config(
            text=f"Your move | Eval: {score:.3f} | Nodes: {nodes} | {elapsed:.2f}s"
        )


def _engine_worker(
    state: GameState,
    depth: int,
    generation: int,
    result_queue: mp.Queue,
) -> None:
    start_time = time.perf_counter()
    move, score, nodes = iterative_deepening_best_move_with_score(
        state, max_depth=depth
    )
    elapsed = time.perf_counter() - start_time
    result_queue.put((generation, move, score, nodes, elapsed))


def main() -> None:
    root = tk.Tk()
    root.title("Gravity Chess")
    app = GravityChessApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

