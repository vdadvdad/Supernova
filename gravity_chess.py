"""Simple minimax-based gravity chess engine.

Rules explanation is loaded from gravity.txt (same directory).
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Process, Queue
import os
import random
import time
from typing import Dict, Iterable, List, Optional, Tuple

BOARD_SIZE = 8
WHITE = "w"
BLACK = "b"
PIECE_TYPES = {"K", "Q", "R", "B", "N", "P"}
PIECE_VALUES = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}
TT_EXACT = "exact"
TT_LOWER = "lower"
TT_UPPER = "upper"

_ZOBRIST_RNG = random.Random(0)
_ZOBRIST_PIECES = [p for p in PIECE_TYPES] + [p.lower() for p in PIECE_TYPES]
_ZOBRIST_TABLE: Dict[str, List[int]] = {
    piece: [_ZOBRIST_RNG.getrandbits(64) for _ in range(BOARD_SIZE * BOARD_SIZE)]
    for piece in _ZOBRIST_PIECES
}
_ZOBRIST_SIDE = _ZOBRIST_RNG.getrandbits(64)


@dataclass(frozen=True)
class Move:
    start: Tuple[int, int]
    end: Tuple[int, int]
    promotion: Optional[str] = None

    def __str__(self) -> str:
        def to_alg(pos: Tuple[int, int]) -> str:
            r, c = pos
            return f"{chr(ord('a') + c)}{r + 1}"

        promo = f"={self.promotion}" if self.promotion else ""
        return f"{to_alg(self.start)}{to_alg(self.end)}{promo}"


@dataclass(frozen=True)
class MoveUndo:
    move: Move
    moved_piece: str
    captured_piece: Optional[str]
    hash_before: int
    material_before: int
    side_before: str
    gravity_moves: List[Tuple[int, int, int, int]]


@dataclass
class NodeCounter:
    count: int = 0


class GameState:
    def __init__(
        self,
        board: List[List[Optional[str]]],
        side_to_move: str,
        hash_value: Optional[int] = None,
        material_diff: Optional[int] = None,
    ) -> None:
        self.board = board
        self.side_to_move = side_to_move
        self._hash = hash_value if hash_value is not None else self._compute_hash()
        self._material_diff = (
            material_diff if material_diff is not None else self._compute_material_diff()
        )

    @staticmethod
    def starting() -> "GameState":
        board: List[List[Optional[str]]] = [[None for _ in range(8)] for _ in range(8)]
        board[0] = ["R", "N", "B", "Q", "K", "B", "N", "R"]
        board[1] = ["P"] * 8
        board[6] = ["p"] * 8
        board[7] = ["r", "n", "b", "q", "k", "b", "n", "r"]
        return GameState(board, WHITE)

    def copy(self) -> "GameState":
        return GameState(
            [row[:] for row in self.board],
            self.side_to_move,
            self._hash,
            self._material_diff,
        )

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def piece_at(self, r: int, c: int) -> Optional[str]:
        return self.board[r][c]

    def is_white(self, piece: str) -> bool:
        return piece.isupper()

    def side_of(self, piece: str) -> str:
        return WHITE if self.is_white(piece) else BLACK

    def king_missing(self, side: str) -> bool:
        target = "K" if side == WHITE else "k"
        return all(target not in row for row in self.board)

    def is_terminal(self) -> Optional[str]:
        if self.king_missing(WHITE):
            return BLACK
        if self.king_missing(BLACK):
            return WHITE
        return None

    def generate_moves(self) -> List[Move]:
        moves: List[Move] = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = self.piece_at(r, c)
                if not piece:
                    continue
                if self.side_of(piece) != self.side_to_move:
                    continue
                moves.extend(self._moves_for_piece(r, c, piece))
        return moves

    def _moves_for_piece(self, r: int, c: int, piece: str) -> Iterable[Move]:
        side = self.side_of(piece)
        forward = 1 if side == WHITE else -1
        piece_type = piece.upper()
        if piece_type == "P":
            yield from self._pawn_moves(r, c, side, forward)
            return
        if piece_type == "N":
            for dr, dc in [
                (2, 1), (2, -1), (-2, 1), (-2, -1),
                (1, 2), (1, -2), (-1, 2), (-1, -2),
            ]:
                nr, nc = r + dr, c + dc
                if not self.in_bounds(nr, nc):
                    continue
                target = self.piece_at(nr, nc)
                if not target or self.side_of(target) != side:
                    yield Move((r, c), (nr, nc))
            return
        if piece_type == "K":
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if not self.in_bounds(nr, nc):
                        continue
                    target = self.piece_at(nr, nc)
                    if not target or self.side_of(target) != side:
                        yield Move((r, c), (nr, nc))
            return
        if piece_type in {"B", "R", "Q"}:
            directions = []
            if piece_type in {"B", "Q"}:
                directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
            if piece_type in {"R", "Q"}:
                directions.extend([(1, 0), (-1, 0), (0, 1), (0, -1)])
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                while self.in_bounds(nr, nc):
                    target = self.piece_at(nr, nc)
                    if not target:
                        yield Move((r, c), (nr, nc))
                    else:
                        if self.side_of(target) != side:
                            yield Move((r, c), (nr, nc))
                        break
                    nr += dr
                    nc += dc

    def _pawn_moves(self, r: int, c: int, side: str, forward: int) -> Iterable[Move]:
        start_row = 1 if side == WHITE else 6
        promotion_row = 7 if side == WHITE else 0
        one = r + forward
        if self.in_bounds(one, c) and not self.piece_at(one, c):
            if one == promotion_row:
                yield Move((r, c), (one, c), promotion="Q")
            else:
                yield Move((r, c), (one, c))
            two = r + 2 * forward
            if r == start_row and self.in_bounds(two, c) and not self.piece_at(two, c):
                yield Move((r, c), (two, c))
        for dc in (-1, 1):
            nr, nc = r + forward, c + dc
            if not self.in_bounds(nr, nc):
                continue
            target = self.piece_at(nr, nc)
            if target and self.side_of(target) != side:
                if nr == promotion_row:
                    yield Move((r, c), (nr, nc), promotion="Q")
                else:
                    yield Move((r, c), (nr, nc))

    def apply_move(self, move: Move) -> "GameState":
        new_state = self.copy()
        sr, sc = move.start
        er, ec = move.end
        piece = new_state.board[sr][sc]
        new_state._set_piece(sr, sc, None)
        if move.promotion:
            piece = move.promotion if piece and piece.isupper() else move.promotion.lower()
        new_state._set_piece(er, ec, piece)
        new_state._apply_gravity()
        new_state.side_to_move = WHITE if self.side_to_move == BLACK else BLACK
        new_state._hash ^= _ZOBRIST_SIDE
        return new_state

    def make_move(self, move: Move) -> MoveUndo:
        sr, sc = move.start
        er, ec = move.end
        piece = self.board[sr][sc]
        if piece is None:
            raise ValueError("No piece on start square.")
        captured = self.board[er][ec]
        hash_before = self._hash
        material_before = self._material_diff
        side_before = self.side_to_move
        self._set_piece(sr, sc, None)
        placed = piece
        if move.promotion:
            placed = move.promotion if piece.isupper() else move.promotion.lower()
        self._set_piece(er, ec, placed)
        gravity_moves = self._apply_gravity_with_record()
        self.side_to_move = WHITE if self.side_to_move == BLACK else BLACK
        self._hash ^= _ZOBRIST_SIDE
        return MoveUndo(
            move=move,
            moved_piece=piece,
            captured_piece=captured,
            hash_before=hash_before,
            material_before=material_before,
            side_before=side_before,
            gravity_moves=gravity_moves,
        )

    def undo_move(self, undo: MoveUndo) -> None:
        for sr, sc, er, ec in reversed(undo.gravity_moves):
            self._move_piece_no_hash(er, ec, sr, sc)
        sr, sc = undo.move.start
        er, ec = undo.move.end
        self._set_piece_no_hash(er, ec, undo.captured_piece)
        self._set_piece_no_hash(sr, sc, undo.moved_piece)
        self.side_to_move = undo.side_before
        self._hash = undo.hash_before
        self._material_diff = undo.material_before

    def _apply_gravity(self) -> None:
        changed = True
        while changed:
            changed = False
            changed |= self._apply_gravity_for_color(WHITE)
            changed |= self._apply_gravity_for_color(BLACK)

    def _apply_gravity_with_record(self) -> List[Tuple[int, int, int, int]]:
        changes: List[Tuple[int, int, int, int]] = []
        changed = True
        while changed:
            changed = False
            changed |= self._apply_gravity_for_color(WHITE, changes)
            changed |= self._apply_gravity_for_color(BLACK, changes)
        return changes

    def _apply_gravity_for_color(
        self,
        side: str,
        changes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> bool:
        direction = 1 if side == WHITE else -1
        moved = False
        if side == WHITE:
            row_range = range(BOARD_SIZE - 2, -1, -1)
        else:
            row_range = range(1, BOARD_SIZE)
        for r in row_range:
            for c in range(BOARD_SIZE):
                piece = self.piece_at(r, c)
                if not piece or self.side_of(piece) != side or piece.upper() == "P":
                    continue
                nr = r + direction
                if not self.in_bounds(nr, c):
                    continue
                if self.piece_at(nr, c) is None:
                    self._move_piece(r, c, nr, c)
                    if changes is not None:
                        changes.append((r, c, nr, c))
                    moved = True
        return moved

    def __str__(self) -> str:
        rows = []
        for r in range(BOARD_SIZE - 1, -1, -1):
            row = self.board[r]
            rows.append(" ".join(piece or "." for piece in row))
        return "\n".join(rows)

    def zobrist_hash(self) -> int:
        return self._hash

    def _compute_hash(self) -> int:
        value = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = self.board[r][c]
                if piece:
                    value ^= _ZOBRIST_TABLE[piece][r * BOARD_SIZE + c]
        if self.side_to_move == BLACK:
            value ^= _ZOBRIST_SIDE
        return value

    def _compute_material_diff(self) -> int:
        diff = 0
        for row in self.board:
            for piece in row:
                if not piece:
                    continue
                value = PIECE_VALUES[piece.upper()]
                diff += value if self.is_white(piece) else -value
        return diff

    def _set_piece(self, r: int, c: int, piece: Optional[str]) -> None:
        existing = self.board[r][c]
        if existing:
            self._hash ^= _ZOBRIST_TABLE[existing][r * BOARD_SIZE + c]
            value = PIECE_VALUES[existing.upper()]
            self._material_diff -= value if self.is_white(existing) else -value
        if piece:
            self._hash ^= _ZOBRIST_TABLE[piece][r * BOARD_SIZE + c]
            value = PIECE_VALUES[piece.upper()]
            self._material_diff += value if self.is_white(piece) else -value
        self.board[r][c] = piece

    def _move_piece(self, sr: int, sc: int, er: int, ec: int) -> None:
        piece = self.board[sr][sc]
        self._set_piece(sr, sc, None)
        self._set_piece(er, ec, piece)

    def _set_piece_no_hash(self, r: int, c: int, piece: Optional[str]) -> None:
        self.board[r][c] = piece

    def _move_piece_no_hash(self, sr: int, sc: int, er: int, ec: int) -> None:
        piece = self.board[sr][sc]
        self.board[sr][sc] = None
        self.board[er][ec] = piece


def parse_fen(fen: str) -> GameState:
    parts = fen.strip().split()
    if len(parts) < 2:
        raise ValueError("FEN must include piece placement and side to move.")
    placement, side = parts[0], parts[1]
    if side not in ("w", "b"):
        raise ValueError("FEN side to move must be 'w' or 'b'.")
    rows = placement.split("/")
    if len(rows) != 8:
        raise ValueError("FEN placement must have 8 ranks.")
    board: List[List[Optional[str]]] = [[None for _ in range(8)] for _ in range(8)]
    for fen_rank_index, row in enumerate(rows):
        r = 7 - fen_rank_index
        c = 0
        for ch in row:
            if ch.isdigit():
                c += int(ch)
                continue
            if ch.upper() not in PIECE_TYPES:
                raise ValueError(f"Invalid piece in FEN: {ch}")
            if c >= 8:
                raise ValueError("FEN rank has too many files.")
            board[r][c] = ch
            c += 1
        if c != 8:
            raise ValueError("FEN rank has too few files.")
    return GameState(board, WHITE if side == "w" else BLACK)


def parse_uci_move(text: str) -> Move:
    move = text.strip()
    if len(move) < 4:
        raise ValueError("UCI move must be at least 4 characters.")
    start_file = ord(move[0]) - ord("a")
    start_rank = int(move[1]) - 1
    end_file = ord(move[2]) - ord("a")
    end_rank = int(move[3]) - 1
    promo = None
    if len(move) >= 5:
        promo = move[4].upper()
    return Move((start_rank, start_file), (end_rank, end_file), promo)


def minimax_best_move_with_score(
    state: GameState,
    depth: int,
    tt: Optional[Dict[int, Tuple[int, float, str]]] = None,
    killer_moves: Optional[Dict[int, List[Move]]] = None,
    history: Optional[Dict[Move, int]] = None,
    pv_move: Optional[Move] = None,
) -> Tuple[Optional[Move], float, int]:
    root_player = state.side_to_move
    q_tt: Dict[int, float] = {}
    counter = NodeCounter()
    moves = _order_moves(
        state,
        state.generate_moves(),
        pv_move,
        killer_moves or {},
        history or {},
        depth,
    )
    if not moves:
        return None, 0.0, 0
    best_move = None
    if state.side_to_move == root_player:
        best_score = float("-inf")
        for move in moves:
            undo = state.make_move(move)
            score = minimax(
                state,
                depth - 1,
                root_player,
                float("-inf"),
                float("inf"),
                tt,
                killer_moves,
                history,
                q_tt,
                counter,
            )
            state.undo_move(undo)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move, float(best_score), counter.count
    else:
        best_score = float("inf")
        q = Queue()
        processes = []
        scores = []
        for move in moves:
            p = Process(
                target=_minimax_process,
                args=(
                    state.apply_move(move),
                    depth - 1,
                    root_player,
                    float("-inf"),
                    float("inf"),
                    tt,
                    killer_moves,
                    history,
                    q,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            scores.append(q.get())
        values = [score for score, _ in scores]
        nodes = sum(node_count for _, node_count in scores)
        best_score = min(values)
        best_move = moves[values.index(best_score)]
    return best_move, float(best_score), nodes


def minimax(
    state: GameState,
    depth: int,
    root_player: str,
    alpha: float,
    beta: float,
    tt: Optional[Dict[int, Tuple[int, float, str]]] = None,
    killer_moves: Optional[Dict[int, List[Move]]] = None,
    history: Optional[Dict[Move, int]] = None,
    q_tt: Optional[Dict[int, float]] = None,
    counter: Optional[NodeCounter] = None,
    q: Optional[Queue] = None,
) -> float:
    if counter is not None:
        counter.count += 1
    if tt is not None:
        key = state.zobrist_hash()
        cached = tt.get(key)
        if cached and cached[0] >= depth:
            _, cached_value, cached_flag = cached
            if cached_flag == TT_EXACT:
                if q:
                    q.put(cached_value)
                return cached_value
            if cached_flag == TT_LOWER and cached_value >= beta:
                if q:
                    q.put(cached_value)
                return cached_value
            if cached_flag == TT_UPPER and cached_value <= alpha:
                if q:
                    q.put(cached_value)
                return cached_value
    alpha_orig = alpha
    beta_orig = beta
    winner = state.is_terminal()
    if winner:
        value = 100 - depth if winner == root_player else -100 + depth
        if q:
            q.put(value)
        return value
    if depth <= 0:
        value = quiescence(state, root_player, alpha, beta, q_tt, 0, 6, counter)
        if q:
            q.put(value)
        return value
    moves = _order_moves(
        state,
        state.generate_moves(),
        None,
        killer_moves or {},
        history or {},
        depth,
    )
    if not moves:
        value = 0.0
        if q:
            q.put(value)
        return value

    maximizing = state.side_to_move == root_player
    if maximizing:
        value = float("-inf")
        for move in moves:
            undo = state.make_move(move)
            value = max(
                value,
                minimax(
                    state,
                    depth - 1,
                    root_player,
                    alpha,
                    beta,
                    tt,
                    killer_moves,
                    history,
                    q_tt,
                ),
            )
            state.undo_move(undo)
            alpha = max(alpha, value)
            if alpha >= beta:
                _record_killer_and_history(state, move, depth, killer_moves, history)
                break
        if tt is not None:
            if value <= alpha_orig:
                flag = TT_UPPER
            elif value >= beta_orig:
                flag = TT_LOWER
            else:
                flag = TT_EXACT
            tt[state.zobrist_hash()] = (depth, value, flag)
        if q:
            q.put(value)
        return value
    value = float("inf")
    for move in moves:
        undo = state.make_move(move)
        value = min(
            value,
            minimax(
                state,
                depth - 1,
                root_player,
                alpha,
                beta,
                tt,
                killer_moves,
                history,
                q_tt,
            ),
        )
        state.undo_move(undo)
        beta = min(beta, value)
        if alpha >= beta:
            _record_killer_and_history(state, move, depth, killer_moves, history)
            break
    if tt is not None:
        if value <= alpha_orig:
            flag = TT_UPPER
        elif value >= beta_orig:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
        tt[state.zobrist_hash()] = (depth, value, flag)
    if q:
        q.put(value)
    return value


def quiescence(
    state: GameState,
    root_player: str,
    alpha: float,
    beta: float,
    q_tt: Optional[Dict[int, float]] = None,
    q_depth: int = 0,
    max_q_depth: int = 6,
    counter: Optional[NodeCounter] = None,
) -> float:
    if counter is not None:
        counter.count += 1
    if q_tt is not None:
        key = state.zobrist_hash()
        cached = q_tt.get(key)
        if cached is not None:
            return cached
    winner = state.is_terminal()
    if winner:
        return 100 if winner == root_player else -100
    stand_pat = float(evaluate_material(state, root_player))
    if q_depth >= max_q_depth:
        return stand_pat
    max_gain = PIECE_VALUES["Q"]
    if stand_pat + max_gain < alpha:
        return stand_pat
    if stand_pat >= beta:
        return stand_pat
    if alpha < stand_pat:
        alpha = stand_pat

    moves = [
        move
        for move in state.generate_moves()
        if _is_capture(state, move) and _is_winning_capture(state, move)
    ]
    if not moves:
        return stand_pat
    moves.sort(key=lambda m: _mvv_lva_score(state, m), reverse=True)

    maximizing = state.side_to_move == root_player
    if maximizing:
        value = stand_pat
        for move in moves:
            undo = state.make_move(move)
            value = max(
                value,
                quiescence(
                    state,
                    root_player,
                    alpha,
                    beta,
                    q_tt,
                    q_depth + 1,
                    max_q_depth,
                    counter,
                ),
            )
            state.undo_move(undo)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        if q_tt is not None:
            key = state.zobrist_hash()
            if key not in q_tt:
                q_tt[key] = value
        return value

    value = stand_pat
    for move in moves:
        undo = state.make_move(move)
        value = min(
            value,
            quiescence(
                state,
                root_player,
                alpha,
                beta,
                q_tt,
                q_depth + 1,
                max_q_depth,
                counter,
            ),
        )
        state.undo_move(undo)
        beta = min(beta, value)
        if alpha >= beta:
            break
    if q_tt is not None:
        key = state.zobrist_hash()
        if key not in q_tt:
            q_tt[key] = value
    return value


def _minimax_process(
    state: GameState,
    depth: int,
    root_player: str,
    alpha: float,
    beta: float,
    tt: Optional[Dict[int, Tuple[int, float, str]]] = None,
    killer_moves: Optional[Dict[int, List[Move]]] = None,
    history: Optional[Dict[Move, int]] = None,
    q: Optional[Queue] = None,
) -> None:
    q_tt: Dict[int, float] = {}
    counter = NodeCounter()
    value = minimax(
        state,
        depth,
        root_player,
        alpha,
        beta,
        tt,
        killer_moves,
        history,
        q_tt,
        counter,
    )
    if q is not None:
        q.put((value, counter.count))

def iterative_deepening_best_move_with_score(
    state: GameState, max_depth: int
) -> Tuple[Optional[Move], float, int]:
    best_move: Optional[Move] = None
    best_score: float = 0.0
    best_nodes: int = 0
    tt: Dict[int, Tuple[int, float, str]] = {}
    killer_moves: Dict[int, List[Move]] = {}
    history: Dict[Move, int] = {}
    for depth in range(1, max_depth + 1):
        move, score, nodes = minimax_best_move_with_score(
            state,
            depth,
            tt,
            killer_moves,
            history,
            best_move,
        )
        if move is not None:
            best_move = move
            best_score = score
            best_nodes = nodes
    return best_move, best_score, best_nodes


def _order_moves(
    state: GameState,
    moves: List[Move],
    pv_move: Optional[Move],
    killer_moves: Dict[int, List[Move]],
    history: Dict[Move, int],
    depth: int,
) -> List[Move]:
    if not moves:
        return []
    ordered: List[Move] = []
    remaining = moves[:]
    if pv_move in remaining:
        ordered.append(pv_move)
        remaining.remove(pv_move)

    captures: List[Move] = []
    quiet: List[Move] = []
    for move in remaining:
        if _is_capture(state, move):
            captures.append(move)
        else:
            quiet.append(move)

    captures.sort(key=lambda m: _mvv_lva_score(state, m), reverse=True)
    ordered.extend(captures)

    killers = killer_moves.get(depth, [])
    killer_ordered = [m for m in killers if m in quiet]
    ordered.extend(killer_ordered)
    quiet = [m for m in quiet if m not in killers]

    quiet.sort(key=lambda m: history.get(m, 0), reverse=True)
    ordered.extend(quiet)
    return ordered


def _is_capture(state: GameState, move: Move) -> bool:
    er, ec = move.end
    return state.piece_at(er, ec) is not None


def _mvv_lva_score(state: GameState, move: Move) -> Tuple[int, int]:
    sr, sc = move.start
    er, ec = move.end
    attacker = state.piece_at(sr, sc)
    victim = state.piece_at(er, ec)
    if not attacker or not victim:
        return (0, 0)
    return (PIECE_VALUES[victim.upper()], -PIECE_VALUES[attacker.upper()])


def _is_winning_capture(state: GameState, move: Move) -> bool:
    sr, sc = move.start
    er, ec = move.end
    attacker = state.piece_at(sr, sc)
    victim = state.piece_at(er, ec)
    if not attacker or not victim:
        return False
    if victim.upper() == "K":
        return True
    return PIECE_VALUES[victim.upper()] >= PIECE_VALUES[attacker.upper()]


def _record_killer_and_history(
    state: GameState,
    move: Move,
    depth: int,
    killer_moves: Optional[Dict[int, List[Move]]],
    history: Optional[Dict[Move, int]],
) -> None:
    if killer_moves is None or history is None:
        return
    if _is_capture(state, move):
        return
    history[move] = history.get(move, 0) + depth * depth
    killers = killer_moves.setdefault(depth, [])
    if move in killers:
        return
    killers.insert(0, move)
    if len(killers) > 2:
        killers.pop()


def evaluate_material(state: GameState, root_player: str) -> int:
    return state._material_diff if root_player == WHITE else -state._material_diff


def load_rules_text() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    rules_path = os.path.join(here, "gravity.txt")
    try:
        with open(rules_path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return "Gravity rules file not found."


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Gravity chess minimax engine")
    parser.add_argument("--rules", action="store_true", help="Print gravity rules")
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="Minimax depth limit (plies) for engine calculation",
    )
    parser.add_argument(
        "--fen",
        type=str,
        default=None,
        help="FEN position to analyze (reads from stdin if omitted)",
    )
    args = parser.parse_args()

    if args.rules:
        print(load_rules_text())
        return

    if args.fen is not None:
        fen = args.fen.strip()
    else:
        if sys.stdin.isatty():
            print("Enter FEN:")
        fen = sys.stdin.readline().strip()
    if not fen:
        raise SystemExit("No FEN provided.")
    state = parse_fen(fen)

    if sys.stdin.isatty():
        print("Engine analyzing position...")
    while True:
        start_time = time.perf_counter()
        move, score, nodes = iterative_deepening_best_move_with_score(
            state, max_depth=args.depth
        )
        elapsed = time.perf_counter() - start_time
        if move:
            print(str(move))
            state = state.apply_move(move)
            print(state)
            print(
                f"Evaluation: {score:.3f} (nodes: {nodes}, time: {elapsed:.3f}s)"
            )
        else:
            print("0000")
            return

        if sys.stdin.isatty():
            print("Enter UCI move (blank to quit):")
        line = sys.stdin.readline()
        if not line:
            break
        move_text = line.strip()
        if not move_text:
            break
        if move_text != "0000":
            state = state.apply_move(parse_uci_move(move_text))


if __name__ == "__main__":
    main()

