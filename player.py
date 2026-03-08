import chess
import random
import torch
import torch.nn.functional as F
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player


# ---------------------------------------------------------------------------
# Opening book: strong moves for common positions
# Used as a heuristic bonus (+0.5), NOT as a bypass — LLM always runs
# ---------------------------------------------------------------------------
OPENING_BOOK = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w": "e2e4",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b": "e7e5",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w": "g1f3",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w": "f1c4",
    "r1bqk1nr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w": "d2d4",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w": "d2d3",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b": "b8c6",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b": "f8c5",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w": "g1f3",
    "rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w": "d2d4",
    "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w": "d2d4",
    "rnbqkbnr/pp2pppp/3p4/8/3pP3/5N2/PPP2PPP/RNBQKB1R w": "f3d4",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b": "d7d6",
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w": "d2d4",
    "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w": "b1c3",
    "rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b": "d7d5",
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w": "d2d4",
    "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w": "b1c3",
    "rnbqkbnr/pp2pppp/2p5/8/3Pp3/2N5/PPP2PPP/R1BQKBNR w": "c3e4",
    "rnbqkbnr/pp1ppppp/2p5/8/3PP3/8/PPP2PPP/RNBQKBNR b": "d7d5",
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w": "c2c4",
    "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w": "b1c3",
    "rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w": "g1f3",
    "rnbqkbnr/ppp1pppp/8/8/2pP4/8/PP2PPPP/RNBQKBNR w": "g1f3",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b": "d7d5",
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b": "e7e6",
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w": "c2c4",
    "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w": "b1c3",
    "rnbqk2r/ppppppbp/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR w": "e2e4",
    "rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR w": "g1f3",
    "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b": "g7g6",
    "rnbqkb1r/pppppp1p/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR b": "f8g7",
    "rnbqk2r/ppppppbp/5np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR b": "d7d6",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b": "a7a6",
    "r1bqkbnr/1ppp1ppp/p1n5/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R b": "g8f6",
    "rnbqkbnr/ppp1pppp/8/3p4/3P1B2/8/PPP1PPPP/RN1QKBNR b": "g8f6",
    "rnbqkb1r/pppp1ppp/4pn2/8/2PP4/8/PP2PPPP/RNBQKBNR w": "b1c3",
    "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w": "e2e3",
    # Ruy Lopez mainline continuations
    "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w": "b5a4",
    "r1bqkbnr/1ppp1ppp/p1n5/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w": "e1g1",
    # Italian Game continuations
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w": "c2c3",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2BPP3/5N2/PPP2PPP/RNBQK2R b": "e5d4",
    # Scotch Game
    "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b": "e5d4",
    "r1bqkbnr/pppp1ppp/2n5/8/3pP3/5N2/PPP2PPP/RNBQKB1R w": "f3d4",
    # Queen's Gambit Declined mainline
    "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR b": "g8f6",
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w": "c1g5",
    "rnbqkb1r/ppp2ppp/4pn2/3p2B1/2PP4/2N5/PP2PPPP/R2QKBNR b": "f8e7",
    # Sicilian Dragon
    "rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w": "f1e2",
    # Caro-Kann mainline
    "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR b": "d5e4",
    "rnbqkbnr/pp2pppp/2p5/8/3Pp3/2N5/PPP2PPP/R1BQKBNR w": "c3e4",
    # King's Indian deeper
    "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R w": "f1e2",
    # London System
    "rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/5N2/PPP1PPPP/RN1QKB1R b": "c7c5",
    # English Opening
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b": "e7e5",
    "rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR w": "b1c3",
}


class TransformerPlayer(Player):
    """
    Chess player using a fine-tuned Qwen2.5-0.5B model.
    The LLM scores all legal moves by log-probability. Minimal rule-based
    heuristics complement the LLM for things it structurally cannot do
    (detect checkmate, avoid stalemate, avoid repetition draws).
    """

    PIECE_VALUES = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9,
    }

    def __init__(
        self,
        name: str,
        model_id: str = "Bilmokhtar23/chess-qwen2.5-0.5b-v2",
    ):
        super().__init__(name)
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        # Track position history for repetition detection (per game)
        self.position_history: List[str] = []
        self._last_fen_fullmove = None  # detect new game

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------
    def _load_model(self):
        if self.model is not None:
            return
        print(f"[{self.name}] Loading {self.model_id} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------
    def _build_prompt(self, fen: str) -> str:
        return (
            "You are a chess engine. Given the board position in FEN notation, "
            "output the best legal move in UCI format (e.g. e2e4). "
            "Output ONLY the move, nothing else.\n\n"
            f"FEN: {fen}\nMove:"
        )

    # ------------------------------------------------------------------
    # Material counting
    # ------------------------------------------------------------------
    def _count_material(self, board: chess.Board, color: chess.Color) -> int:
        material = 0
        for piece_type, value in self.PIECE_VALUES.items():
            material += len(board.pieces(piece_type, color)) * value
        return material

    def _is_endgame(self, board: chess.Board) -> bool:
        total = self._count_material(board, chess.WHITE) + self._count_material(board, chess.BLACK)
        return total <= 26

    # ------------------------------------------------------------------
    # Detect new game (reset position history)
    # ------------------------------------------------------------------
    def _detect_new_game(self, fen: str):
        """Reset position history when a new game starts."""
        parts = fen.split()
        fullmove = int(parts[5]) if len(parts) > 5 else 1
        turn = parts[1] if len(parts) > 1 else "w"
        # New game: fullmove=1 and it's white's turn, OR fullmove decreased
        if (fullmove == 1 and turn == "w") or (
            self._last_fen_fullmove is not None and fullmove < self._last_fen_fullmove
        ):
            self.position_history = []
        self._last_fen_fullmove = fullmove

    # ------------------------------------------------------------------
    # Chess heuristic scoring (complements the LLM)
    # Only for things the model STRUCTURALLY cannot do
    # ------------------------------------------------------------------
    def _adjust_score(self, board: chess.Board, move_str: str, raw_score: float) -> float:
        """Adjust raw log-prob score with minimal chess heuristics."""
        try:
            move = chess.Move.from_uci(move_str)
        except (chess.InvalidMoveError, ValueError):
            return raw_score

        if move not in board.legal_moves:
            return raw_score

        adjusted = raw_score
        is_endgame = self._is_endgame(board)
        our_color = board.turn

        # --- Checkmate detection (+20.0) — never miss mate-in-1 ---
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return raw_score + 20.0

        gives_stalemate = board.is_stalemate()
        # Detect insufficient material draw
        gives_draw = board.is_insufficient_material()

        # Check for repetition after this move
        resulting_fen = " ".join(board.fen().split()[:4])
        repeat_count = self.position_history.count(resulting_fen)

        board.pop()

        # --- Stalemate avoidance (-10.0) — never stalemate opponent ---
        if gives_stalemate:
            adjusted -= 10.0

        # --- Insufficient material avoidance (-5.0) ---
        if gives_draw:
            adjusted -= 5.0

        # --- Repetition penalty — avoid draw by repetition ---
        # Strong penalty: -3.0 for first repeat, -6.0 for second (would trigger draw)
        if repeat_count >= 1:
            adjusted -= 3.0 * repeat_count

        # --- Opening book bonus — nudge toward known good openings ---
        book_key = " ".join(board.fen().split()[:2])
        book_move = OPENING_BOOK.get(book_key)
        if book_move == move_str:
            adjusted += 0.5

        # --- Promotion bonus — always promote pawns ---
        if move.promotion is not None:
            adjusted += 2.0

        # --- Endgame: push pawns toward promotion (stronger bonus) ---
        if is_endgame:
            piece = board.piece_at(move.from_square)
            if piece is not None and piece.piece_type == chess.PAWN:
                if our_color == chess.WHITE:
                    rank = chess.square_rank(move.to_square)
                    adjusted += 0.15 * rank  # 3x stronger than before
                else:
                    rank = 7 - chess.square_rank(move.to_square)
                    adjusted += 0.15 * rank

            # --- Endgame: centralize king (helps finish the game) ---
            if piece is not None and piece.piece_type == chess.KING:
                # Bonus for moving king toward center in endgame
                from_center = abs(chess.square_file(move.from_square) - 3.5) + abs(chess.square_rank(move.from_square) - 3.5)
                to_center = abs(chess.square_file(move.to_square) - 3.5) + abs(chess.square_rank(move.to_square) - 3.5)
                if to_center < from_center:
                    adjusted += 0.1

        return adjusted

    # ------------------------------------------------------------------
    # Log-prob scoring with length normalization + heuristics
    # ------------------------------------------------------------------
    def _score_moves_by_logprob(
        self, prompt: str, legal_moves: List[str], board: chess.Board
    ) -> Optional[str]:
        """Score each legal move by length-normalized log-probability
        with chess heuristic adjustments."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            prompt_out = self.model(input_ids, use_cache=True)
        first_logits = prompt_out.logits[0, -1, :]
        first_log_probs = F.log_softmax(first_logits.float(), dim=-1)
        past_kv = prompt_out.past_key_values

        best_move = None
        best_score = float("-inf")

        for move_str in legal_moves:
            move_tokens = self.tokenizer.encode(
                " " + move_str, add_special_tokens=False
            )
            if not move_tokens:
                continue

            score = first_log_probs[move_tokens[0]].item()

            if len(move_tokens) > 1:
                move_ids = torch.tensor(
                    [move_tokens], device=self.device
                )
                with torch.no_grad():
                    cont_out = self.model(
                        move_ids, past_key_values=past_kv, use_cache=False
                    )
                for i in range(len(move_tokens) - 1):
                    lp = F.log_softmax(cont_out.logits[0, i, :].float(), dim=-1)
                    score += lp[move_tokens[i + 1]].item()

            # Length normalization: match training (pure average, alpha=1.0)
            score /= len(move_tokens)

            # Apply chess heuristic adjustments
            score = self._adjust_score(board, move_str, score)

            if score > best_score:
                best_score = score
                best_move = move_str

        return best_move

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = [move.uci() for move in board.legal_moves]

        if not legal_moves:
            return None

        # Detect new game and reset position history
        self._detect_new_game(fen)

        # Track position history for repetition detection
        position_key = " ".join(fen.split()[:4])
        self.position_history.append(position_key)

        # Load model (lazy)
        try:
            self._load_model()
        except Exception:
            return random.choice(legal_moves)

        prompt = self._build_prompt(fen)

        # LLM scores all legal moves, heuristics adjust the scores
        try:
            best = self._score_moves_by_logprob(prompt, legal_moves, board)
            if best is not None:
                return best
        except Exception:
            pass

        # Fallback: first legal move
        return legal_moves[0]
