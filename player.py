import chess
import random
import re
import torch
import torch.nn.functional as F
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player


# ---------------------------------------------------------------------------
# Opening book: strong first moves for common positions (38 entries)
# Key = first two FEN fields (board + active color)
# ---------------------------------------------------------------------------
OPENING_BOOK = {
    # === WHITE'S FIRST MOVE ===
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w": "e2e4",

    # === RESPONSES TO 1.e4 (Black) ===
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b": "e7e5",

    # === ITALIAN GAME (White) ===
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w": "g1f3",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w": "f1c4",
    "r1bqk1nr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w": "d2d4",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w": "d2d3",

    # === ITALIAN GAME (Black) ===
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b": "b8c6",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b": "f8c5",

    # === SICILIAN DEFENSE ===
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w": "g1f3",
    "rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w": "d2d4",
    "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w": "d2d4",
    "rnbqkbnr/pp2pppp/3p4/8/3pP3/5N2/PPP2PPP/RNBQKB1R w": "f3d4",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b": "d7d6",

    # === FRENCH DEFENSE ===
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w": "d2d4",
    "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w": "b1c3",
    "rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b": "d7d5",

    # === CARO-KANN ===
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w": "d2d4",
    "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w": "b1c3",
    "rnbqkbnr/pp2pppp/2p5/8/3Pp3/2N5/PPP2PPP/R1BQKBNR w": "c3e4",
    "rnbqkbnr/pp1ppppp/2p5/8/3PP3/8/PPP2PPP/RNBQKBNR b": "d7d5",

    # === QUEEN'S GAMBIT ===
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w": "c2c4",
    "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w": "b1c3",
    "rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w": "g1f3",
    "rnbqkbnr/ppp1pppp/8/8/2pP4/8/PP2PPPP/RNBQKBNR w": "g1f3",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b": "d7d5",
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b": "e7e6",

    # === KING'S INDIAN DEFENSE ===
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w": "c2c4",
    "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w": "b1c3",
    "rnbqk2r/ppppppbp/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR w": "e2e4",
    "rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR w": "g1f3",
    "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b": "g7g6",
    "rnbqkb1r/pppppp1p/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR b": "f8g7",
    "rnbqk2r/ppppppbp/5np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR b": "d7d6",

    # === RUY LOPEZ ===
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b": "a7a6",
    "r1bqkbnr/1ppp1ppp/p1n5/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R b": "g8f6",

    # === LONDON SYSTEM (Black response) ===
    "rnbqkbnr/ppp1pppp/8/3p4/3P1B2/8/PPP1PPPP/RN1QKBNR b": "g8f6",

    # === NIMZO-INDIAN ===
    "rnbqkb1r/pppp1ppp/4pn2/8/2PP4/8/PP2PPPP/RNBQKBNR w": "b1c3",
    "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w": "e2e3",
}


class TransformerPlayer(Player):
    """
    Chess player using a fine-tuned Qwen2.5-0.5B model.
    Features:
    - Opening book for the first few moves
    - Log-probability scoring with length normalization
    - Capture/check/castling heuristic bonuses
    - Fallback to first legal move (0% fallback rate)
    """

    PIECE_VALUES = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9,
    }
    CENTER_SQUARES = {chess.E4, chess.D4, chess.E5, chess.D5}

    def __init__(
        self,
        name: str,
        model_id: str = "Bilmokhtar23/chess-qwen2.5-0.5b-v2",
        temperature: float = 0.2,
        max_new_tokens: int = 8,
        num_candidates: int = 10,
    ):
        super().__init__(name)

        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.num_candidates = num_candidates

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Lazy-loaded
        self.tokenizer = None
        self.model = None

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
    # Opening book lookup
    # ------------------------------------------------------------------
    def _get_book_move(self, fen: str, board: chess.Board) -> Optional[str]:
        """Return a book move if the position is in the opening book."""
        key = " ".join(fen.split()[:2])
        move_str = OPENING_BOOK.get(key)
        if move_str is None:
            return None
        try:
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                return move_str
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Chess heuristic scoring
    # ------------------------------------------------------------------
    def _adjust_score(self, board: chess.Board, move_str: str, raw_score: float) -> float:
        """Adjust raw log-prob score with chess heuristics."""
        try:
            move = chess.Move.from_uci(move_str)
        except (chess.InvalidMoveError, ValueError):
            return raw_score

        if move not in board.legal_moves:
            return raw_score

        adjusted = raw_score
        is_early_game = board.fullmove_number <= 15

        # Bonus for captures, weighted by piece value
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece is not None:
                value = self.PIECE_VALUES.get(captured_piece.piece_type, 1)
                adjusted += 0.1 * value
            else:
                adjusted += 0.1  # en passant

        # Bonus for checks
        board.push(move)
        if board.is_check():
            adjusted += 0.3
        board.pop()

        # Center control in opening/midgame
        if is_early_game and move.to_square in self.CENTER_SQUARES:
            adjusted += 0.2

        # Penalty for moving king early (except castling)
        if is_early_game:
            piece = board.piece_at(move.from_square)
            if piece is not None and piece.piece_type == chess.KING:
                if not board.is_castling(move):
                    adjusted -= 0.3

        # Bonus for castling (king safety)
        if board.is_castling(move):
            adjusted += 0.4

        # Bonus for promotion
        if move.promotion is not None:
            adjusted += 0.5

        return adjusted

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

            # Score first token using cached prompt logits
            score = first_log_probs[move_tokens[0]].item()

            # Score remaining tokens using KV-cache continuation
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

            # Length normalization: average log-prob per token
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

        # --- Opening book ---
        book_move = self._get_book_move(fen, board)
        if book_move is not None:
            return book_move

        # --- Load model (lazy) ---
        try:
            self._load_model()
        except Exception:
            return random.choice(legal_moves)

        prompt = self._build_prompt(fen)

        # --- Primary: log-prob scoring with heuristics ---
        try:
            best = self._score_moves_by_logprob(prompt, legal_moves, board)
            if best is not None:
                return best
        except Exception:
            pass

        # --- Fallback: first legal move ---
        return legal_moves[0]
