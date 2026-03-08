import chess
import random
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player


# ---------------------------------------------------------------------------
# Opening book: strong moves for common positions (38 entries)
# Used as a heuristic bonus (+3.0), NOT as a bypass — LLM always runs
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
}

# Center distance table for king centralization in endgames
_KING_CENTER_DISTANCE = {}
for sq in range(64):
    file_dist = abs(chess.square_file(sq) - 3.5)
    rank_dist = abs(chess.square_rank(sq) - 3.5)
    _KING_CENTER_DISTANCE[sq] = file_dist + rank_dist


class TransformerPlayer(Player):
    """
    Chess player using a fine-tuned Qwen2.5-0.5B model.
    The LLM scores all legal moves by log-probability. Rule-based heuristics
    complement the LLM scores to improve move selection.
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
    ):
        super().__init__(name)
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        # Track position history for repetition detection
        self.position_history: List[str] = []

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
        return total <= 26  # roughly when queens are off + some pieces traded

    # ------------------------------------------------------------------
    # Chess heuristic scoring (complements the LLM)
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
        is_endgame = self._is_endgame(board)
        our_color = board.turn

        # --- Checkmate detection (+20.0) — never miss mate-in-1 ---
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return raw_score + 20.0
        gives_stalemate = board.is_stalemate()
        gives_check = board.is_check()
        # Check for repetition after this move
        resulting_fen = " ".join(board.fen().split()[:4])
        repeat_count = self.position_history.count(resulting_fen)

        # --- Mate-in-2 detection (+15.0) — find forced mates ---
        mate_in_2 = False
        if gives_check:
            opponent_moves = list(board.legal_moves)
            if opponent_moves:
                all_lead_to_mate = True
                for opp_move in opponent_moves:
                    board.push(opp_move)
                    found_mate = False
                    for our_reply in board.legal_moves:
                        board.push(our_reply)
                        if board.is_checkmate():
                            found_mate = True
                            board.pop()
                            break
                        board.pop()
                    board.pop()
                    if not found_mate:
                        all_lead_to_mate = False
                        break
                mate_in_2 = all_lead_to_mate

        board.pop()

        if mate_in_2:
            return raw_score + 15.0

        # --- Stalemate avoidance (-10.0) — never stalemate opponent ---
        if gives_stalemate:
            adjusted -= 10.0

        # --- Repetition penalty — avoid draw by repetition ---
        if repeat_count >= 1:
            adjusted -= 0.5 * repeat_count

        # --- Opening book bonus — nudge toward known good openings ---
        book_key = " ".join(board.fen().split()[:2])
        book_move = OPENING_BOOK.get(book_key)
        if book_move == move_str:
            adjusted += 0.5

        # --- Promotion bonus — always promote pawns ---
        if move.promotion is not None:
            adjusted += 1.0

        # --- Smart capture bonus — take free/winning captures ---
        captured_piece = board.piece_at(move.to_square)
        moving_piece = board.piece_at(move.from_square)
        if captured_piece is not None and moving_piece is not None:
            cap_val = self.PIECE_VALUES.get(captured_piece.piece_type, 0)
            mov_val = self.PIECE_VALUES.get(moving_piece.piece_type, 0)
            # Check if the destination square is attacked by opponent after capture
            board.push(move)
            is_recapturable = board.is_attacked_by(board.turn, move.to_square)
            board.pop()
            if not is_recapturable:
                # Free capture — bonus proportional to captured value
                adjusted += 0.15 * cap_val
            elif cap_val > mov_val:
                # Winning trade — bonus for net gain
                adjusted += 0.15 * (cap_val - mov_val)
            elif cap_val == mov_val:
                # Equal trade — small bonus when ahead in material
                our_mat = self._count_material(board, our_color)
                opp_mat = self._count_material(board, not our_color)
                if our_mat >= opp_mat + 3:
                    adjusted += 0.2

        # --- Hanging piece avoidance — penalize moving to attacked squares ---
        if moving_piece is not None and captured_piece is None:
            mov_val = self.PIECE_VALUES.get(moving_piece.piece_type, 0)
            if mov_val >= 3:  # knights, bishops, rooks, queens
                board.push(move)
                is_attacked = board.is_attacked_by(board.turn, move.to_square)
                is_defended = board.is_attacked_by(not board.turn, move.to_square)
                board.pop()
                if is_attacked and not is_defended:
                    adjusted -= 0.1 * mov_val

        # --- Endgame: push pawns toward promotion ---
        if is_endgame:
            if moving_piece is not None and moving_piece.piece_type == chess.PAWN:
                if our_color == chess.WHITE:
                    rank = chess.square_rank(move.to_square)
                    adjusted += 0.05 * rank
                else:
                    rank = 7 - chess.square_rank(move.to_square)
                    adjusted += 0.05 * rank
            # King centralization in endgame
            if moving_piece is not None and moving_piece.piece_type == chess.KING:
                from_dist = _KING_CENTER_DISTANCE[move.from_square]
                to_dist = _KING_CENTER_DISTANCE[move.to_square]
                if to_dist < from_dist:
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

            # Weak length normalization (alpha=0.3) to reduce tokenization bias
            score /= len(move_tokens) ** 0.3

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
