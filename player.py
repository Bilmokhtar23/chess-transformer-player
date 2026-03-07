import chess
import random
import re
import torch
import torch.nn.functional as F
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player


class TransformerPlayer(Player):
    """
    Chess player using a fine-tuned SmolLM2-360M model.
    Primary inference: scores ALL legal moves by log-probability and picks the best.
    Fallback: multi-candidate generation, then greedy, then first legal move.
    """

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(
        self,
        name: str,
        model_id: str = "Bilmokhtar23/smollm2-360m-chess",
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
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
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
    # Legal move probability scoring (primary inference)
    # ------------------------------------------------------------------
    def _score_moves_by_logprob(
        self, prompt: str, legal_moves: List[str]
    ) -> Optional[str]:
        """Score each legal move by log-probability of the model generating it.

        Uses KV-cache: one forward pass for the prompt, then for each move
        only the move tokens are fed through. This makes scoring ~30 moves
        roughly as fast as generating one sequence.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Single forward pass for the prompt -- cache the key/value states
        with torch.no_grad():
            prompt_out = self.model(input_ids, use_cache=True)
        # Logits at the last prompt token predict the first generated token
        first_logits = prompt_out.logits[0, -1, :]
        first_log_probs = F.log_softmax(first_logits.float(), dim=-1)
        past_kv = prompt_out.past_key_values

        best_move = None
        best_score = float("-inf")

        for move_str in legal_moves:
            # Tokenize with leading space (natural continuation after "Move:")
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
                # cont_out.logits shape: (1, len(move_tokens), vocab_size)
                # Position i in cont_out predicts token at position i+1
                for i in range(len(move_tokens) - 1):
                    lp = F.log_softmax(cont_out.logits[0, i, :].float(), dim=-1)
                    score += lp[move_tokens[i + 1]].item()

            if score > best_score:
                best_score = score
                best_move = move_str

        return best_move

    # ------------------------------------------------------------------
    # Move extraction and validation (for generation fallback)
    # ------------------------------------------------------------------
    def _extract_move(self, text: str) -> Optional[str]:
        match = self.UCI_REGEX.search(text)
        return match.group(1).lower() if match else None

    def _is_legal(self, board: chess.Board, move_str: str) -> bool:
        try:
            mv = chess.Move.from_uci(move_str)
            return mv in board.legal_moves
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Generation fallback
    # ------------------------------------------------------------------
    def _generate_move(self, prompt: str, board: chess.Board) -> Optional[str]:
        """Fall back to multi-candidate generation if log-prob scoring fails."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                num_return_sequences=self.num_candidates,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        for seq in outputs:
            decoded = self.tokenizer.decode(seq, skip_special_tokens=True)
            if decoded.startswith(prompt):
                decoded = decoded[len(prompt):]
            move = self._extract_move(decoded)
            if move and self._is_legal(board, move):
                return move

        # Greedy as last generation attempt
        with torch.no_grad():
            greedy = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        decoded = self.tokenizer.decode(greedy[0], skip_special_tokens=True)
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):]
        move = self._extract_move(decoded)
        if move and self._is_legal(board, move):
            return move

        return None

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def get_move(self, fen: str) -> Optional[str]:
        try:
            self._load_model()
        except Exception:
            board = chess.Board(fen)
            moves = list(board.legal_moves)
            return random.choice(moves).uci() if moves else None

        board = chess.Board(fen)
        legal_moves = [move.uci() for move in board.legal_moves]

        if not legal_moves:
            return None

        prompt = self._build_prompt(fen)

        # --- Primary: log-probability scoring over all legal moves ---
        try:
            best = self._score_moves_by_logprob(prompt, legal_moves)
            if best is not None:
                return best
        except Exception:
            pass

        # --- Fallback 1: generation-based move selection ---
        try:
            move = self._generate_move(prompt, board)
            if move is not None:
                return move
        except Exception:
            pass

        # --- Fallback 2: first legal move (deterministic, never None) ---
        return legal_moves[0]
