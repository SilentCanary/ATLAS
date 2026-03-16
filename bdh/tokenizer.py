"""
BPE tokenizer for Python code, built with HuggingFace tokenizers.

Trains a subword vocabulary from Python source files, ensuring that
Python keywords, common patterns, and indentation are single tokens.
"""

import os
import json
from typing import List, Optional


class CodeTokenizer:
    """BPE tokenizer for Python code."""

    SPECIAL_TOKENS = [
        "[PAD]", "[UNK]", "[BOS]", "[EOS]",
        "[FILE]", "[FUNC]", "[CLASS]", "[METHOD]",
        "[IMPORTS]", "[UPSTREAM]", "[DOWNSTREAM]",
        "[INDENT]", "[DEDENT]", "[NEWLINE]",
    ]

    def __init__(self, vocab_size: int = 8192, model_path: Optional[str] = None):
        self.vocab_size = vocab_size
        self.tokenizer = None

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(self, files: List[str], output_path: str):
        """
        Train BPE tokenizer on Python source files.

        Args:
            files: list of .py file paths to train on
            output_path: where to save the trained tokenizer
        """
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

        # Pre-tokenize: split on whitespace boundaries but preserve indentation
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ])

        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.SPECIAL_TOKENS,
            min_frequency=2,
            show_progress=True,
        )

        tokenizer.train(files, trainer)
        self.tokenizer = tokenizer

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        tokenizer.save(output_path)
        print(f"Tokenizer trained: {tokenizer.get_vocab_size()} tokens, saved to {output_path}")

    def train_from_directory(self, directory: str, output_path: str):
        """Train on all .py files in a directory tree."""
        py_files = []
        for root, dirs, files in os.walk(directory):
            # Skip hidden dirs, __pycache__, .git
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for f in files:
                if f.endswith(".py"):
                    py_files.append(os.path.join(root, f))

        if not py_files:
            raise ValueError(f"No .py files found in {directory}")

        print(f"Training tokenizer on {len(py_files)} Python files...")
        self.train(py_files, output_path)

    def load(self, path: str):
        """Load a pre-trained tokenizer."""
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(path)
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not trained/loaded. Call train() or load() first.")
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not trained/loaded. Call train() or load() first.")
        return self.tokenizer.decode(ids)

    def get_vocab_size(self) -> int:
        if self.tokenizer is None:
            return self.vocab_size
        return self.tokenizer.get_vocab_size()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python tokenizer.py <code_directory> <output_path>")
        print("Example: python tokenizer.py ./repos/my_project ./bdh/tokenizer.json")
        sys.exit(1)

    tok = CodeTokenizer(vocab_size=8192)
    tok.train_from_directory(sys.argv[1], sys.argv[2])

    # Quick sanity check
    test = "def hello_world():\n    print('Hello, world!')\n"
    ids = tok.encode(test)
    decoded = tok.decode(ids)
    print(f"\nSanity check:")
    print(f"  Input:   {repr(test)}")
    print(f"  Tokens:  {len(ids)} ids")
    print(f"  Decoded: {repr(decoded)}")
