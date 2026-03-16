"""
Training script for Code-BDH.

Usage:
    python -m bdh.train --code-dir ./repos/my_project --epochs 5

Or from ATLAS pipeline:
    from bdh.train import train_code_bdh
    model = train_code_bdh(code_dir="./repos/project", parsed_repo=parsed, graph=graph)
"""

import os
import time
import argparse
import json
from dataclasses import asdict

import torch

from bdh.bdh import BDH, CodeBDHConfig
from bdh.tokenizer import CodeTokenizer
from bdh.data_pipeline import CodeDataPipeline


CHECKPOINT_DIR = "bdh/checkpoints"
TOKENIZER_PATH = "bdh/tokenizer.json"
TRAIN_DATA_PATH = "bdh/train_data.bin"


def train_code_bdh(
    code_dir: str,
    parsed_repo: dict = None,
    graph=None,
    config: CodeBDHConfig = None,
    max_iters: int = 3000,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    eval_interval: int = 100,
    checkpoint_interval: int = 500,
    device: str = None,
):
    """
    Train a Code-BDH model.

    Args:
        code_dir: directory with Python source files for training
        parsed_repo: optional ATLAS parsed repo for structured data
        graph: optional NetworkX graph for annotated data
        config: model configuration (defaults to CodeBDHConfig)
        max_iters: training iterations
        batch_size: batch size
        learning_rate: AdamW learning rate
        eval_interval: how often to evaluate
        checkpoint_interval: how often to save checkpoints
        device: "cuda" or "cpu" (auto-detected if None)

    Returns:
        trained BDH model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Tokenizer ---
    tokenizer = CodeTokenizer(vocab_size=8192)
    if os.path.exists(TOKENIZER_PATH):
        print(f"Loading existing tokenizer from {TOKENIZER_PATH}")
        tokenizer.load(TOKENIZER_PATH)
    else:
        print("Training tokenizer...")
        tokenizer.train_from_directory(code_dir, TOKENIZER_PATH)

    # --- Data ---
    pipeline = CodeDataPipeline(tokenizer, block_size=config.block_size if config else 1024)

    if os.path.exists(TRAIN_DATA_PATH):
        print(f"Loading cached training data from {TRAIN_DATA_PATH}")
        data = pipeline.load_dataset(TRAIN_DATA_PATH)
    else:
        print("Preparing training data...")
        pipeline.prepare_dataset(
            code_dir=code_dir,
            parsed_repo=parsed_repo,
            graph=graph,
            output_path=TRAIN_DATA_PATH,
        )
        data = pipeline.load_dataset(TRAIN_DATA_PATH)

    # Split 90/10
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    # --- Model ---
    if config is None:
        config = CodeBDHConfig(vocab_size=tokenizer.get_vocab_size())
    else:
        config.vocab_size = tokenizer.get_vocab_size()

    model = BDH(config).to(device)
    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,} ({param_count / 1e6:.1f}M)")
    print(f"Neuron count: {model.get_neuron_count():,}")

    # Compile if available and on CUDA (skip on CPU to avoid compiler requirements)
    compiled = False
    if device == "cuda":
        try:
            model = torch.compile(model)
            compiled = True
            print("Model compiled with torch.compile")
        except Exception:
            pass
    if not compiled:
        print("Using eager mode (no torch.compile)")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

    # --- Training ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    best_val_loss = float("inf")

    print(f"\nStarting training: {max_iters} iterations, batch_size={batch_size}")
    print("-" * 60)

    t0 = time.time()
    for iteration in range(max_iters):
        model.train()

        X, Y = pipeline.get_batch(train_data, batch_size, device)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits, loss = model(X, Y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, loss = model(X, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        # --- Logging ---
        if iteration % 10 == 0:
            dt = time.time() - t0
            tok_per_sec = (iteration + 1) * batch_size * config.block_size / dt if dt > 0 else 0
            print(f"iter {iteration:5d} | loss {loss.item():.4f} | {tok_per_sec:.0f} tok/s")

        # --- Eval ---
        if iteration > 0 and iteration % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                eval_losses = []
                for _ in range(10):
                    Xv, Yv = pipeline.get_batch(val_data, batch_size, device)
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            _, val_loss = model(Xv, Yv)
                    else:
                        _, val_loss = model(Xv, Yv)
                    eval_losses.append(val_loss.item())

                avg_val_loss = sum(eval_losses) / len(eval_losses)
                print(f"  VAL loss: {avg_val_loss:.4f} {'*BEST*' if avg_val_loss < best_val_loss else ''}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    _save_checkpoint(model, optimizer, config, iteration, avg_val_loss,
                                     os.path.join(CHECKPOINT_DIR, "best.pt"))

        # --- Checkpoint ---
        if iteration > 0 and iteration % checkpoint_interval == 0:
            _save_checkpoint(model, optimizer, config, iteration, loss.item(),
                             os.path.join(CHECKPOINT_DIR, f"iter_{iteration}.pt"))

    # Final save
    _save_checkpoint(model, optimizer, config, max_iters, loss.item(),
                     os.path.join(CHECKPOINT_DIR, "final.pt"))

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.0f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return model


def _save_checkpoint(model, optimizer, config, iteration, loss, path):
    """Save model checkpoint."""
    # Unwrap compiled model if needed
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "iteration": iteration,
        "loss": loss,
    }, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path: str, device: str = "cpu"):
    """Load a model from checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    config = CodeBDHConfig(**checkpoint["config"])
    model = BDH(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config


# --- CLI entry point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Code-BDH model")
    parser.add_argument("--code-dir", required=True, help="Directory with Python source files")
    parser.add_argument("--parsed-repo", default=None, help="Path to ATLAS parsed_repo.json")
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    parsed = None
    if args.parsed_repo and os.path.exists(args.parsed_repo):
        with open(args.parsed_repo, "r") as f:
            parsed = json.load(f)

    train_code_bdh(
        code_dir=args.code_dir,
        parsed_repo=parsed,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )
