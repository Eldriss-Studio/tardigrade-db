"""Q-B with logprob metric: does injection elevate the probability of the
planted answer at the answer position?

The Q-B v1 (HORIZON password) and Q-B v2 (Hall Brennan room) runs both
showed substring-match=0.00 across all N. But qualitative inspection of
the generated text showed inject and control producing *different*
outputs at N up to 4096, proving injection has measurable effect. The
substring metric was the wrong instrument.

This script replaces substring match with **next-token logprob** at the
answer position:

  - Inject condition: forward (fact-KV cache + N tokens of filler +
    completion prompt that ends right before the planted answer) →
    extract softmax(logits) for the next token → read the probability of
    the planted-answer's first token.
  - Control condition: same prompt without the fact-KV in cache. Same
    answer token. Read its probability.

Effect size = log(P_inject / P_control). Positive means injection
boosted the planted answer. Magnitude tells us how much.

Sweep N to characterise injection survival as a function of how far
the active context has advanced past the injected fact.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent
UMBRELLA = HERE.parent.parent.parent
TARDIGRADE_PY = UMBRELLA / "tardigrade-db" / "python"
if TARDIGRADE_PY.is_dir():
    sys.path.insert(0, str(TARDIGRADE_PY))

from tardigrade_db import Engine
from tardigrade_hooks import CalibrationRegistry, select_query_layer
from tardigrade_hooks.kp_injector import KnowledgePackStore

MODEL_ID = "google/gemma-3-4b-it"
DEVICE = "cuda"
N_TRIALS_PER_CELL = int(os.environ.get("LOGPROB_N", "10"))

_FILLER_SEED = """When in the course of human events, it becomes necessary
for one people to dissolve the political bands which have connected them
with another, and to assume among the powers of the earth, the separate
and equal station to which the laws of nature and of nature's god entitle
them. The afternoon light filtered through the dusty window. A clock
ticked somewhere down the corridor. The chair creaked when she settled
into it. Outside, a single bird called and was answered by another.
"""


def build_filler_tokens(tokenizer, target_n_tokens: int) -> list[int]:
    out: list[int] = []
    while len(out) < target_n_tokens:
        out.extend(tokenizer.encode(_FILLER_SEED, add_special_tokens=False))
    return out[:target_n_tokens]


def load_model_and_engine(engine_dir: Path):
    print(f"[{time.strftime('%H:%M:%S')}] loading {MODEL_ID} 4-bit on {DEVICE}…", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, output_hidden_states=True,
    )
    getattr(model, "eval")()  # PyTorch eval-mode toggle via getattr

    engine = Engine(str(engine_dir))
    registry = CalibrationRegistry()
    print(f"[{time.strftime('%H:%M:%S')}] resolving calibration…", flush=True)
    cal = select_query_layer(model, tok, registry=registry)
    print(f"  → strategy={cal.best_strategy}, layer={cal.best_layer}", flush=True)
    kps = KnowledgePackStore(engine, model, tok, owner=1,
                              calibration_registry=registry)
    return tok, model, engine, kps


def logprob_of_token(model, ext_cache, prompt_ids: torch.Tensor,
                    target_token_id: int) -> float:
    """Forward prompt_ids through model with ext_cache as past, return the
    log-probability of target_token_id as the immediate next-token.

    Returns log(p) where p ∈ (0, 1]. -inf possible (handled).
    """
    with torch.no_grad():
        out = model(prompt_ids, past_key_values=ext_cache, use_cache=False)
    # logits at the LAST position predict the NEXT token.
    logits = out.logits[0, -1, :]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return float(log_probs[target_token_id].item())


def run_q_b_logprob(tok, model, engine, kps) -> list[dict]:
    """For each N in {256, 1024, 2048, 4096, 8192}, run n trials. Each
    trial:
      1. Pick a 3-digit room number.
      2. Build fact: "The Velmoor Conference is in Hall Brennan-NNN."
      3. Inject condition: store fact, retrieve_and_inject to get
         cache populated with fact-KV.
      4. Forward N tokens of filler with that cache (extends to F+N).
      5. Build completion prompt: " The Velmoor Conference is in Hall Brennan-"
         (ending right before the answer).
      6. Get logprob of the planted answer's first token in this state.
      7. Control: same, but no fact-KV (empty starting cache).
      8. Record both logprobs and their delta.
    """
    Ns = [256, 1024, 2048, 4096, 8192]
    trials: list[dict] = []
    rng = random.Random(11)

    for n_filler in Ns:
        for trial_idx in range(N_TRIALS_PER_CELL):
            room_num = rng.randint(100, 999)
            room = f"Hall Brennan-{room_num}"
            fact = f"The Velmoor Conference is in {room}."

            # Tokenize the planted answer's first token. Format is
            # "Brennan-NNN" — the NNN portion may be one or multiple tokens
            # depending on the tokenizer. Take the FIRST token after
            # "Brennan-" as the target.
            full_text = f"The Velmoor Conference is in Hall Brennan-{room_num}"
            full_ids = tok.encode(full_text, add_special_tokens=False)
            prefix_text = "The Velmoor Conference is in Hall Brennan-"
            prefix_ids = tok.encode(prefix_text, add_special_tokens=False)
            # Common-prefix trim: drop tokens that match. Whatever's left
            # after the prefix is the room-number portion.
            i = 0
            while i < len(prefix_ids) and i < len(full_ids) and prefix_ids[i] == full_ids[i]:
                i += 1
            answer_token_id = full_ids[i] if i < len(full_ids) else None
            if answer_token_id is None:
                continue  # shouldn't happen with these prompts

            # Build the completion prompt that ends with "...Hall Brennan-".
            # We append this prompt AFTER the filler when measuring logprob.
            completion_prompt = " " + prefix_text
            completion_ids = tok.encode(completion_prompt, add_special_tokens=False)
            completion_t = torch.tensor([completion_ids], device=model.device)

            t_start = time.perf_counter()

            # --- INJECT condition -----------------------------------------
            pack_id = kps.store(fact, auto_link=False)
            cache, _, _ = kps.retrieve_and_inject(
                "Where is the Velmoor Conference?",
            )
            filler_ids = build_filler_tokens(tok, n_filler)
            filler_t = torch.tensor([filler_ids], device=model.device)
            try:
                with torch.no_grad():
                    ext = model(filler_t, past_key_values=cache, use_cache=True)
                    inject_cache = ext.past_key_values
                logprob_inject = logprob_of_token(model, inject_cache,
                                                    completion_t, answer_token_id)
                err_inject = None
            except Exception as e:
                logprob_inject = float("-inf")
                err_inject = f"{type(e).__name__}: {e}"

            try:
                kps.forget(pack_id)
            except Exception:
                pass

            # --- CONTROL condition (no injection) -------------------------
            try:
                with torch.no_grad():
                    ext = model(filler_t, use_cache=True)
                    control_cache = ext.past_key_values
                logprob_control = logprob_of_token(model, control_cache,
                                                     completion_t, answer_token_id)
                err_control = None
            except Exception as e:
                logprob_control = float("-inf")
                err_control = f"{type(e).__name__}: {e}"

            # --- Record ---------------------------------------------------
            delta = logprob_inject - logprob_control  # log(P_inj / P_ctl)
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            trials.append({
                "n_filler": n_filler,
                "trial_idx": trial_idx,
                "room_num": room_num,
                "answer_token_id": answer_token_id,
                "logprob_inject": logprob_inject,
                "logprob_control": logprob_control,
                "delta_log_ratio": delta,
                "error_inject": err_inject,
                "error_control": err_control,
                "elapsed_ms": elapsed_ms,
            })

        # Per-N summary
        cell = [t for t in trials if t["n_filler"] == n_filler]
        deltas = [t["delta_log_ratio"] for t in cell
                  if math.isfinite(t["delta_log_ratio"])]
        inj_lp = [t["logprob_inject"] for t in cell
                   if math.isfinite(t["logprob_inject"])]
        ctl_lp = [t["logprob_control"] for t in cell
                   if math.isfinite(t["logprob_control"])]
        if deltas:
            mean_delta = statistics.mean(deltas)
            median_delta = statistics.median(deltas)
            mean_inj = statistics.mean(inj_lp)
            mean_ctl = statistics.mean(ctl_lp)
            print(f"  [Q-B logprob] N={n_filler:>5}  "
                  f"meanΔ={mean_delta:+.3f}  medianΔ={median_delta:+.3f}  "
                  f"E[logp_inj]={mean_inj:.3f}  E[logp_ctl]={mean_ctl:.3f}  "
                  f"n={len(cell)}", flush=True)
        else:
            print(f"  [Q-B logprob] N={n_filler:>5}  (no finite trials)",
                  flush=True)

    return trials


def main() -> int:
    print(f"[{time.strftime('%H:%M:%S')}] N_TRIALS_PER_CELL={N_TRIALS_PER_CELL}")
    with tempfile.TemporaryDirectory() as tmpdir:
        tok, model, engine, kps = load_model_and_engine(Path(tmpdir))

        print(f"[{time.strftime('%H:%M:%S')}] === Q-B logprob sweep ===")
        trials = run_q_b_logprob(tok, model, engine, kps)

        results = {
            "model_id": MODEL_ID,
            "quantization": "4bit-nf4-bf16",
            "n_trials_per_cell": N_TRIALS_PER_CELL,
            "device": DEVICE,
            "metric": "next-token logprob of planted-answer first token",
            "q_b_logprob_trials": trials,
        }
        out_path = HERE / f"results_logprob.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"[{time.strftime('%H:%M:%S')}] wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
