"""
Unified representation extraction across all model architectures.

Design decision (locked, do not change without updating the methods section):
    For every model — BERT, GPT-2, Pythia, LLaMA, Qwen, Gemma — we extract
    hidden_states[layer] at the position of the LAST non-padding input token.

Rationale:
    - Cross-model comparability requires one extraction rule.
    - The Linzen prefix ends right before the target verb, so the last token
      is the model's most informed prediction-relevant state.
    - This deliberately departs from prior work that uses [MASK] for BERT.
      We document this in the paper.

Practical notes:
    - We use bf16 for all forward passes (saves VRAM, no accuracy hit at
      inference time on these models).
    - Output is float32 on CPU for downstream probe training.
    - All paths use pathlib so the code runs on Windows (5070) and Linux
      (Lambda) unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .data import LinzenExample  # for backward compatibility


# Protocol-style type alias: anything with .sentence, .zc, .ze works.
# Both LinzenExample and tasks.Example satisfy this.
ExampleLike = LinzenExample  # type alias - any duck-compatible object accepted at runtime


# Device selection -----------------------------------------------------------

def pick_device() -> torch.device:
    """Pick the best available device. Works on Windows + Linux."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS for Apple Silicon (not your case but cheap to support)
        return torch.device("mps")
    return torch.device("cpu")


# Dataset wrapper ------------------------------------------------------------

class LinzenTorchDataset(Dataset):
    def __init__(self, examples: list):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        return {"sentence": ex.sentence, "zc": ex.zc, "ze": ex.ze}


def make_collate_fn(tokenizer, max_length: int = 256,
                    add_special_tokens: bool = False):
    """
    Collate function for the DataLoader.

    add_special_tokens defaults to False because BERT tokenizers add [CLS]/[SEP]
    by default, which would put a [SEP] token at the last position and corrupt
    extraction. Causal LM tokenizers (GPT-2, LLaMA, etc.) typically don't add
    anything by default, so this is safe across all our models.
    """
    def collate(batch: list[dict]):
        sentences = [b["sentence"] for b in batch]
        zc = torch.tensor([b["zc"] for b in batch], dtype=torch.long)
        ze = torch.tensor([b["ze"] for b in batch], dtype=torch.long)
        toks = tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        return toks, zc, ze
    return collate


# Model loading --------------------------------------------------------------

@dataclass
class ModelBundle:
    """Everything we need to extract hidden states from a model."""
    name: str
    model: torch.nn.Module
    tokenizer: object
    n_layers: int
    hidden_size: int
    device: torch.device
    dtype: torch.dtype


def load_model(
    model_name: str,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = False,
) -> ModelBundle:
    """
    Load a HuggingFace model + tokenizer for representation extraction.

    We use AutoModel (not AutoModelForCausalLM / ForMaskedLM) because we only
    need hidden states, not the LM head. This saves memory.
    """
    device = device or pick_device()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        # Many causal LM tokenizers (GPT-2, LLaMA, Qwen) lack a pad token.
        # Using EOS as pad is fine because we mask via attention_mask.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        output_hidden_states=True,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Robustly determine layer count and hidden size from config
    config = model.config
    n_layers = (
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "n_layer", None)
        or getattr(config, "n_layers", None)
    )
    hidden_size = (
        getattr(config, "hidden_size", None)
        or getattr(config, "n_embd", None)
        or getattr(config, "d_model", None)
    )
    if n_layers is None or hidden_size is None:
        raise RuntimeError(
            f"Could not determine n_layers / hidden_size for {model_name}"
        )

    return ModelBundle(
        name=model_name,
        model=model,
        tokenizer=tokenizer,
        n_layers=n_layers,
        hidden_size=hidden_size,
        device=device,
        dtype=dtype,
    )


def select_layers(n_layers: int, k: int = 5) -> list[int]:
    """
    Pick k layers spread across the model depth.
    For Pythia-160M (12 layers) with k=5: [1, 4, 6, 9, 12]
    For LLaMA-3.2-3B (28 layers) with k=5: [1, 7, 14, 21, 28]
    Layer 0 (embeddings) is excluded; we want post-block residual states.
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    if k > n_layers:
        return list(range(1, n_layers + 1))
    # Linspace from 1 to n_layers inclusive, then dedupe + sort
    import numpy as np
    layers = np.linspace(1, n_layers, k).round().astype(int).tolist()
    return sorted(set(layers))


# Extraction -----------------------------------------------------------------

@torch.no_grad()
def _last_token_hidden(hidden_states: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
    """
    hidden_states: (B, T, H), attention_mask: (B, T) with 1 for real tokens.
    Returns (B, H): hidden state at the LAST real token in each sequence.
    """
    # last_idx[b] = index of last 1 in attention_mask[b]
    seq_lens = attention_mask.sum(dim=1) - 1  # (B,)
    batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_idx, seq_lens]


def _validate_extraction_position(
    bundle: ModelBundle,
    sample_sentences: list[str],
    max_length: int = 256,
) -> dict:
    """
    Sanity-check the extraction position once before processing the full
    dataset. Verifies that:
        1. The "last non-pad token" position actually corresponds to the
           final word of the prefix (not an EOS/BOS token added by the
           tokenizer).
        2. No tokenizer is silently adding extra special tokens that would
           push the meaningful content away from the last position.

    Returns a dict of diagnostic info. Raises if something is wrong.
    """
    tokenizer = bundle.tokenizer
    encoded = tokenizer(
        sample_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    last_pos = attention_mask.sum(dim=1) - 1
    last_ids = input_ids[torch.arange(len(input_ids)), last_pos]
    last_strs = [tokenizer.decode([tid]).strip() for tid in last_ids.tolist()]

    # Check for silently-added special tokens at the end
    eos_id = tokenizer.eos_token_id
    sep_id = getattr(tokenizer, "sep_token_id", None)  # BERT
    bad = []
    for i, tid in enumerate(last_ids.tolist()):
        if tid == eos_id and eos_id is not None:
            bad.append((i, "eos", last_strs[i]))
        elif tid == sep_id and sep_id is not None:
            bad.append((i, "sep", last_strs[i]))

    info = {
        "model": bundle.name,
        "n_samples_checked": len(sample_sentences),
        "last_token_strings": last_strs[:5],
        "special_token_at_end": bad,
        "tokenizer_class": type(tokenizer).__name__,
    }

    if bad:
        raise RuntimeError(
            f"[{bundle.name}] Tokenizer is appending special tokens "
            f"({set(b[1] for b in bad)}) to the end of inputs. The last-token "
            f"extraction position would point to a special token, not the "
            f"prefix's final word. Set add_special_tokens=False or strip them."
            f"\nFirst-5 last tokens decoded: {last_strs[:5]}"
        )
    return info


@torch.no_grad()
def extract_layer_reps(
    bundle: ModelBundle,
    examples: list,
    layer_idx: int,
    batch_size: int = 32,
    max_length: int = 256,
    show_progress: bool = True,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract last-token residual stream representations at a single layer.

    Returns:
        X: (N, H) float32 CPU tensor of representations
        zc: (N,) long CPU tensor of target labels
        ze: (N,) long CPU tensor of attractor labels
    """
    if not (0 <= layer_idx <= bundle.n_layers):
        raise ValueError(
            f"layer_idx={layer_idx} out of range [0, {bundle.n_layers}]"
        )

    if validate:
        # Cheap once-per-call sanity check that catches tokenizer surprises
        sample = [ex.sentence for ex in examples[:8]]
        _validate_extraction_position(bundle, sample, max_length=max_length)

    ds = LinzenTorchDataset(examples)
    collate = make_collate_fn(bundle.tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate)

    X_chunks: list[torch.Tensor] = []
    zc_chunks: list[torch.Tensor] = []
    ze_chunks: list[torch.Tensor] = []

    iterator = tqdm(loader, desc=f"extract L{layer_idx}",
                    disable=not show_progress)
    for toks, zc, ze in iterator:
        toks = {k: v.to(bundle.device) for k, v in toks.items()}
        out = bundle.model(**toks)
        # hidden_states is a tuple of length n_layers + 1 (embeddings + each block)
        h = out.hidden_states[layer_idx]   # (B, T, H), bf16
        rep = _last_token_hidden(h, toks["attention_mask"])  # (B, H)
        X_chunks.append(rep.float().cpu())
        zc_chunks.append(zc)
        ze_chunks.append(ze)

    return (
        torch.cat(X_chunks, dim=0),
        torch.cat(zc_chunks, dim=0),
        torch.cat(ze_chunks, dim=0),
    )


def extract_all_layers(
    bundle: ModelBundle,
    examples: list,
    layers: Iterable[int],
    batch_size: int = 32,
    max_length: int = 256,
    cache_dir: Path | None = None,
) -> dict[int, dict]:
    """
    Extract representations for the given layers.

    Returns a dict mapping layer_idx -> {"X": ..., "zc": ..., "ze": ...}.
    If cache_dir is given, saves each layer's tensors plus a JSON
    provenance sidecar that records exactly how the cache was produced
    (model name, layer, dtype, dataset hash). Subsequent calls reuse the
    cache only if the provenance matches.
    """
    from .repro import hash_examples, write_provenance, read_provenance

    out: dict[int, dict] = {}
    layers = list(layers)

    # Validate position once for this model, not per layer
    sample = [ex.sentence for ex in examples[:8]]
    _validate_extraction_position(bundle, sample, max_length=max_length)

    data_hash = hash_examples(examples)

    for layer_idx in layers:
        cache_path = None
        prov_path = None
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            safe_name = bundle.name.replace("/", "_")
            cache_path = cache_dir / f"{safe_name}_L{layer_idx}_n{len(examples)}.pt"
            prov_path = cache_path.with_suffix(".json")

            if cache_path.exists() and prov_path.exists():
                prov = read_provenance(prov_path)
                if prov and prov.get("data_hash") == data_hash:
                    out[layer_idx] = torch.load(cache_path, weights_only=True)
                    continue
                # else: stale cache, fall through and re-extract

        X, zc, ze = extract_layer_reps(
            bundle, examples, layer_idx,
            batch_size=batch_size, max_length=max_length,
            validate=False,  # already validated above
        )
        out[layer_idx] = {"X": X, "zc": zc, "ze": ze}

        if cache_path is not None:
            torch.save(out[layer_idx], cache_path)
            write_provenance(prov_path, {
                "model": bundle.name,
                "layer": layer_idx,
                "n_examples": len(examples),
                "hidden_size": bundle.hidden_size,
                "n_layers": bundle.n_layers,
                "dtype": str(bundle.dtype),
                "extraction_rule": "last non-padding input token",
                "add_special_tokens": False,
                "data_hash": data_hash,
                "max_length": max_length,
            })
    return out
