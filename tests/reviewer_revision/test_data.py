from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.extraction import (
    _last_token_hidden,
    _validate_extraction_position,
    extract_all_layers,
)
from src.repro import hash_examples
from src.tasks import Example


class _FakeTokenizer:
    eos_token_id = 99
    sep_token_id = 98
    padding_side = "left"

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __call__(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids.clone(),
            "attention_mask": self.attention_mask.clone(),
        }

    def decode(self, token_ids: list[int]) -> str:
        return f"token-{token_ids[0]}"


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        del attention_mask
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 2) + 10 * self.calls
        self.calls += 1
        return SimpleNamespace(hidden_states=(hidden, hidden + 1.0))


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        ([[1, 1, 0, 0], [1, 1, 1, 0]], [1.0, 6.0]),
        ([[0, 0, 1, 1], [0, 1, 1, 1]], [3.0, 7.0]),
    ],
)
def test_last_token_hidden_supports_both_padding_sides(mask, expected) -> None:
    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4, 1)
    selected = _last_token_hidden(hidden, torch.tensor(mask))
    assert selected[:, 0].tolist() == expected


def test_last_token_hidden_rejects_a_row_without_attended_tokens() -> None:
    hidden = torch.zeros(2, 3, 4)
    mask = torch.tensor([[1, 1, 0], [0, 0, 0]])

    with pytest.raises(ValueError, match="no attended tokens"):
        _last_token_hidden(hidden, mask)


def test_extraction_validation_uses_true_last_attended_index_for_left_padding() -> None:
    tokenizer = _FakeTokenizer(
        input_ids=torch.tensor([[0, 0, 41, 42], [0, 51, 52, 53]]),
        attention_mask=torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]]),
    )
    bundle = SimpleNamespace(name="left-padded", tokenizer=tokenizer)

    report = _validate_extraction_position(bundle, ["a", "b"])

    assert report["last_token_positions"] == [3, 3]
    assert report["last_token_ids"] == [42, 53]
    assert report["last_token_strings"] == ["token-42", "token-53"]


def test_corrected_extraction_cache_provenance_round_trip(tmp_path: Path) -> None:
    from src.reviewer_revision.data import (
        load_validated_cache,
        select_representation_cache,
    )

    tokenizer = _FakeTokenizer(
        input_ids=torch.tensor([[0, 0, 41, 42], [0, 51, 52, 53]]),
        attention_mask=torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]]),
    )
    bundle = SimpleNamespace(
        name="fake/model",
        model=_FakeModel(),
        tokenizer=tokenizer,
        n_layers=1,
        hidden_size=2,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    examples = [
        Example("a", 0, 0),
        Example("b", 1, 1),
        Example("c", 0, 0),
        Example("d", 1, 1),
    ]
    cache_dir = tmp_path / "fake_model_sva"

    extract_all_layers(
        bundle,
        examples,
        [1],
        batch_size=2,
        cache_dir=cache_dir,
        cache_tag="inter",
    )
    selection = select_representation_cache(
        tmp_path,
        model_id="fake/model",
        task="sva",
        layer=1,
        tag="inter",
        expected_data_hash=hash_examples(examples),
    )
    loaded = load_validated_cache(selection)

    assert loaded.X.dtype == torch.float32
    assert selection.provenance["dtype"] == "torch.float32"
    assert selection.provenance["representation_dtype"] == "torch.float32"
    assert selection.provenance["model_forward_dtype"] == "torch.bfloat16"
    assert (
        selection.provenance["extraction_code_version"]
        == "last-nonpadding-mask-index-v2"
    )


def test_phase2_sst2_reconstruction_matches_archived_fold_hashes() -> None:
    from src.reviewer_revision.data import reconstruct_phase2_folds

    reconstruction = reconstruct_phase2_folds(
        "sst2", [Path("data/sst2.tsv")], max_examples=30_000, seed=42
    )

    assert reconstruction.n_loaded == 30_000
    assert reconstruction.n_deduplicated == 29_929
    assert reconstruction.fold_sizes == {
        "candidate": 10_524,
        "evaluator": 5_260,
        "intervention": 7_892,
        "test": 2_636,
    }
    assert reconstruction.fold_hashes == {
        "candidate": "e138d74b39d84198",
        "evaluator": "02c7bfa2a11d4ee7",
        "intervention": "bb4c90474ad84d6b",
        "test": "ae5663c139bf577b",
    }
    assert reconstruction.n_assigned == sum(reconstruction.fold_sizes.values())
    assert reconstruction.n_assigned == 26_312
    assert reconstruction.n_excluded == 3_617
    assert len(reconstruction.manifest) == reconstruction.n_deduplicated
    assert len({row["example_id"] for row in reconstruction.manifest}) == len(
        reconstruction.manifest
    )
    excluded = [row for row in reconstruction.manifest if not row["included"]]
    assert len(excluded) == reconstruction.n_excluded
    assert {row["exclusion_reason"] for row in excluded} == {
        "archived_phase2_four_cell_balance"
    }
    assert sorted(row["source_index"] for row in reconstruction.manifest) == list(
        range(reconstruction.n_deduplicated)
    )


def test_grouped_subdivision_is_deterministic_and_group_disjoint() -> None:
    from src.reviewer_revision.data import subdivide_grouped

    examples: list[Example] = []
    for group_index in range(48):
        zc = (group_index // 2) % 2
        ze = group_index % 2
        sentence = f"duplicate group {group_index}"
        examples.extend([Example(sentence, zc, ze), Example(sentence, zc, ze)])
    fractions = {
        "direction_fit": 1 / 3,
        "fresh_decoder_fit": 1 / 3,
        "orientation_calibration": 1 / 6,
        "final_test": 1 / 6,
    }

    first = subdivide_grouped(examples, fractions, seed=20260829, task_name="sst2")
    second = subdivide_grouped(examples, fractions, seed=20260829, task_name="sst2")

    assert first.subset_hashes == second.subset_hashes
    assert first.manifest == second.manifest
    assert first.source_data_hash == hash_examples(examples)
    assert {row["source_data_hash"] for row in first.manifest} == {
        first.source_data_hash
    }
    owner: dict[str, str] = {}
    for row in first.manifest:
        assert owner.setdefault(row["group_id"], row["subset"]) == row["subset"]
    assert set(first.folds) == set(fractions)
    assert sum(map(len, first.folds.values())) == len(examples)
    assert sorted(row["source_index"] for row in first.manifest) == list(
        range(len(examples))
    )


def _write_cache(
    directory: Path,
    *,
    filename_prefix: str,
    model_id: str,
    tag: str,
    layer: int,
    data_hash: str,
    X: torch.Tensor | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{filename_prefix}_{tag}_L{layer}_{data_hash}.pt"
    tensor = X if X is not None else torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]]
    )
    torch.save(
        {
            "X": tensor,
            "zc": torch.tensor([0, 0, 1, 1]),
            "ze": torch.tensor([0, 1, 0, 1]),
        },
        path,
    )
    provenance = {
        "model": model_id,
        "layer": layer,
        "n_examples": 4,
        "hidden_size": int(tensor.shape[1]),
        "dtype": str(tensor.dtype),
        "extraction_rule": "last non-padding input token",
        "data_hash": data_hash,
        "cache_tag": tag,
    }
    path.with_suffix(".json").write_text(json.dumps(provenance), encoding="utf-8")
    return path


def test_cache_selection_uses_verified_data_hash_not_lexicographic_order(tmp_path: Path) -> None:
    from src.reviewer_revision.data import (
        load_validated_cache,
        select_representation_cache,
    )

    model_id = "EleutherAI/pythia-160m"
    cache_dir = tmp_path / "EleutherAI_pythia-160m_sst2"
    _write_cache(
        cache_dir,
        filename_prefix="EleutherAI_pythia-160m",
        model_id=model_id,
        tag="inter",
        layer=6,
        data_hash="0000000000000000",
    )
    expected_path = _write_cache(
        cache_dir,
        filename_prefix="EleutherAI_pythia-160m",
        model_id=model_id,
        tag="inter",
        layer=6,
        data_hash="bb4c90474ad84d6b",
    )

    selection = select_representation_cache(
        tmp_path,
        model_id=model_id,
        task="sst2",
        layer=6,
        tag="inter",
        expected_data_hash="bb4c90474ad84d6b",
    )
    validated = load_validated_cache(
        selection,
        expected_zc=torch.tensor([0, 0, 1, 1]),
        expected_ze=torch.tensor([0, 1, 0, 1]),
        expected_group_ids=["a", "b", "c", "d"],
    )

    assert selection.path == expected_path
    assert selection.data_hash == "bb4c90474ad84d6b"
    assert selection.cache_sha256 == hashlib.sha256(expected_path.read_bytes()).hexdigest()
    assert validated.n_examples == 4
    assert validated.mean_feature_variance > 0
    assert set(validated.class_conditioned_variance) == {0, 1}
    assert set(validated.target_class_conditioned_variance) == {0, 1}
    assert set(validated.control_class_conditioned_variance) == {0, 1}


def test_cache_selection_rejects_stale_only_candidates(tmp_path: Path) -> None:
    from src.reviewer_revision.data import (
        CacheSelectionError,
        select_representation_cache,
    )

    model_id = "EleutherAI/pythia-160m"
    cache_dir = tmp_path / "EleutherAI_pythia-160m_sst2"
    _write_cache(
        cache_dir,
        filename_prefix="EleutherAI_pythia-160m",
        model_id=model_id,
        tag="inter",
        layer=6,
        data_hash="0000000000000000",
    )

    with pytest.raises(CacheSelectionError, match="bb4c90474ad84d6b"):
        select_representation_cache(
            tmp_path,
            model_id=model_id,
            task="sst2",
            layer=6,
            tag="inter",
            expected_data_hash="bb4c90474ad84d6b",
        )


def test_cache_selection_rejects_more_than_one_exact_candidate(tmp_path: Path) -> None:
    from src.reviewer_revision.data import (
        CacheSelectionError,
        select_representation_cache,
    )

    model_id = "EleutherAI/pythia-160m"
    cache_dir = tmp_path / "EleutherAI_pythia-160m_sst2"
    for prefix in ("EleutherAI_pythia-160m", "copy_EleutherAI_pythia-160m"):
        _write_cache(
            cache_dir,
            filename_prefix=prefix,
            model_id=model_id,
            tag="inter",
            layer=6,
            data_hash="bb4c90474ad84d6b",
        )

    with pytest.raises(CacheSelectionError, match="ambiguous"):
        select_representation_cache(
            tmp_path,
            model_id=model_id,
            task="sst2",
            layer=6,
            tag="inter",
            expected_data_hash="bb4c90474ad84d6b",
        )


def test_cache_validation_rejects_collapsed_representations(tmp_path: Path) -> None:
    from src.reviewer_revision.data import (
        CacheValidationError,
        load_validated_cache,
        select_representation_cache,
    )

    model_id = "Qwen/Qwen2.5-1.5B"
    cache_dir = tmp_path / "Qwen_Qwen2.5-1.5B_sst2"
    _write_cache(
        cache_dir,
        filename_prefix="Qwen_Qwen2.5-1.5B",
        model_id=model_id,
        tag="inter",
        layer=14,
        data_hash="bb4c90474ad84d6b",
        X=torch.ones(4, 2),
    )
    selection = select_representation_cache(
        tmp_path,
        model_id=model_id,
        task="sst2",
        layer=14,
        tag="inter",
        expected_data_hash="bb4c90474ad84d6b",
    )

    with pytest.raises(CacheValidationError, match="collapsed"):
        load_validated_cache(selection)


def test_cache_validation_rejects_nonfinite_labels_and_group_mismatches(
    tmp_path: Path,
) -> None:
    from src.reviewer_revision.data import (
        CacheValidationError,
        load_validated_cache,
        select_representation_cache,
    )

    model_id = "Qwen/Qwen2.5-1.5B"
    cache_dir = tmp_path / "Qwen_Qwen2.5-1.5B_sst2"
    _write_cache(
        cache_dir,
        filename_prefix="Qwen_Qwen2.5-1.5B",
        model_id=model_id,
        tag="inter",
        layer=14,
        data_hash="bb4c90474ad84d6b",
        X=torch.tensor(
            [[0.0, 1.0], [1.0, 0.0], [float("nan"), 1.0], [3.0, 0.0]]
        ),
    )
    selection = select_representation_cache(
        tmp_path,
        model_id=model_id,
        task="sst2",
        layer=14,
        tag="inter",
        expected_data_hash="bb4c90474ad84d6b",
    )
    with pytest.raises(CacheValidationError, match="NaN"):
        load_validated_cache(selection)

    _write_cache(
        cache_dir,
        filename_prefix="Qwen_Qwen2.5-1.5B",
        model_id=model_id,
        tag="inter",
        layer=14,
        data_hash="bb4c90474ad84d6b",
    )
    selection = select_representation_cache(
        tmp_path,
        model_id=model_id,
        task="sst2",
        layer=14,
        tag="inter",
        expected_data_hash="bb4c90474ad84d6b",
    )
    with pytest.raises(CacheValidationError, match="group-ID"):
        load_validated_cache(selection, expected_group_ids=["too", "short"])
    with pytest.raises(CacheValidationError, match="target labels"):
        load_validated_cache(selection, expected_zc=torch.tensor([1, 1, 0, 0]))


def test_cache_validation_rejects_hidden_size_and_dtype_sidecar_mismatches(tmp_path: Path):
    from src.reviewer_revision.data import (
        CacheValidationError,
        load_validated_cache,
        select_representation_cache,
    )

    model_id = "Qwen/Qwen2.5-1.5B"
    cache_dir = tmp_path / "Qwen_Qwen2.5-1.5B_sst2"
    path = _write_cache(
        cache_dir,
        filename_prefix="Qwen_Qwen2.5-1.5B",
        model_id=model_id,
        tag="inter",
        layer=14,
        data_hash="bb4c90474ad84d6b",
    )
    sidecar = path.with_suffix(".json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance["hidden_size"] = 999
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")
    selection = select_representation_cache(
        tmp_path,
        model_id=model_id,
        task="sst2",
        layer=14,
        tag="inter",
        expected_data_hash="bb4c90474ad84d6b",
    )
    with pytest.raises(CacheValidationError, match="hidden size"):
        load_validated_cache(selection)

    provenance["hidden_size"] = 2
    provenance["dtype"] = "torch.float64"
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")
    selection = select_representation_cache(
        tmp_path,
        model_id=model_id,
        task="sst2",
        layer=14,
        tag="inter",
        expected_data_hash="bb4c90474ad84d6b",
    )
    with pytest.raises(CacheValidationError, match="dtype"):
        load_validated_cache(selection)


def test_cache_validation_accepts_legacy_forward_dtype_for_float32_representations(
    tmp_path: Path,
):
    from src.reviewer_revision.data import (
        load_validated_cache,
        select_representation_cache,
    )

    model_id = "Qwen/Qwen2.5-1.5B"
    cache_dir = tmp_path / "Qwen_Qwen2.5-1.5B_sst2"
    path = _write_cache(
        cache_dir,
        filename_prefix="Qwen_Qwen2.5-1.5B",
        model_id=model_id,
        tag="inter",
        layer=14,
        data_hash="bb4c90474ad84d6b",
    )
    provenance = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    provenance["dtype"] = "torch.bfloat16"
    path.with_suffix(".json").write_text(json.dumps(provenance), encoding="utf-8")
    selection = select_representation_cache(
        tmp_path,
        model_id=model_id,
        task="sst2",
        layer=14,
        tag="inter",
        expected_data_hash="bb4c90474ad84d6b",
    )

    cache = load_validated_cache(selection)

    assert cache.X.dtype == torch.float32


def test_cache_selection_checks_requested_cache_sha256(tmp_path: Path) -> None:
    from src.reviewer_revision.data import (
        CacheSelectionError,
        select_representation_cache,
    )

    model_id = "Qwen/Qwen2.5-1.5B"
    cache_dir = tmp_path / "Qwen_Qwen2.5-1.5B_sst2"
    path = _write_cache(
        cache_dir,
        filename_prefix="Qwen_Qwen2.5-1.5B",
        model_id=model_id,
        tag="inter",
        layer=14,
        data_hash="bb4c90474ad84d6b",
    )
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    selection = select_representation_cache(
        tmp_path,
        model_id=model_id,
        task="sst2",
        layer=14,
        tag="inter",
        expected_data_hash="bb4c90474ad84d6b",
        expected_cache_sha256=actual_sha256,
    )
    assert selection.cache_sha256 == actual_sha256
    with pytest.raises(CacheSelectionError, match="SHA-256"):
        select_representation_cache(
            tmp_path,
            model_id=model_id,
            task="sst2",
            layer=14,
            tag="inter",
            expected_data_hash="bb4c90474ad84d6b",
            expected_cache_sha256="0" * 64,
        )


def test_cross_layer_validation_requires_identical_split_order(tmp_path: Path) -> None:
    from src.reviewer_revision.data import (
        CacheValidationError,
        load_validated_cache,
        select_representation_cache,
        validate_cross_layer_cache_identity,
    )

    model_id = "Qwen/Qwen2.5-1.5B"
    cache_dir = tmp_path / "Qwen_Qwen2.5-1.5B_sst2"
    for layer in (8, 14):
        _write_cache(
            cache_dir,
            filename_prefix="Qwen_Qwen2.5-1.5B",
            model_id=model_id,
            tag="inter",
            layer=layer,
            data_hash="bb4c90474ad84d6b",
        )
    caches = {}
    for layer in (8, 14):
        selection = select_representation_cache(
            tmp_path,
            model_id=model_id,
            task="sst2",
            layer=layer,
            tag="inter",
            expected_data_hash="bb4c90474ad84d6b",
        )
        caches[layer] = load_validated_cache(
            selection, expected_group_ids=["a", "b", "c", "d"]
        )
    validate_cross_layer_cache_identity(caches)

    bad_labels = replace(caches[14], zc=torch.tensor([1, 1, 0, 0]))
    with pytest.raises(CacheValidationError, match="label order"):
        validate_cross_layer_cache_identity({8: caches[8], 14: bad_labels})
    bad_groups = replace(caches[14], group_ids=("d", "c", "b", "a"))
    with pytest.raises(CacheValidationError, match="group IDs"):
        validate_cross_layer_cache_identity({8: caches[8], 14: bad_groups})
