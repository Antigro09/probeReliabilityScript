"""Apply the Phase-1 scored-object audit to a released result table.

The command accepts JSONL or CSV after an external pipeline exports one row
per candidate. It checks whether the target varies inside each ranking cell,
measures saturation, detects duplicate candidate rows, and compares max with
mean aggregation when per-method scores are available.

Example:
    python -m scripts.audit_external_results \
      --input external/results.jsonl \
      --cell-fields model,layer,task \
      --candidate-fields architecture,seed \
      --target-field reliability \
      --per-method-field per_method \
      --out external/audit.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--cell-fields", required=True)
    p.add_argument("--candidate-fields", required=True)
    p.add_argument("--target-field", required=True)
    p.add_argument("--per-method-field", default=None)
    p.add_argument("--saturation-delta", type=float, default=0.02)
    p.add_argument("--tie-tolerance", type=float, default=1e-9)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def _fields(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def _get(row, dotted):
    cur = row
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"missing field {dotted!r}")
        cur = cur[part]
    return cur


def load_rows(path: Path):
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        return rows
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    raise SystemExit("--input must end in .jsonl or .csv")


def _number(value, field):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{field!r} must be numeric or null, got {value!r}") from exc
    return number if math.isfinite(number) else None


def audit(rows, cell_fields, candidate_fields, target_field,
          per_method_field=None, saturation_delta=0.02, tie_tolerance=1e-9):
    if not rows:
        raise ValueError("input contains no rows")
    by_cell = defaultdict(list)
    seen = set()
    duplicates = []
    null_targets = 0
    target_values = []

    for row_index, row in enumerate(rows):
        try:
            cell = tuple(_get(row, field) for field in cell_fields)
            candidate = tuple(_get(row, field) for field in candidate_fields)
            target = _number(_get(row, target_field), target_field)
        except KeyError as exc:
            raise SystemExit(f"row {row_index}: {exc}") from exc
        key = cell + candidate
        if key in seen:
            duplicates.append(key)
        seen.add(key)
        if target is None:
            null_targets += 1
        else:
            target_values.append(target)
        by_cell[cell].append((candidate, target, row))

    distinct_hist = Counter()
    cell_records = []
    for cell, entries in by_cell.items():
        finite = [target for _, target, _ in entries if target is not None]
        unique = []
        for value in sorted(finite):
            if not unique or abs(value - unique[-1]) > tie_tolerance:
                unique.append(value)
        distinct_hist[len(unique)] += 1
        cell_records.append({
            "cell": list(cell),
            "n_candidates": len(entries),
            "n_nonnull": len(finite),
            "n_distinct_target": len(unique),
            "target_range": max(finite) - min(finite) if finite else None,
        })

    n_cells = len(by_cell)
    constant_cells = sum(record["n_distinct_target"] <= 1 for record in cell_records)
    saturation = (
        float(np.mean(np.asarray(target_values) >= 1.0 - saturation_delta))
        if target_values else None
    )

    method_summary = None
    if per_method_field:
        gaps = []
        method_saturation = Counter()
        method_counts = Counter()
        malformed = 0
        for row_index, row in enumerate(rows):
            try:
                methods = _get(row, per_method_field)
            except KeyError as exc:
                raise SystemExit(f"row {row_index}: {exc}") from exc
            if not isinstance(methods, dict):
                malformed += 1
                continue
            values = []
            for name, raw in methods.items():
                value = raw.get("R") if isinstance(raw, dict) else raw
                value = _number(value, f"{per_method_field}.{name}")
                if value is None:
                    continue
                values.append(value)
                method_counts[name] += 1
                method_saturation[name] += int(value >= 1.0 - saturation_delta)
            if values:
                gaps.append(max(values) - float(np.mean(values)))
        method_summary = {
            "n_rows_with_methods": len(gaps),
            "n_malformed": malformed,
            "max_minus_mean_median": float(np.median(gaps)) if gaps else None,
            "max_minus_mean_p95": float(np.percentile(gaps, 95)) if gaps else None,
            "saturation_by_method": {
                name: method_saturation[name] / method_counts[name]
                for name in sorted(method_counts)
            },
        }

    return {
        "n_rows": len(rows),
        "n_cells": n_cells,
        "n_duplicate_candidate_rows": len(duplicates),
        "duplicate_candidate_keys_first_20": [list(x) for x in duplicates[:20]],
        "n_null_targets": null_targets,
        "distinct_target_histogram": {str(k): v for k, v in sorted(distinct_hist.items())},
        "fraction_cells_candidate_independent": constant_cells / n_cells if n_cells else None,
        "saturation_delta": saturation_delta,
        "target_saturation_fraction": saturation,
        "target_range_global": max(target_values) - min(target_values) if target_values else None,
        "per_method": method_summary,
        "cells": cell_records,
    }


def main():
    args = parse_args()
    result = audit(
        load_rows(args.input),
        _fields(args.cell_fields),
        _fields(args.candidate_fields),
        args.target_field,
        per_method_field=args.per_method_field,
        saturation_delta=args.saturation_delta,
        tie_tolerance=args.tie_tolerance,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"Rows={result['n_rows']} cells={result['n_cells']}")
    print(f"Candidate-independent cells={result['fraction_cells_candidate_independent']:.1%}")
    print(f"Target saturation={result['target_saturation_fraction']}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
