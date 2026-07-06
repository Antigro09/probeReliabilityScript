"""
Regenerate data/gender.tsv from the Winogender schemas (WS2).

Why this exists
---------------
The v1 gender dataset was truncated at the word "that", immediately BEFORE
the pronoun. Every one of its 148 unique prefixes therefore appeared with
both FEM and MASC labels — the label was not a function of the input, the
task was unlearnable by construction, and gender probes sat at chance.
This script rebuilds the dataset with the pronoun INCLUDED in the input.

Design decisions (defaults follow the recommendation; final sign-off and
any overrides belong in PREREGISTRATION_v2.md):
    - --style full-sentence (default): keep the entire instantiated
      sentence. Last-token extraction then tests whether gender information
      persists to the end of the sentence — a non-trivial probing target.
    - --style post-pronoun: truncate immediately after the pronoun. The
      label is then trivially decodable from the final token embedding.
      Kept only as an explicit ablation option.
    - NEUTRAL occupations are DROPPED. The v1 loader mapped NEUTRAL->ze=1,
      polluting the selectivity feature with occupations that are not
      female-skewed. Ze must be a clean binary skew feature.
    - Only sentences whose pronoun refers to the OCCUPATION are used
      (--referent occupation), so Zc (pronoun gender) and Ze (occupation
      skew) are properties of the same entity.

Source data (NO built-in fallback — a wrong path is an error, never a
silently substituted synthetic dataset; the v1 synthetic fallback is how a
broken dataset shipped twice):
    - templates:  winogender-schemas data/templates.tsv
                  (occupation, other-participant, answer, sentence with
                  $OCCUPATION / $PARTICIPANT / $*_PRONOUN placeholders)
    - stats:      winogender-schemas data/occupations-stats.tsv
                  (occupation, bergsma_pct_female, bls_pct_female)

Get them either by cloning https://github.com/rudinger/winogender-schemas
and passing --templates/--stats, or with --download (fetches the two files
from that repository and stores copies under data/winogender/ for
provenance; requires network access).

Usage:
    python -m scripts.generate_gender_data --download
    python -m scripts.generate_gender_data \
        --templates ../winogender-schemas/data/templates.tsv \
        --stats ../winogender-schemas/data/occupations-stats.tsv

Attribution: Winogender schemas are from Rudinger, Naradowsky, Leonard &
Van Durme, "Gender Bias in Coreference Resolution", NAACL 2018.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WINOGENDER_RAW_BASE = (
    "https://raw.githubusercontent.com/rudinger/winogender-schemas/master/data"
)

PRONOUNS = {
    "FEM": {"$NOM_PRONOUN": "she", "$POSS_PRONOUN": "her", "$ACC_PRONOUN": "her"},
    "MASC": {"$NOM_PRONOUN": "he", "$POSS_PRONOUN": "his", "$ACC_PRONOUN": "him"},
}
PRONOUN_TOKENS = {
    "FEM": {"she", "her", "hers"},
    "MASC": {"he", "his", "him"},
}
PRONOUN_PLACEHOLDERS = tuple(PRONOUNS["FEM"].keys())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--templates", type=Path, default=None,
                   help="Path to winogender-schemas templates.tsv")
    p.add_argument("--stats", type=Path, default=None,
                   help="Path to winogender-schemas occupations-stats.tsv")
    p.add_argument("--download", action="store_true",
                   help="Fetch templates/stats from the winogender-schemas "
                        "repository into data/winogender/ (requires network)")
    p.add_argument("--style", choices=["full-sentence", "post-pronoun"],
                   default="full-sentence",
                   help="full-sentence (recommended) keeps the whole sentence; "
                        "post-pronoun truncates right after the pronoun "
                        "(trivially decodable; ablation only)")
    p.add_argument("--referent", choices=["occupation", "participant", "both"],
                   default="occupation",
                   help="Which templates to use, by pronoun referent")
    p.add_argument("--stat-source", choices=["bls", "bergsma"], default="bls",
                   help="Which %%-female statistic decides occupation skew")
    p.add_argument("--fem-threshold", type=float, default=60.0,
                   help="pct_female >= this -> FEM_SKEW")
    p.add_argument("--masc-threshold", type=float, default=40.0,
                   help="pct_female <= this -> MASC_SKEW; occupations in "
                        "between are NEUTRAL and dropped")
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "data" / "gender.tsv")
    p.add_argument("--min-examples", type=int, default=150,
                   help="Refuse to write fewer examples than this. The "
                        "occupation-referent Winogender subset with NEUTRAL "
                        "occupations dropped yields 188 at the 60/40 BLS "
                        "thresholds; the default guards against gross "
                        "under-generation while accepting that full output. "
                        "Tighter skew thresholds shrink the set — pass a "
                        "lower value explicitly if you intend that.")
    return p.parse_args()


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def download_sources() -> tuple[Path, Path]:
    import urllib.request
    dest_dir = PROJECT_ROOT / "data" / "winogender"
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for fname in ("templates.tsv", "occupations-stats.tsv"):
        url = f"{WINOGENDER_RAW_BASE}/{fname}"
        dest = dest_dir / fname
        print(f"[download] {url} -> {dest}")
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
        except Exception as e:
            fail(f"could not download {url}: {e}\n"
                 f"Clone https://github.com/rudinger/winogender-schemas and "
                 f"pass --templates/--stats instead.")
        dest.write_bytes(data)
        out_paths.append(dest)
    return out_paths[0], out_paths[1]


def read_tsv(path: Path) -> list[list[str]]:
    if not path.exists():
        fail(f"{path} does not exist. See --help for how to obtain the "
             f"winogender-schemas source files. There is no fallback.")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                rows.append(line.split("\t"))
    if not rows:
        fail(f"{path} is empty")
    return rows


def load_stats(path: Path, source: str) -> dict[str, float]:
    rows = read_tsv(path)
    header = [h.strip().lower() for h in rows[0]]
    try:
        occ_col = header.index("occupation")
    except ValueError:
        fail(f"{path}: expected a header with an 'occupation' column, "
             f"got {header}")
    col_name = "bls_pct_female" if source == "bls" else "bergsma_pct_female"
    try:
        pct_col = header.index(col_name)
    except ValueError:
        fail(f"{path}: no column {col_name!r} in header {header}")
    stats = {}
    for i, row in enumerate(rows[1:], start=2):
        try:
            stats[row[occ_col].strip().lower()] = float(row[pct_col])
        except (IndexError, ValueError) as e:
            fail(f"{path}:{i}: cannot parse row {row}: {e}")
    return stats


def load_templates(path: Path) -> list[dict]:
    """Parse templates.tsv defensively.

    Expected columns: occupation, other-participant, answer, sentence.
    The answer column marks the pronoun's referent; the known encoding is
    0 = occupation, 1 = participant, and referent words are also accepted.
    Anything else is a hard error — silent skipping of unparseable rows is
    exactly how a broken dataset ships unnoticed.
    """
    rows = read_tsv(path)
    # Skip a header row if present (first cell names the column rather than
    # containing an actual occupation word with stats).
    start = 1 if "occupation" in rows[0][0].strip().lower() else 0
    templates = []
    for i, row in enumerate(rows[start:], start=start + 1):
        if len(row) != 4:
            fail(f"{path}:{i}: expected 4 tab-separated columns "
                 f"(occupation, other-participant, answer, sentence), "
                 f"got {len(row)}: {row}")
        occupation, participant, answer, sentence = (c.strip() for c in row)
        ans_norm = answer.lower()
        if ans_norm in {"0", occupation.lower()}:
            referent = "occupation"
        elif ans_norm in {"1", participant.lower()}:
            referent = "participant"
        else:
            fail(f"{path}:{i}: cannot interpret answer column {answer!r} "
                 f"(occupation={occupation!r}, participant={participant!r}). "
                 f"Update load_templates() for this format — do not skip.")
        if "$OCCUPATION" not in sentence:
            fail(f"{path}:{i}: sentence has no $OCCUPATION placeholder: "
                 f"{sentence!r}")
        if not any(ph in sentence for ph in PRONOUN_PLACEHOLDERS):
            fail(f"{path}:{i}: sentence has no pronoun placeholder: "
                 f"{sentence!r}")
        templates.append({
            "occupation": occupation.lower(),
            "participant": participant.lower(),
            "referent": referent,
            "sentence": sentence,
        })
    return templates


def instantiate(template: dict, gender: str, use_someone: bool) -> str:
    """Fill placeholders for one (template, gender, participant-variant)."""
    sent = template["sentence"]
    sent = sent.replace("$OCCUPATION", template["occupation"])
    if use_someone:
        # Winogender's instantiation drops the article: "the someone" -> "someone".
        sent = sent.replace("the $PARTICIPANT", "someone")
        sent = sent.replace("The $PARTICIPANT", "Someone")
        sent = sent.replace("$PARTICIPANT", "someone")
    else:
        sent = sent.replace("$PARTICIPANT", template["participant"])
    for placeholder, word in PRONOUNS[gender].items():
        sent = sent.replace(placeholder, word)
    if "$" in sent:
        fail(f"unreplaced placeholder in instantiated sentence: {sent!r} "
             f"(template: {template['sentence']!r})")
    return sent


def truncate_post_pronoun(sentence: str, gender: str) -> str:
    """Cut the sentence immediately after the first gendered pronoun."""
    tokens = sentence.split(" ")
    for i, tok in enumerate(tokens):
        if tok.strip(".,;:!?\"'").lower() in PRONOUN_TOKENS[gender]:
            # Strip trailing punctuation so the pronoun is the final token.
            return " ".join(tokens[:i + 1]).rstrip(".,;:!?\"'")
    fail(f"no {gender} pronoun found in {sentence!r}; cannot truncate")


def main():
    args = parse_args()
    if args.download:
        templates_path, stats_path = download_sources()
    else:
        if args.templates is None or args.stats is None:
            fail("provide --templates and --stats, or use --download. "
                 "There is deliberately no bundled fallback dataset.")
        templates_path, stats_path = args.templates, args.stats

    templates = load_templates(templates_path)
    stats = load_stats(stats_path, args.stat_source)
    print(f"[load] {len(templates)} templates, {len(stats)} occupation stats")

    if args.masc_threshold >= args.fem_threshold:
        fail("--masc-threshold must be below --fem-threshold")

    def skew_of(occupation: str) -> str | None:
        if occupation not in stats:
            fail(f"occupation {occupation!r} from templates has no entry in "
                 f"{stats_path}; refusing to guess its skew")
        pct = stats[occupation]
        if pct >= args.fem_threshold:
            return "FEM_SKEW"
        if pct <= args.masc_threshold:
            return "MASC_SKEW"
        return None  # NEUTRAL -> dropped

    kept, dropped_neutral, dropped_referent = [], set(), 0
    for tpl in templates:
        if args.referent != "both" and tpl["referent"] != args.referent:
            dropped_referent += 1
            continue
        skew = skew_of(tpl["occupation"])
        if skew is None:
            dropped_neutral.add(tpl["occupation"])
            continue
        for gender in ("FEM", "MASC"):
            for use_someone in (False, True):
                sent = instantiate(tpl, gender, use_someone)
                if args.style == "post-pronoun":
                    sent = truncate_post_pronoun(sent, gender)
                kept.append((sent, gender, skew))

    print(f"[filter] dropped {dropped_referent} templates by referent, "
          f"{len(dropped_neutral)} NEUTRAL occupations: "
          f"{sorted(dropped_neutral)}")

    # ---- Self-checks: refuse to write a dataset with the v1 failure modes ----
    if len(kept) < args.min_examples:
        fail(f"only {len(kept)} examples generated (< {args.min_examples})")

    zc_by_sentence: dict[str, set] = defaultdict(set)
    for sent, gender, _ in kept:
        zc_by_sentence[sent].add(gender)
    conflicting = [s for s, g in zc_by_sentence.items() if len(g) > 1]
    if conflicting:
        fail(f"{len(conflicting)} sentences carry BOTH gender labels — this "
             f"is the v1 truncation bug. First offender: {conflicting[0]!r}")

    for sent, gender, _ in kept:
        toks = {t.strip(".,;:!?\"'").lower() for t in sent.split()}
        if not toks & PRONOUN_TOKENS[gender]:
            fail(f"sentence lacks its {gender} pronoun: {sent!r}")

    counts = Counter((g, s) for _, g, s in kept)
    print("[counts] per (gender, skew) cell:")
    for key in sorted(counts):
        print(f"    {key}: {counts[key]}")
    if len(counts) < 4:
        fail(f"expected all 4 (gender x skew) cells populated, got "
             f"{sorted(counts)}")

    # ---- Write output + provenance sidecar ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for sent, gender, skew in kept:
            if "\t" in sent:
                fail(f"tab inside sentence would corrupt the TSV: {sent!r}")
            f.write(f"{sent}\t{gender}\t{skew}\n")

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    sidecar = args.out.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps({
        "generator": "scripts/generate_gender_data.py",
        "git_commit": commit,
        "templates": str(templates_path),
        "stats": str(stats_path),
        "stat_source": args.stat_source,
        "style": args.style,
        "referent": args.referent,
        "fem_threshold": args.fem_threshold,
        "masc_threshold": args.masc_threshold,
        "dropped_neutral_occupations": sorted(dropped_neutral),
        "n_examples": len(kept),
        "cell_counts": {f"{g}/{s}": c for (g, s), c in sorted(counts.items())},
        "source_attribution": "Rudinger et al., NAACL 2018 (winogender-schemas)",
    }, indent=2))

    print(f"[write] {len(kept)} examples -> {args.out}")
    print(f"[write] provenance -> {sidecar}")


if __name__ == "__main__":
    main()
