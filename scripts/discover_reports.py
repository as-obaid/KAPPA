#!/usr/bin/env python3
"""Survey a stratified sample of reports with an LLM and aggregate the findings.

Argparse plus one call into kappa.labels, per the repo contract: logic lives in
the package so it is testable and importable, not here.

    python scripts/discover_reports.py --n-latin 150 --n-greek 50 \
        --model Qwen/Qwen3.5-122B-A10B --tp 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kappa.data.paths import data_root, raw_dir  # noqa: E402
from kappa.labels.discover import (  # noqa: E402
    FINDINGS_COLUMNS, RECORDS_COLUMNS, aggregate, build_prompt, findings_table,
    parse_response, per_script_quality, records_table, verify_spans,
)
from kappa.labels.engine import load_backend  # noqa: E402
from kappa.labels.schema import validate_extraction  # noqa: E402
from kappa.labels.sample import describe, load_reports, stratified_sample  # noqa: E402
from kappa.tracking import init_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-csv", type=Path, default=None,
                    help="defaults to <data_root>/meta/train.csv, then <raw>/train.csv")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="defaults to <data_root>/interim/discovery")
    ap.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "transformers"])
    ap.add_argument("--tp", type=int, default=2, help="tensor parallel size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="3072 truncated 1 of 300 reports mid-JSON; 4096 still fits "
                         "max-model-len 6144 against a measured 1,859-token prompt")
    ap.add_argument("--max-model-len", type=int, default=8192,
                    help="prompt+output budget; measured max prompt is 1,859 tokens")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                    help="raise toward 0.95 when tp is capped by the Slurm QOS")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="skip CUDA graph capture to buy back GPU memory")
    ap.add_argument("--n-latin", type=int, default=150)
    ap.add_argument("--n-turkish", type=int, default=50)
    ap.add_argument("--n-greek", type=int, default=50)
    ap.add_argument("--n-cyrillic", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total reports; use for a smoke test")
    ap.add_argument("--anchors-only", action="store_true",
                    help="extract the 58 annotated studies instead of unlabelled "
                         "ones, to validate the label derivation against ground "
                         "truth. Spends the project's only held-out set: measure "
                         "once, do not tune against it. See sample.py rule 1.")
    args = ap.parse_args()

    train_csv = args.train_csv
    if train_csv is None:
        for candidate in (data_root() / "meta" / "train.csv", raw_dir() / "train.csv"):
            if candidate.is_file():
                train_csv = candidate
                break
    if train_csv is None or not train_csv.is_file():
        print("error: could not find train.csv; pass --train-csv", file=sys.stderr)
        return 2

    out_dir = args.out_dir or (data_root() / "interim" / "discovery")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_reports(train_csv)
    per_script = {"latin": args.n_latin, "turkish": args.n_turkish,
                  "greek": args.n_greek, "cyrillic": args.n_cyrillic}
    sample = stratified_sample(rows, per_script=per_script, seed=args.seed,
                               anchors_only=args.anchors_only)
    if args.anchors_only:
        print("NOTE: --anchors-only — extracting the 58 annotated studies. "
              "This is the project's only ground truth; measure once.")
    if args.limit:
        # Round-robin by script so a smoke test still covers every writing system.
        by_script: dict[str, list[dict]] = {}
        for row in sample:
            by_script.setdefault(row["script"], []).append(row)
        interleaved: list[dict] = []
        while len(interleaved) < args.limit and any(by_script.values()):
            for bucket in by_script.values():
                if bucket and len(interleaved) < args.limit:
                    interleaved.append(bucket.pop(0))
        sample = interleaved

    print(f"train.csv : {train_csv}")
    print(f"corpus    : {len(rows)} studies, {sum(r['is_anchor'] for r in rows)} anchors")
    print(f"sample    : {len(sample)} reports — {describe(sample)}")
    print(f"model     : {args.model} (backend={args.backend}, tp={args.tp})")

    run = init_run(
        f"discover-{len(sample)}",
        job_type="reconnaissance",
        # Log the settings that change the output, not just the ones that name
        # the run, so two runs can be compared from their configs alone.
        config={"model": args.model, "backend": args.backend, "tp": args.tp,
                "seed": args.seed, "n": len(sample), "per_script": per_script,
                "max_tokens": args.max_tokens, "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "enforce_eager": args.enforce_eager,
                "temperature": 0.0},
        tags=["phase-01b", "discovery"],
    )

    backend = load_backend(
        args.backend, model=args.model, tp=args.tp, seed=args.seed,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    prompts = [build_prompt(r["report"]) for r in sample]
    print(f"generating {len(prompts)} responses...")
    t0 = time.time()
    outputs = backend.generate(prompts, max_tokens=args.max_tokens)
    elapsed = time.time() - t0
    print(f'generated in {elapsed / 60:.1f} min ({elapsed / max(len(prompts), 1):.2f} s/report)')
    # Sizing the full 4,407-report run is one of the things this pass is for --
    # but a linear extrapolation from a small batch is badly wrong, because
    # throughput improves with the number of prompts vLLM can schedule
    # together. Measured with a Gemma model on the same code and GPU: 5.65
    # s/report at n=5 versus 0.81 s/report at n=300, a 7x difference. So a smoke
    # test overstates the full run by roughly that factor.
    per_report = elapsed / max(len(prompts), 1)
    print(f'extrapolated to 4,407 reports: {4407 * per_report / 60:.0f} min '
          f'(at this batch size)')
    if len(prompts) < 100:
        print(f'  ^ batch of {len(prompts)} is too small to extrapolate from; '
              f'throughput measured ~7x better at n=300 on gemma, so read this '
              f'as an upper bound (~{4407 * per_report / 7 / 60:.0f} min)')

    records, n_parsed, n_valid, n_warned = [], 0, 0, 0
    for row, raw in zip(sample, outputs):
        parsed = parse_response(raw)
        # Parsing proves it is JSON; validation proves it is the JSON the prompt
        # asked for. Downstream reads the *validated* dump, so coerced fields (blank
        # strings, "null", unknown assertions) never reach the aggregates.
        if parsed is None:
            model, warns = None, ["response was not parseable JSON"]
        else:
            model, warns = validate_extraction(parsed)
        record = dict(model.model_dump()) if model is not None else {}
        record["_study_uid"] = row["study_uid"]
        record["_script"] = row["script"]
        record["_parsed"] = parsed is not None
        record["_valid"] = model is not None
        record["_schema_warnings"] = warns
        if warns:
            n_warned += 1
        if model is not None:
            n_parsed += 1
            n_valid += 1
            verbatim, tolerant, total = verify_spans(record, row["report"])
            # `_spans_ok` is the tolerant count -- see verify_spans. `_spans_exact`
            # is kept so a gap shows up as the normalisation artifact it is.
            record["_spans_ok"], record["_spans_total"] = tolerant, total
            record["_spans_exact"] = verbatim
        else:
            record["_raw"] = raw[:4000]
            record["_spans_ok"], record["_spans_total"] = 0, 0
            record["_spans_exact"] = 0
        records.append(record)

    records_path = out_dir / "records.jsonl"
    with open(records_path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    report_path = out_dir / "report.md"
    sources = {r["study_uid"]: r["report"] for r in sample}
    report_path.write_text(aggregate(records, sources), encoding="utf-8")

    spans_ok = sum(r["_spans_ok"] for r in records)
    spans_exact = sum(r["_spans_exact"] for r in records)
    spans_all = sum(r["_spans_total"] for r in records)
    print(f"parsed    : {n_parsed}/{len(records)}")
    print(f"validated : {n_valid}/{len(records)} "
          f"({n_warned} needed at least one coercion)")
    print(f"spans     : {spans_ok}/{spans_all} faithful"
          + (f" ({100 * spans_ok / spans_all:.1f}%)" if spans_all else ""))
    print(f"  of which: {spans_exact}/{spans_all} byte-exact"
          + (f" ({100 * spans_exact / spans_all:.1f}%)" if spans_all else "")
          + " — the gap is Unicode/whitespace, not paraphrasing")
    print(f"wrote     : {records_path}")
    print(f"wrote     : {report_path}")

    metrics = {
        "n_sampled": len(records),
        "n_parsed": n_parsed,
        "parse_rate": n_parsed / max(len(records), 1),
        "n_valid": n_valid,
        "schema_valid_rate": n_valid / max(len(records), 1),
        "n_schema_warned": n_warned,
        "spans_verbatim": spans_ok,
        "spans_exact": spans_exact,
        "spans_total": spans_all,
        "span_fidelity": spans_ok / spans_all if spans_all else 0.0,
        "span_fidelity_exact": spans_exact / spans_all if spans_all else 0.0,
        "seconds_per_report": elapsed / max(len(records), 1),
    }
    for row in per_script_quality(records):
        metrics[f"parse_rate/{row['script']}"] = row["parse_rate"]
        metrics[f"findings_per_report/{row['script']}"] = row["findings_per_report"]
        if row["span_fidelity"] is not None:
            metrics[f"span_fidelity/{row['script']}"] = row["span_fidelity"]
    run.log(metrics)

    # A browsable table beats a log line when the point is to read the output
    # and judge it. Two tables, deliberately:
    #
    # `records` is report-level and carries no findings blob. Serialising all
    # findings into one length-capped JSON string truncates the finding-dense
    # reports most worth reviewing into invalid JSON (21% of rows on a 300-report
    # run), and a cap that mangles the rows that matter is worse than no table.
    #
    # `findings` is one row per finding, which is the unit extraction quality is
    # actually judged in, and it carries the span plus its per-span verdict so a
    # reviewer can see *which* span failed rather than only how many did.
    try:
        import wandb

        rec_table = wandb.Table(columns=list(RECORDS_COLUMNS))
        for row in records_table(records, sources):
            rec_table.add_data(*row)

        find_table = wandb.Table(columns=list(FINDINGS_COLUMNS))
        for row in findings_table(records, sources):
            find_table.add_data(*row)

        run.log({"records": rec_table, "findings": find_table})
    except Exception as exc:  # noqa: BLE001
        print(f"note: could not log w&b tables ({exc})")

    run.finish()

    # A run that parsed nothing is a failed run, and it must exit nonzero. A
    # reasoning model can spend the whole token budget on a preamble and return
    # no parseable response at all; a zero exit would then let a dependent job
    # (for example a Slurm `afterok` chain) start on no data.
    if n_parsed == 0 and records:
        print(
            f"FAILED: 0 of {len(records)} responses were parseable JSON. "
            "Check the first `_raw` in records.jsonl -- a reasoning preamble "
            "that consumes the token budget looks exactly like this.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
