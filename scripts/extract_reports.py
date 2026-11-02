#!/usr/bin/env python3
"""Extract structured findings from every train report, resumably and in shards.

The reconnaissance script cannot do this job. It holds all records in memory and
writes once at the end, and the full corpus is 4,407 reports at a measured
0.81 s/report -- about 59 minutes, close to the walltime cap of a short GPU
queue. A run that reaches the wall would write nothing at all. So the pass is
sharded, incremental and resumable, and this script is the argparse shell over
`kappa.labels.shard` and `kappa.labels.engine`, which hold the arithmetic so it
can be tested without a GPU.

    # what will run, on a laptop, no GPU and no model load
    python scripts/extract_reports.py --shard 0/8 --dry-run

    # one array task
    python scripts/extract_reports.py --shard 0/8 --resume \
        --model google/gemma-4-31B-it --tp 1

Each shard writes its own file (`records.shard0of8.jsonl`), flushed and
`fsync`ed per record, so a walltime kill costs at most the batch inside
`generate()`. Re-running the same command with `--resume` re-extracts exactly
the studies that are missing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kappa.data.paths import data_root, raw_dir  # noqa: E402
from kappa.labels.discover import build_prompt, parse_response, verify_spans  # noqa: E402
from kappa.labels.engine import (  # noqa: E402
    DEFAULT_BATCH_SIZE, MAX_OUTPUT_TOKENS, Generation, align_assertion_confidence,
    generate_incremental, load_backend,
)
from kappa.labels.sample import load_reports  # noqa: E402
from kappa.labels.schema import ReportExtraction, validate_extraction  # noqa: E402
from kappa.labels.shard import (  # noqa: E402
    RecordWriter, WorkUnit, describe_units, parse_shard, pending, plan_units, read_done,
    records_for_unit, select_shard, shard_path,
)
from kappa.tracking import init_run  # noqa: E402

#: The measured throughput figure, used only to state an ETA in the log.
SECONDS_PER_REPORT = 0.81


def build_payload(generation: Generation, report: str, with_logprobs: bool) -> dict:
    """Turn one response into the fields every study sharing this text gets.

    Parsing proves the response is JSON; validation proves it is the JSON that
    was asked for. Both verdicts are recorded rather than being used to filter,
    because a study dropped here has no row downstream and no explanation for
    its absence, whereas a row carrying ``_valid: false`` is a counted failure.

    ``finish_reason == "length"`` is kept as ``_truncated``. A response cut at
    the token cap is still a string and still parses as JSON often enough to
    look healthy, so this is the only reliable signal that findings were lost.
    """
    parsed = parse_response(generation.text)
    if parsed is None:
        model, warns = None, ["response was not parseable JSON"]
    else:
        model, warns = validate_extraction(parsed)

    payload: dict = dict(model.model_dump()) if model is not None else {}
    payload["_parsed"] = parsed is not None
    payload["_valid"] = model is not None
    payload["_schema_warnings"] = warns
    payload["_finish_reason"] = generation.finish_reason
    payload["_truncated"] = generation.finish_reason == "length"
    payload["_n_output_tokens"] = generation.n_output_tokens

    if model is not None:
        verbatim, tolerant, total = verify_spans(payload, report)
        payload["_spans_ok"], payload["_spans_total"] = tolerant, total
        payload["_spans_exact"] = verbatim
    else:
        # The raw text is the only way to diagnose a non-JSON response later, and
        # a reasoning preamble that ate the token budget looks exactly like this.
        payload["_raw"] = generation.text[:4000]
        payload["_spans_ok"] = payload["_spans_total"] = payload["_spans_exact"] = 0

    # Confidences are aligned only when logprobs were actually requested.
    # Aligning without them would attach the "misaligned" warning to every
    # record of every transformers-backend run, which trains the reader to
    # ignore the one warning that matters.
    if with_logprobs:
        confidences, align_warnings = align_assertion_confidence(
            len(payload.get("findings", []) or []), generation.assertion_logprobs
        )
        payload["_assertion_confidence"] = confidences
        payload["_schema_warnings"] = list(warns) + align_warnings
    else:
        payload["_assertion_confidence"] = None
    return payload


def _resolve_train_csv(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.is_file() else None
    for candidate in (data_root() / "meta" / "train.csv", raw_dir() / "train.csv"):
        if candidate.is_file():
            return candidate
    return None


def _shard_of(units: list[WorkUnit], index: int, n: int) -> list[WorkUnit]:
    """``select_shard``, but loud when it selects nothing."""
    chosen = select_shard(units, index, n)
    if not chosen and units:
        print(f"warning: shard {index}/{n} selected 0 of {len(units)} units",
              file=sys.stderr)
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-csv", type=Path, default=None,
                    help="defaults to <data_root>/meta/train.csv, then <raw>/train.csv")
    ap.add_argument("--out", type=Path, default=None,
                    help="base JSONL path; defaults to "
                         "<data_root>/interim/extraction/records.jsonl. --shard i/n "
                         "inserts .shard<i>of<n> before the suffix")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="extract only the i-th of n shards, 0-indexed. Sharding is "
                         "by content hash and happens AFTER de-duplication, so two "
                         "studies sharing a report can never land in two shards")
    ap.add_argument("--resume", dest="resume", action="store_true", default=True,
                    help="skip studies the output file already holds (default). On by "
                         "default because Slurm may requeue an array task and a "
                         "requeue that redoes 8 minutes of GPU work is pure loss")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="extract everything in the shard, appending to the output")
    ap.add_argument("--overwrite", action="store_true",
                    help="discard the existing output first; required to start over")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of model calls; for a smoke test")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                    help="prompts per backend call, i.e. what a kill costs "
                         f"(default {DEFAULT_BATCH_SIZE})")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit without loading a model")

    ap.add_argument("--model", default="google/gemma-4-31B-it",
                    help="Hugging Face repo id; it must already be in the local HF "
                         "cache when running offline. Used with google/gemma-4-31B-it "
                         "(59G) and Qwen/Qwen3.5-27B (52G)")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "transformers"])
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor parallel size. Defaults to 1, for queues that allow "
                         "one GPU per job")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=MAX_OUTPUT_TOKENS,
                    help=f"clamped to engine.MAX_OUTPUT_TOKENS={MAX_OUTPUT_TOKENS}. A "
                         "finite cap also bounds the structured-output hang, vLLM "
                         "#14151, which otherwise burns the whole shard's walltime")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--enforce-eager", action="store_true",
                    help="skip CUDA graph capture to buy back GPU memory")
    ap.add_argument("--no-prefix-caching", dest="prefix_caching", action="store_false",
                    default=True,
                    help="disable prefix caching. Leave it on: the survey preamble is "
                         "byte-identical across all prompts and is 62%% of prompt "
                         "characters at the median, so caching it removes most of the "
                         "prefill on every request after the first")
    ap.add_argument("--quantization", default=None,
                    help="e.g. fp8. Off by default because bf16 is the configuration "
                         "this pass is measured in. The argument for fp8 is "
                         "concurrency, not arithmetic: a 27-31B model at bf16 leaves "
                         "room for only tens of concurrent sequences on a 150GB H200, "
                         "which under-fills the batch and drops prefill efficiency. "
                         "Not measured here")
    ap.add_argument("--kv-cache-dtype", default="auto",
                    help="e.g. fp8. Halves the per-sequence KV footprint, which is the "
                         "other half of the concurrency ceiling. Also not measured here")
    ap.add_argument("--logprobs", type=int, default=5,
                    help="top-k logprobs per token. A softmax over the assertion "
                         "token's logprobs is a calibrated per-finding probability; "
                         "asking the model how sure it is measures worse. 0 disables")
    ap.add_argument("--json-schema", action="store_true",
                    help="enforce the ReportExtraction schema with xgrammar. Off by "
                         "default: the measured gemma run returned 300/300 parseable "
                         "JSON without it. JSON schema and not regex -- gemma 4 31B "
                         "has an open vLLM bug (#39071) producing degenerate output "
                         "under guided regex")
    args = ap.parse_args()

    train_csv = _resolve_train_csv(args.train_csv)
    if train_csv is None:
        print("error: could not find train.csv; pass --train-csv", file=sys.stderr)
        return 2

    index, n_shards = parse_shard(args.shard) if args.shard else (None, None)
    base = args.out or (data_root() / "interim" / "extraction" / "records.jsonl")
    out_path = shard_path(base, index, n_shards)
    shard_label = f"{index}/{n_shards}" if index is not None else "all"

    rows = load_reports(train_csv)
    units = plan_units(rows)
    # De-duplication first, sharding second. The other order puts two studies
    # that share a report text into two different shards, where each is
    # extracted independently -- paying twice for one call and producing two
    # answers to the same question that a non-deterministic engine will not
    # agree on. Anchors are included: their held-out value was already spent by
    # the anchor validation run, and leaving them out would leave 58 training
    # studies with no derived labels.
    mine = units if index is None else _shard_of(units, index, n_shards)

    print(f"train.csv : {train_csv}")
    print(f"corpus    : {len(rows)} studies, {sum(r['is_anchor'] for r in rows)} anchors")
    print(f"plan      : {describe_units(units)}")
    print(f"shard     : {shard_label} — {describe_units(mine)}")
    print(f"out       : {out_path}")

    if out_path.exists() and not (args.resume or args.overwrite):
        print(f"error: {out_path} exists. Pass --resume to continue it, or "
              "--overwrite to start over; appending blindly would duplicate work.",
              file=sys.stderr)
        return 2
    if args.overwrite and out_path.exists():
        out_path.rename(out_path.with_suffix(out_path.suffix + ".superseded"))
        print("out       : previous file moved aside to "
              f"{out_path.name}.superseded (--overwrite)")

    done: frozenset[str] = frozenset()
    if args.resume:
        state = read_done(out_path)
        done = state.study_uids
        print(f"resume    : {len(done)} studies already written "
              f"from {state.n_lines} lines")
        for note in state.warnings:
            print(f"  warning : {note}")

    todo = pending(mine, done)
    if args.limit:
        todo = todo[: args.limit]
    n_studies = sum(len(unit.study_uids) for unit in todo)
    print(f"todo      : {len(todo)} calls covering {n_studies} studies "
          f"(~{len(todo) * SECONDS_PER_REPORT / 60:.0f} min at the measured "
          "0.81 s/report)")

    if args.dry_run:
        print("dry-run   : no model loaded, nothing written")
        return 0
    if not todo:
        print("nothing to do; shard is complete")
        return 0

    with_logprobs = args.logprobs > 0
    json_schema = ReportExtraction.model_json_schema() if args.json_schema else None

    run = init_run(
        f"extract-{shard_label.replace('/', 'of')}",
        job_type="extraction",
        config={"model": args.model, "backend": args.backend, "tp": args.tp,
                "seed": args.seed, "shard": shard_label, "n_calls": len(todo),
                "n_studies": n_studies, "batch_size": args.batch_size,
                "max_tokens": args.max_tokens, "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "enforce_eager": args.enforce_eager,
                "enable_prefix_caching": args.prefix_caching,
                "quantization": args.quantization,
                "kv_cache_dtype": args.kv_cache_dtype,
                "logprobs": args.logprobs, "json_schema": bool(json_schema),
                "temperature": 0.0},
        tags=["phase-01c", "extraction"],
    )

    backend = load_backend(
        args.backend, model=args.model, tp=args.tp, seed=args.seed,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=args.prefix_caching,
        quantization=args.quantization,
        kv_cache_dtype=args.kv_cache_dtype,
        logprobs=args.logprobs if with_logprobs else None,
        json_schema=json_schema,
    )

    prompts = [build_prompt(unit.report) for unit in todo]
    print(f"generating {len(prompts)} responses in batches of {args.batch_size}...")
    t0 = time.time()
    n_parsed = n_valid = n_truncated = 0

    # The writer appends and fsyncs per record; see RecordWriter for why flush
    # alone is not enough. Everything written before a kill survives it.
    with RecordWriter(out_path) as writer:
        if writer.repaired_tail:
            print(f"repaired  : {writer.repaired_tail}")
        for position, generation in generate_incremental(
            backend, prompts, max_tokens=args.max_tokens, batch_size=args.batch_size
        ):
            unit = todo[position]
            payload = build_payload(generation, unit.report, with_logprobs)
            n_parsed += bool(payload["_parsed"])
            n_valid += bool(payload["_valid"])
            n_truncated += bool(payload["_truncated"])
            for record in records_for_unit(unit, payload, shard=shard_label):
                writer.write(record)
            if (position + 1) % args.batch_size == 0 or position + 1 == len(prompts):
                so_far = time.time() - t0
                print(f"  {position + 1}/{len(prompts)} calls, "
                      f"{writer.n_written} records written, "
                      f"{so_far / 60:.1f} min "
                      f"({so_far / (position + 1):.2f} s/call)", flush=True)

        elapsed = time.time() - t0
        n_written = writer.n_written

    # Per-call, not per-report: de-duplication means these differ, and quoting a
    # per-report figure computed from calls would understate the cost of the
    # next run by however much duplication that corpus happens to have.
    print(f"generated : {len(prompts)} calls in {elapsed / 60:.1f} min "
          f"({elapsed / max(len(prompts), 1):.2f} s/call)")
    print(f"parsed    : {n_parsed}/{len(prompts)}")
    print(f"validated : {n_valid}/{len(prompts)}")
    print(f"truncated : {n_truncated}/{len(prompts)} hit the token cap")
    print(f"wrote     : {n_written} records to {out_path}")

    run.log({"n_calls": len(prompts), "n_records": n_written, "n_parsed": n_parsed,
             "parse_rate": n_parsed / max(len(prompts), 1), "n_valid": n_valid,
             "schema_valid_rate": n_valid / max(len(prompts), 1),
             "n_truncated": n_truncated,
             "seconds_per_call": elapsed / max(len(prompts), 1)})
    run.finish()

    # A shard that parsed nothing must exit nonzero, so that a dependent job (for
    # example a Slurm `afterok` chain) does not start on a shard with no data.
    if n_parsed == 0:
        print(f"FAILED: 0 of {len(prompts)} responses were parseable JSON. Read the "
              f"first `_raw` in {out_path.name} -- a reasoning preamble that consumed "
              "the token budget looks exactly like this.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
