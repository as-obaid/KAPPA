"""Batch text generation, with a swappable backend.

vLLM is the intended backend and `transformers` is a working fallback, because
the vLLM install is the most likely thing to fail on a cluster and a
reconnaissance pass over 300 short reports does not need its throughput.

Both backends run greedy: temperature 0 and a fixed seed. **That is not the
same as reproducible, and the difference was measured.** Two 300-report gemma
runs at identical model, seed, sample and temperature produced byte-identical
findings for only 197 of 300 reports, with 13 differing in finding count
(deltas -5 to +5, plus one +46 from a token-budget truncation). vLLM batches
requests together and the batch composition changes the floating-point
reduction order, so greedy decoding is only deterministic per fixed batch.

The practical consequence: **never cite a finding or span count from a run as
an exact figure.** Distributions and rates are stable to ~1.5pp across re-runs;
counts are not.

    from kappa.labels.engine import load_backend

    backend = load_backend("vllm", model="google/gemma-4-31B-it", tp=1)
    outputs = backend.generate(prompts, max_tokens=4096)

``generate_incremental`` is the durability boundary. ``generate()`` returns
nothing until every prompt is done, which is why a 59-minute reconnaissance run
killed at the 1-hour walltime wrote a zero-record file; the incremental form
yields each batch's results so the caller can flush them to disk as they land.

``generate_verbose`` returns the same text plus the two things the plain string
throws away: why generation stopped, and the model's own distribution over the
``assertion`` field. Verbalized confidence ("how sure are you?") is measured
elsewhere to be poorly calibrated; the logprobs under constrained decoding are
a different quantity, and persisting them costs one extra field per finding and
buys a soft target for a later pass. See ``assertion_positions``.

Three vLLM API facts this module is pinned to, all of which have dead tutorials
in circulation:

* ``guided_json`` / ``guided_regex`` / ``guided_choice`` / ``guided_grammar``
  were **removed in vLLM 0.12.0**. On 0.28.0 the only schema-enforcement API is
  ``StructuredOutputsParams`` / ``structured_outputs``. Any snippet passing
  ``guided_json=`` is not "old style", it is a ``TypeError``.
* The grammar backend is requested as ``"xgrammar:no-fallback"`` on purpose. Without
  the suffix a grammar that fails to compile falls back to *unconstrained*
  generation, which is the worst outcome available: the run completes, the JSON
  is mostly fine, and nothing anywhere says the schema stopped being enforced.
* **Speculative decoding is deliberately absent.** It crashes or silently
  breaks structured output, so there is no knob for it here to be turned on by
  someone reading a throughput guide.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
    "Backend",
    "Generation",
    "load_backend",
    "generate_incremental",
    "DEFAULT_BATCH_SIZE",
    "MAX_OUTPUT_TOKENS",
    "ASSERTION_LITERALS",
    "assertion_positions",
    "assertion_confidence",
    "align_assertion_confidence",
]

#: Hard ceiling on generated tokens, applied whatever the caller asks for.
#:
#: 4,096 is measured: at 3,072 one report of 300 was cut mid-JSON and lost all
#: 46 of its findings, and 4,096 fixed it against a corpus whose longest report
#: is 4,630 characters. The ceiling exists for a second reason though -- vLLM
#: issue #14151 is a hang under structured output, and a request with a finite
#: token budget terminates where an unbounded one runs to the walltime and takes
#: the whole shard with it.
MAX_OUTPUT_TOKENS = 4096

#: Prompts per call to the backend, i.e. how much work a walltime kill costs.
#:
#: Nothing is written to disk until a batch returns, so this is the loss window:
#: at the measured 0.81 s/report the default is roughly 1.7 minutes of exposure.
#: It is not free. Throughput is measured **non-linear** in batch size on this
#: node -- 5.65 s/report at n=5 against 0.81 s/report at n=300 -- so shrinking
#: the batch costs throughput. n=128 sits between the two measured points and
#: has **not** been measured here; pass ``--batch-size`` equal to the shard size
#: to reproduce the n=300-style regime exactly, at the price of losing the whole
#: shard to a kill. Prefix caching softens the trade, because the survey
#: preamble stays in the cache across batches rather than being re-prefilled.
DEFAULT_BATCH_SIZE = 128

#: The only three values ``assertion`` may take. Kept here rather than imported
#: from ``schema`` so that this module stays free of the validation layer.
ASSERTION_LITERALS = ("affirmed", "negated", "uncertain")

#: Matches the point in a JSON stream where the next token is the `assertion`
#: value. Anchored to the end because it is applied to a rolling buffer.
_ASSERTION_OPEN = re.compile(r'"assertion"\s*:\s*"$')

#: How much decoded text to keep while looking for the pattern above. The
#: longest thing it can match is short; 64 characters is generous.
_BUFFER_CHARS = 64


@dataclass(frozen=True)
class Generation:
    """One response, plus the metadata a bare string discards.

    ``finish_reason`` is load-bearing. A response truncated at the token cap is
    still a *string*, and it parses as JSON often enough to look fine; the only
    reliable signal that the model was cut off is ``"length"`` here. Recording
    it turns silent truncation into a counted failure.
    """

    text: str
    finish_reason: str | None = None
    n_output_tokens: int = 0
    #: Top-k logprob distributions at each ``assertion`` value position, in
    #: generation order. Empty when logprobs were not requested.
    assertion_logprobs: tuple[dict[str, float], ...] = field(default_factory=tuple)


def assertion_positions(
    tokens: Sequence[tuple[str, Mapping[str, float]]],
) -> list[dict[str, float]]:
    """Top-k distributions at every ``assertion`` value position, in order.

    ``tokens`` is ``(chosen token text, {candidate token text: logprob})`` per
    generated position, which is what both backends can produce and what a test
    can write by hand.

    The position is found by tracking the decoded JSON, not by looking for
    tokens that resemble an assertion value. That distinction is the whole
    point: ``"un"`` is a prefix of ``"uncertain"`` and also of half the English
    language, so prefix-matching the token stream would return a position for
    every ``"unremarkable"`` in a ``description`` field and quietly shift every
    finding's confidence onto the wrong finding. Anchoring on
    ``"assertion": "`` cannot do that.

    Candidate tokens are folded onto the literal they begin, and only the
    literals survive. Tokenisers split these words differently -- a leading
    ``aff``, ``neg`` or ``unc`` is common, sometimes with the opening quote
    attached -- so the raw token text is not comparable across models and the
    literal is.
    """
    out: list[dict[str, float]] = []
    buffer = ""
    expecting = False
    for chosen, dist in tokens:
        if expecting:
            out.append(_fold_to_literals(dist))
            expecting = False
        buffer = (buffer + chosen)[-_BUFFER_CHARS:]
        if _ASSERTION_OPEN.search(buffer):
            expecting = True
    return out


def _fold_to_literals(dist: Mapping[str, float]) -> dict[str, float]:
    """Best logprob per assertion literal, with an upper bound for the absent.

    A literal missing from the top-k is **not** dropped, and that is the whole
    point of this function rather than an implementation detail. Dropping it
    renormalises over whatever survived, so a top-5 that happens to contain only
    ``affirmed`` reports probability exactly 1.0 -- a hard-coded certainty that
    depends on the tokeniser rather than on the model. Every such finding would
    arrive downstream looking maximally confident.

    A token outside the top-k has a logprob no greater than the smallest one
    inside it, so the smallest observed value is a valid **upper** bound for a
    missing literal, and substituting it can only *understate* the confidence.
    Erring downward is the safe direction for a soft target.

    Non-finite logprobs are dropped first. Under structured output vLLM masks
    unreachable tokens to ``-inf``, and using ``-inf`` as the bound would put
    ``nan`` into the softmax when every candidate is masked.
    """
    finite = {token: float(lp) for token, lp in dist.items() if math.isfinite(float(lp))}
    folded: dict[str, float] = {}
    for token, logprob in finite.items():
        literal = _literal_for(token)
        if literal is None:
            continue
        if literal not in folded or logprob > folded[literal]:
            folded[literal] = logprob
    # No literal at all means the model was not answering this field; say so by
    # returning nothing rather than inventing a uniform distribution.
    if not folded:
        return {}
    bound = min(finite.values())
    for literal in ASSERTION_LITERALS:
        folded.setdefault(literal, bound)
    return folded


def _literal_for(token: str) -> str | None:
    """The assertion literal ``token`` begins, or None if it begins none or many."""
    cleaned = token.strip().lstrip('"').strip()
    if not cleaned:
        return None
    matches = [lit for lit in ASSERTION_LITERALS if lit.startswith(cleaned)]
    return matches[0] if len(matches) == 1 else None


def assertion_confidence(dist: Mapping[str, float]) -> tuple[str | None, float | None]:
    """``(most likely literal, its probability)`` from one folded distribution.

    The softmax is taken over the three assertion literals only, never over the
    full vocabulary. Renormalising is what makes the number a probability *of
    the decision that was actually made*: under structured output no other
    continuation is reachable, so mass sitting on unreachable tokens would
    depress every confidence by an amount that varies with the tokeniser.

    ``dist`` is expected to be the output of ``_fold_to_literals``, so a literal
    the top-k never showed is already present at its upper bound rather than
    absent -- see there for why that matters.
    """
    if not dist:
        return None, None
    best = max(dist, key=lambda literal: dist[literal])
    top = dist[best]
    total = sum(math.exp(value - top) for value in dist.values())
    return best, 1.0 / total


def align_assertion_confidence(
    n_findings: int, dists: Sequence[Mapping[str, float]]
) -> tuple[list[float | None], list[str]]:
    """Per-finding confidences, or nothing at all if the counts disagree.

    Refusing to align on a length mismatch is the guard that matters. Zipping a
    12-element finding list against 11 distributions produces 11 plausible
    numbers attached to the wrong findings, and there is no way to notice that
    downstream -- a confidence is not obviously misplaced the way a missing one
    is. So a mismatch yields ``None`` for every finding plus a warning that
    says both counts.
    """
    if n_findings and len(dists) != n_findings:
        return [None] * n_findings, [
            f"assertion logprobs misaligned: {n_findings} findings, "
            f"{len(dists)} assertion positions; confidences dropped"
        ]
    return [assertion_confidence(dist)[1] for dist in dists[:n_findings]], []


class Backend(Protocol):
    def generate(self, prompts: list[str], max_tokens: int = 2048) -> list[str]: ...

    def generate_verbose(
        self, prompts: list[str], max_tokens: int = 2048
    ) -> list[Generation]: ...


class VLLMBackend:
    """vLLM offline batch inference over a chat model."""

    def __init__(
        self,
        model: str,
        tp: int = 2,
        max_model_len: int = 8192,
        seed: int = 0,
        gpu_memory_utilization: float = 0.90,
        enforce_eager: bool = False,
        enable_prefix_caching: bool = True,
        quantization: str | None = None,
        kv_cache_dtype: str = "auto",
        logprobs: int | None = None,
        json_schema: dict | None = None,
        structured_outputs_backend: str = "xgrammar:no-fallback",
        disable_any_whitespace: bool = True,
    ):
        """``gpu_memory_utilization`` and ``enforce_eager`` exist because the
        Slurm QOS caps GPUs per job, which caps how far the weights can be
        sharded.

        The cluster this ran on allowed at most **2** GPUs per job, so `tp`
        cannot exceed 2 here no matter how many H200s the node holds. The
        122B at bf16 is 250 GB, i.e. 125 GB on each of two 150 GB cards, and the
        default 0.90 utilisation budget (135 GB) leaves only ~10 GB for KV cache,
        activations and CUDA graphs. Raising utilisation and skipping graph
        capture is what makes that fit; `enforce_eager` costs throughput and buys
        back the graph memory, which is the right trade for a few hundred
        prompts.

        ``enable_prefix_caching`` is on by default because it is free here and
        large. Every prompt in this pass begins with the byte-identical survey
        preamble: measured over the 4,262 distinct reports, the preamble is
        1,688 characters against a median prompt of 2,702, i.e. **62% of prompt
        characters at the median and 60% at the mean**. Prefix caching removes
        the prefill for that shared span on every request after the first.
        Characters are not tokens and the non-Latin scripts tokenise worse than
        the ASCII preamble, so the token share is somewhat lower than 60% for
        Greek and Cyrillic reports -- treat 60% as the Latin-report figure.

        ``quantization`` and ``kv_cache_dtype`` default to off, because bf16 is
        the configuration this pass is measured working in and switching weights
        to FP8 on an untested guess is how a measured pipeline stops being
        measured. When they are turned on, the argument for them is concurrency
        rather than arithmetic: a 27-31B model at bf16 spends ~60 GB of a 150 GB
        card on weights, and whatever is left after activations and CUDA graphs
        divided by the per-sequence KV footprint is the ceiling on how many
        sequences vLLM can run at once. Below the batch size that ceiling
        under-fills the batch, and an under-filled batch loses prefill
        efficiency -- which is the cost, not the FLOPs saved by the narrower
        multiply. FP8 weights roughly halve the weight term and an FP8 KV cache
        roughly halves the per-sequence term, so the ceiling moves twice.
        **None of that is measured on this cluster.**

        ``json_schema`` turns on structured output. It is off by default because
        the measured gemma run returned 300/300 parseable JSON without it, so
        the schema buys insurance rather than a fix. When it is on the grammar
        backend is requested with the ``:no-fallback`` suffix so a compile error
        raises instead of degrading to unconstrained generation, and
        ``disable_any_whitespace`` stops the model spending the token budget on
        indentation. Both are passed as engine configuration, and if this vLLM
        build spells that key differently the constructor raises here -- which
        is the intended behaviour, because the alternative is a run that quietly
        enforced nothing.
        """
        from vllm import LLM, SamplingParams

        self._seed = seed
        self._logprobs = logprobs
        self._json_schema = json_schema
        self._SamplingParams = SamplingParams
        engine_kwargs: dict[str, Any] = {
            "model": model,
            "tensor_parallel_size": tp,
            "max_model_len": max_model_len,
            "seed": seed,
            "trust_remote_code": True,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "enable_prefix_caching": enable_prefix_caching,
            "kv_cache_dtype": kv_cache_dtype,
        }
        if quantization:
            engine_kwargs["quantization"] = quantization
        if json_schema is not None:
            engine_kwargs["structured_outputs_config"] = {
                "backend": structured_outputs_backend,
                "disable_any_whitespace": disable_any_whitespace,
            }
        self.llm = LLM(**engine_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def _chat(self, prompt: str) -> str:
        """Render the chat template, with reasoning traces off where supported.

        ``enable_thinking=False`` is why this is not a one-liner. Qwen3.5 ships
        a chat template that emits a reasoning preamble by default, and on the
        122B-A10B-FP8 smoke test that preamble consumed the **entire** 4,096
        token budget: all 5 responses opened with ``Thinking Process:\n\n1.
        **Analyze the Request:**`` and were still mid-reasoning when they hit
        the cap, so 0 of 5 returned any JSON at all. The job exited 0 and the
        aggregate reported "0 returned parseable JSON" -- a full GPU-hour
        producing nothing, and it would have repeated on the 300- and
        4,407-report runs.

        Raising ``--max-tokens`` is the wrong fix: the survey wants the JSON,
        not the reasoning, and paying for thousands of prose tokens per report
        at 73 s/report is what made this arm ~90x slower than gemma.

        Passed as a keyword because templates that do not declare it ignore
        unknown variables -- gemma renders identically with or without it -- but
        the call is still guarded, because a template that *validates* its
        arguments would otherwise take the whole run down.
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def _params(self, max_tokens: int) -> Any:
        # seed=self._seed, not 0: --seed was silently ignored here, so the flag
        # affected only the engine seed and the sample draw.
        capped = min(max_tokens, MAX_OUTPUT_TOKENS)
        if capped < max_tokens:
            print(
                f"note: max_tokens {max_tokens} clamped to the hard cap "
                f"{MAX_OUTPUT_TOKENS}; see engine.MAX_OUTPUT_TOKENS"
            )
        kwargs: dict[str, Any] = {
            "temperature": 0.0,
            "max_tokens": capped,
            "seed": self._seed,
        }
        if self._logprobs:
            kwargs["logprobs"] = self._logprobs
        if self._json_schema is not None:
            from vllm.sampling_params import StructuredOutputsParams

            kwargs["structured_outputs"] = StructuredOutputsParams(
                json=self._json_schema
            )
        return self._SamplingParams(**kwargs)

    def _ordered(self, prompts: list[str], max_tokens: int) -> list[Any]:
        outs = self.llm.generate([self._chat(p) for p in prompts], self._params(max_tokens))
        # vLLM does not guarantee input order in its results, so re-key on the
        # request id rather than trusting the list order. The id is a decimal
        # string today; `int()` on a build that prefixes it (`"cmpl-0"`) would
        # raise *after* the GPU work is done, which is the most expensive
        # possible moment to fail, so an unparseable id falls back to a string
        # sort of that group instead.
        return sorted(outs, key=_request_order)

    def generate(self, prompts: list[str], max_tokens: int = 2048) -> list[str]:
        return [o.outputs[0].text for o in self._ordered(prompts, max_tokens)]

    def generate_verbose(
        self, prompts: list[str], max_tokens: int = 2048
    ) -> list[Generation]:
        """Text plus finish reason, token count and the assertion distributions.

        Requesting ``logprobs`` allocates a top-k dictionary per generated
        token, so the response objects are an order of magnitude larger than the
        strings. Everything except the assertion positions is dropped here, in
        this loop, rather than carried into the record -- a full top-5 trace over
        4,262 reports is a multi-gigabyte artifact that answers no question
        anybody has.
        """
        out: list[Generation] = []
        for result in self._ordered(prompts, max_tokens):
            completion = result.outputs[0]
            out.append(
                Generation(
                    text=completion.text,
                    finish_reason=getattr(completion, "finish_reason", None),
                    n_output_tokens=len(getattr(completion, "token_ids", ()) or ()),
                    assertion_logprobs=tuple(
                        assertion_positions(_token_stream(completion))
                    ),
                )
            )
        return out


def _request_order(output: Any) -> tuple[int, int, str]:
    """Sort key over vLLM request ids that tolerates a non-numeric id."""
    rid = str(getattr(output, "request_id", ""))
    try:
        return (0, int(rid), "")
    except ValueError:
        return (1, 0, rid)


def _token_stream(completion: Any) -> list[tuple[str, Mapping[str, float]]]:
    """``(chosen token text, {token text: logprob})`` per generated position.

    vLLM returns ``logprobs`` as a list parallel to ``token_ids``, each entry a
    dict keyed by token id. Converting to token *text* here is what lets
    ``assertion_positions`` stay a pure function with hand-written test input,
    and it is also the only form that is comparable across tokenisers.
    """
    logprobs = getattr(completion, "logprobs", None)
    token_ids = getattr(completion, "token_ids", None)
    if not logprobs or not token_ids:
        return []
    stream: list[tuple[str, Mapping[str, float]]] = []
    for token_id, candidates in zip(token_ids, logprobs):
        if not candidates:
            continue
        dist = {
            (getattr(value, "decoded_token", None) or ""): float(value.logprob)
            for value in candidates.values()
        }
        chosen = candidates.get(token_id)
        stream.append((getattr(chosen, "decoded_token", None) or "", dist))
    return stream


class TransformersBackend:
    """Fallback path. Slower, but adequate for a few hundred short reports."""

    def __init__(self, model: str, seed: int = 0, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(seed)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=getattr(torch, dtype), device_map="auto"
        )

    def generate(self, prompts: list[str], max_tokens: int = 2048) -> list[str]:
        out = []
        for prompt in prompts:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            ids = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(
                    **ids, max_new_tokens=min(max_tokens, MAX_OUTPUT_TOKENS),
                    do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
                )
            out.append(self.tokenizer.decode(gen[0][ids.input_ids.shape[1]:],
                                             skip_special_tokens=True))
        return out

    def generate_verbose(
        self, prompts: list[str], max_tokens: int = 2048
    ) -> list[Generation]:
        """Same text, and explicitly no logprobs.

        This path returns empty ``assertion_logprobs`` rather than approximating
        them. A fallback that fabricated confidences would make the two backends
        look interchangeable in a field where they are not, and the caller's
        alignment check cannot distinguish "no logprobs" from "wrong logprobs".
        """
        return [Generation(text=text) for text in self.generate(prompts, max_tokens)]


def generate_incremental(
    backend: Backend,
    prompts: Sequence[str],
    *,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[tuple[int, Generation]]:
    """Yield ``(prompt index, Generation)`` as each batch finishes, not at the end.

    This exists because of exactly one incident. The reconnaissance script
    called ``generate()`` once over the whole sample and wrote ``records.jsonl``
    after it returned, so the file did not exist until the last report was done
    -- and a Slurm walltime ``SIGKILL`` at minute 59 of a 59-minute run left an
    empty output directory. Streaming the results out in batches is what makes
    the loss proportional to ``batch_size`` instead of total.

    Yielding rather than returning a list is load-bearing: the caller writes and
    ``fsync``s each record as it arrives, so the records already on disk survive
    the kill. A version of this function that built the full list first would
    have the same bug with more machinery.

    The index is the position in ``prompts``, so the caller can pair the result
    with its work unit without relying on the backend preserving order.
    """
    if batch_size < 1:
        raise ValueError(f"batch size must be >= 1, got {batch_size}")
    for start in range(0, len(prompts), batch_size):
        chunk = list(prompts[start : start + batch_size])
        for offset, generation in enumerate(
            backend.generate_verbose(chunk, max_tokens=max_tokens)
        ):
            yield start + offset, generation


def load_backend(
    kind: str,
    model: str,
    tp: int = 2,
    seed: int = 0,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.90,
    enforce_eager: bool = False,
    enable_prefix_caching: bool = True,
    quantization: str | None = None,
    kv_cache_dtype: str = "auto",
    logprobs: int | None = None,
    json_schema: dict | None = None,
) -> Backend:
    """Build a backend, defaulting to offline mode for compute nodes."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if kind == "vllm":
        return VLLMBackend(
            model=model,
            tp=tp,
            seed=seed,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            enable_prefix_caching=enable_prefix_caching,
            quantization=quantization,
            kv_cache_dtype=kv_cache_dtype,
            logprobs=logprobs,
            json_schema=json_schema,
        )
    if kind == "transformers":
        return TransformersBackend(model=model, seed=seed)
    raise ValueError(f"unknown backend {kind!r}; expected 'vllm' or 'transformers'")
