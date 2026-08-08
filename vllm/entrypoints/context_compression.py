# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
ACE v2 — input-layer context compression: retrieval, not heuristics.

WHERE THIS RUNS, AND WHY NOT IN THE KV BLOCK MANAGER
====================================================
This is a *request-layer* compressor. It rewrites the chat history of an
incoming ``/v1/chat/completions`` request before it is templated and prefilled.
It does not touch the KV cache, the block manager, or the scheduler.

That placement is deliberate. Scoring KV blocks by relevance was investigated
and does not work in vLLM's design:

  * ``get_new_blocks()`` draws victims from a free queue that only ever holds
    blocks with ``ref_cnt == 0``. Nothing alive references them, so there is no
    live query to score them against.
  * ``KVCacheBlock`` carries no tokens — only a one-way hash. There is no text
    to run a relevance function over.
  * Prefix hashes are *chained*. Evicting a block from the middle of a sequence
    silently breaks the correctness of every hash after it.
  * There is no masking surface for a hole in the middle of an attended
    context.

None of those blockers apply here. The text is in the request. The live query
is the last user message. The cache is untouched — we simply send fewer tokens.
And there is no hole in the middle of a sequence; there is a shorter request.

WHAT IT SCORES WITH, AND WHAT WAS MEASURED
==========================================
Measured on LoCoMo (200 questions, the benchmark's own F1, a real model
answering, same ~8% budget). Higher is better:

    full context (ceiling) ..... 22.56   (22,198 tokens)
    tail truncation ............  6.62
    hand-written heuristics ....  6.07   <-- BELOW the do-nothing floor
    BM25 without IDF ........... 20.34
    BM25 ....................... 23.59
    embeddings ................. 23.95
    weighted-sum hybrid ........ 26.36
    reciprocal RANK fusion ..... 28.09   (1,884 tokens)

Two things that table says, and this file is built around:

  * RETRIEVING WELL BEATS HAVING EVERYTHING. 28.09 against the 22.56 of the
    full context, on 8% of the tokens. Irrelevant context is not neutral — it
    distracts. Compressing well is an improvement, not a lesser evil.

  * Turning the heuristics on costs **-3.02 F1**, and dropping IDF costs
    **-3.25 F1**. Both are refuted, not merely unhelpful. The retired
    implementation lives in ``context_compression_refuted.py`` and is not on
    any serving path.

DESIGN RULES (each one is a measured result, not a preference)
=============================================================
1. No content heuristics. Not one rule about which words matter. If a line
   matters, its relevance to the live question says so.

2. No hand-written stoplist. IDF computed over *this* conversation does that
   job by construction: whatever appears everywhere gets ~zero weight, with no
   dictionary and without knowing the language. A hand list ages; endogenous
   IDF adapts per conversation.

3. Real BM25 — frequency saturation (k1) and length normalisation (b). Summing
   IDF and dividing by a constant rewards long lines and never saturates, which
   are precisely the two failures BM25 exists to correct.

4. Scored against the LIVE QUESTION (the last user message), not the opening
   task. What is relevant changes every turn.

5. Knobs derived from MEASURED PRESSURE, not constants. ``pressure`` is
   ``budget / history_tokens``; the recent-turn reserve, the MMR lambda and the
   candidate pool all follow from it and from the redundancy actually observed
   in the material. A fixed lambda is the same disease as a heuristic, one
   floor up.

6. Recency is a SWITCH, not a weight. Measured across a budget sweep it helps
   +8.6 under pressure and hurts -7.9 with slack, and scaling the weight
   changes nothing (0.048 and 0.15 decide identically) — as an addend it is
   binary. So it is implemented as two strata separated by 10: a line with any
   relevance can never fall below a line without it, not even after the MMR
   penalty (which is bounded by 0.8), and recency only breaks ties *among the
   irrelevant*, and only when ``pressure < 0.25``.

7. Real budget contract. The sum of per-line costs is NOT what gets emitted:
   per-message template overhead and the omission markers are missing from it,
   which is how an earlier version emitted 1.6-1.8x the requested budget. So
   the emitted output is measured for real and the worst-scoring lines are
   handed back until it genuinely fits. Token counting uses the server's own
   tokenizer — ``len(text)/4`` badly underestimates code, and unlike a browser
   we have the real tokenizer right here.

8. Marked gaps. Wherever something is omitted a visible marker is left. A model
   that knows something is missing asks; a model that believes it saw
   everything invents.

9. STALENESS IS CORRECTNESS, NOT EFFICIENCY. An agent reads a file at turn 3
   and edits it at turn 20. BM25 retrieves *both* versions and the old one
   scores just as high, because it shares its whole vocabulary with the new
   one. The model then sees content that is no longer true, with no signal
   that it is stale — and no amount of relevance fixes a fact being false.
   Measured with a read -> edit -> re-read probe over 8 seeds: the superseded
   version survived 8/8 without this, 0/8 with it, while the current version
   still survived 8/8. No recall cost. Superseded results are DEMOTED to the
   filler stratum, never deleted: "this is no longer true" and "this never
   existed" are different claims and only the first one is true. See
   ``_mark_superseded``.

10. THE BUDGET IS A CEILING, NOT A QUOTA. Filling it to the brim with lines
    that share not one word with the live question spent 38% of a 3,000-token
    budget and 56% of a 16,000-token one on pure filler. That is not free:
    this file's own headline says retrieving well BEATS the full context
    (28.09 vs 22.56), so irrelevant text is not neutral ballast, it distracts.
    The elastic window therefore stops when the evidence runs out, not when
    the tokens do. It is a TRADE, not a gift — see ``AceCompressionConfig.
    elastic`` for the measured table.

11. HEAD RESERVE, UNCONDITIONAL. The opening messages are kept verbatim and
    unscored. The start of a session carries the ASSIGNMENT — the task, the
    paths, the constraints — which the rest of the session takes as read, so
    nobody repeats it, so there is no lexical overlap for BM25 to find. It is
    also the one thing an agent cannot reconstruct by looking at the code.
    Measured with a probe asking for the original assignment mid-session:
    0/5 without the reserve, 5/5 with it, at a tight budget. Capped at 5%.
    External backing: the first tokens act as attention SINKS and absorb
    45-55% of the mass — Xiao et al., "Efficient Streaming Language Models
    with Attention Sinks", ICLR 2024, arXiv:2309.17453.

12. THE TAIL RESERVE IS A FLOOR, NOT A CEILING IN DISGUISE. Stopping at the
    first message that does not fit turned a "38% reserved for recent turns"
    promise into 3-5% delivered, because ONE large tool result in the
    second-to-last position blocked everything before it. While the floor
    (10% of the budget) has not been reached, a message that does not fit is
    now truncated through the middle instead of being discarded.

13. WHERE A LINE WAS IS PART OF WHAT IT IS. Dedup used to strip the line-number
    gutter before hashing, so ``12→return 42;`` and ``87→return 42;`` collapsed
    into one and the model was handed one site out of three. "Where is X
    defined?" is the commonest question an agent of code asks and the answer IS
    the number, so an incomplete answer that looks complete is the worst shape
    it can take. Whitespace and the gutter's *style* are still noise; the
    position is not. See ``dedup_key``.
    Unlike rules 9-12 this one has NO measured gain behind it — on the
    retrieval bench it moves nothing (85.7% and 89.3%, identical either way),
    because those probes never ask "where". It is justified by the mechanism,
    and that is said rather than dressed up with a number.

DEFAULT PATH AND THE SEMANTIC HALF
==================================
The default serving path is lexical BM25 (23.59 measured, already above the
22.56 full-context ceiling). The rank-fusion result (28.09) needs a second,
embedding model, which the entrypoint layer does not have: the served model is
generative. ``rrf_fuse`` and the ``embed`` hook are implemented and tested so
that fusion is available wherever an embedder exists, but nothing wires one by
default. This is stated rather than hidden because the headline number belongs
to the fused configuration, not to what ships enabled here.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import regex as re

from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "AceCompressionConfig",
    "AceCompressionStats",
    "TokenCounter",
    "BM25Index",
    "compress_chat_history",
    "maybe_compress_chat_messages",
    "rrf_fuse",
    "terms",
]

# Visible omission markers. Rule 8: the model must be able to see that
# something was removed, otherwise it fills the hole with invention.
LINES_OMITTED = "[... {n} lines omitted by ACE ...]"
MESSAGES_OMITTED = "[... {n} earlier messages omitted by ACE ...]"

# Relevance stratum offset. Any line with non-zero query relevance is lifted by
# this much, so the MMR diversity penalty (bounded by MMR_LAMBDA_MAX < 1) can
# reorder *within* a stratum but can never promote an irrelevant line above a
# relevant one. Rule 6.
_RELEVANT_STRATUM = 10.0
_MMR_LAMBDA_MAX = 0.8
_MMR_LAMBDA_MIN = 0.15

# First line of a message carries provenance ("Tool result for X", a file
# header). Floored above the irrelevant stratum, below the relevant one.
_PROVENANCE_FLOOR = 0.5

# Smallest tail slice worth emitting. Below this a truncated message is all
# marker and no content, so it is dropped into the compressible pool instead.
_MIN_TAIL_ROOM = 40


# ---------------------------------------------------------------------------
# Terms
# ---------------------------------------------------------------------------

# No stoplist (rule 2). Two-character tokens are admitted on purpose: short
# identifiers exist in code (`fs`, `db`, `id`) and are sometimes exactly what is
# being asked about. The leading class is any Unicode letter or underscore
# rather than [a-z_], so the tokenizer is not silently English-only — which
# matters because the IDF that replaces the stoplist is computed from these
# tokens and has to work in whatever language the conversation is in.
_TERM_RE = re.compile(r"[^\W\d][\w./-]+|\d{2,}", re.UNICODE)
_SIM_RE = re.compile(r"[\w./-]{2,}", re.UNICODE)

# Line-number gutter, in the shapes agent tools emit it: "  42→foo" (a Read
# tool) and "42:foo" / "42: foo" (grep -n, with and without the space). The
# NUMBER is captured, not discarded — see ``dedup_key``. The colon does not
# require a following space: real ``grep -n`` output has none, and demanding
# one meant the commonest shape of all was not recognised as a gutter.
_GUTTER_RE = re.compile(r"^\s*(\d+)\s*[:→]")
_WS_RE = re.compile(r"\s+")


def terms(text: str) -> list[str]:
    """Content terms of a string, lowercased. No stoplist by design."""
    return _TERM_RE.findall(text.lower())


def sim_tokens(text: str) -> frozenset[str]:
    """Token set used for redundancy/diversity comparisons."""
    return frozenset(_SIM_RE.findall(text.lower()))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def dedup_key(line: str) -> str:
    """Normalised key for near-exact duplicate detection.

    Whitespace is noise and is normalised away. The line NUMBER is not: it is
    part of the line's identity.

    This used to strip the gutter unconditionally, which collapsed the output
    of ``grep -n``: ``10:  return 42;`` and ``87:  return 42;`` produced the
    same key and only one survived. But "where is X defined?" is the commonest
    question an agent of code asks, and the answer IS the number — handing back
    one site out of three is an incomplete answer that looks complete.

    So only lines that are equal in WHERE THEY WERE as well as in what they say
    collapse: real repetition, the ``}`` or the ``[tool result]`` header coming
    out twenty times. The gutter's *style* is still noise, so the same line seen
    once through a Read (``42→``) and once through a grep (``42:``) still
    collapses — it is the same line of the same file.

    Justified by the mechanism, not by a measured gain: on the retrieval bench
    it moves nothing (85.7% and 89.3%, identical either way), because those
    probes never ask "where".
    """
    match = _GUTTER_RE.match(line)
    if match is None:
        return _WS_RE.sub(" ", line).strip().lower()
    body = _WS_RE.sub(" ", line[match.end() :]).strip().lower()
    return f"{match.group(1)}|{body}"


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


# ---------------------------------------------------------------------------
# BM25 with endogenous IDF
# ---------------------------------------------------------------------------


class BM25Index:
    """
    BM25 over the lines of the history being packed.

    The corpus is the history itself, so IDF is endogenous: ubiquitous tokens
    ("the", "const", formatting punctuation, whatever the boilerplate of this
    particular conversation happens to be) collapse to ~zero weight without
    anybody listing them. Removing IDF was measured at -3.25 F1, which is why
    the document frequencies must come from a real corpus of many lines rather
    than from the handful of lines inside one message.
    """

    def __init__(
        self,
        docs: Sequence[str],
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.n = max(len(docs), 1)
        self._tf: list[dict[str, int]] = []
        self._dl: list[int] = []
        df: dict[str, int] = {}
        total_len = 0
        for doc in docs:
            tf: dict[str, int] = {}
            for t in terms(doc):
                tf[t] = tf.get(t, 0) + 1
            dl = sum(tf.values())
            self._tf.append(tf)
            self._dl.append(dl)
            total_len += dl
            for t in tf:
                df[t] = df.get(t, 0) + 1
        self._df = df
        self.avgdl = (total_len / self.n) or 1.0

    def idf(self, term: str) -> float:
        """Robertson IDF with the usual floor, so a term present in more than
        half the corpus contributes ~nothing instead of contributing
        negatively."""
        n_t = self._df.get(term, 0)
        return max(math.log(1.0 + (self.n - n_t + 0.5) / (n_t + 0.5)), 1e-6)

    def score(self, doc_idx: int, query_terms: Iterable[str]) -> float:
        tf = self._tf[doc_idx]
        if not tf:
            return 0.0
        dl = self._dl[doc_idx]
        k1, b = self.k1, self.b
        norm = k1 * (1.0 - b + b * dl / self.avgdl)
        total = 0.0
        for q in query_terms:
            f = tf.get(q)
            if not f:
                continue
            total += self.idf(q) * (f * (k1 + 1.0)) / (f + norm)
        return total

    def score_all(self, query_terms: Sequence[str]) -> list[float]:
        """Score every document at once.

        Intersecting the query term set with each line's term set keeps the
        inner loop proportional to the line rather than to the query, which is
        what makes a long history affordable to index on every turn.
        """
        scores = [0.0] * len(self._tf)
        if not query_terms:
            return scores
        qset = set(query_terms)
        idf = {q: self.idf(q) for q in qset}
        k1, b, avgdl = self.k1, self.b, self.avgdl
        for i, tf in enumerate(self._tf):
            if not tf:
                continue
            hits = qset.intersection(tf)
            if not hits:
                continue
            norm = k1 * (1.0 - b + b * self._dl[i] / avgdl)
            total = 0.0
            for q in hits:
                f = tf[q]
                total += idf[q] * (f * (k1 + 1.0)) / (f + norm)
            scores[i] = total
        return scores

    def rank_query_terms(self, query_terms: Sequence[str], limit: int) -> list[str]:
        """Keep the ``limit`` most discriminative query terms.

        A cost bound, not a stoplist: the ordering is IDF, computed from this
        very history, and the terms it drops are the ones whose IDF is near
        zero — so they were contributing near zero to the score anyway. It is
        the same mechanism that replaces the hand-written stoplist (rule 2),
        reused to keep a pathologically long "question" from making indexing
        quadratic.
        """
        if limit <= 0 or len(query_terms) <= limit:
            return list(query_terms)
        return sorted(query_terms, key=self.idf, reverse=True)[:limit]


def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    """
    Reciprocal rank fusion.

    Input: several id lists, each ordered most- to least-relevant.
    Output: id -> fused score.

    POSITIONS are fused, not scores, so there is nothing to calibrate between
    an unbounded BM25 score and a cosine in [-1, 1] — a classic source of
    fragility. Measured better than the weighted sum (28.09 vs 26.36), and the
    reason is that lexical and semantic retrieval do different jobs: split by
    vocabulary overlap between question and answer, embeddings win where there
    is none (24.28 vs 18.60) and BM25 wins where there is (29.33 vs 23.58).
    Fusion keeps both.
    """
    out: dict[int, float] = {}
    for ranking in rankings:
        for rank, ident in enumerate(ranking):
            out[ident] = out.get(ident, 0.0) + 1.0 / (k + rank + 1)
    return out


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------


class TokenCounter:
    """
    Counts tokens with the server's real tokenizer.

    Rule 7. ``len(text)/4`` is a prose heuristic; on code and tool output it
    underestimates badly, and an underestimated budget is how a compressor ends
    up emitting more than it promised. At the entrypoint layer the real
    tokenizer is already in hand — this is the advantage the browser-side
    implementation did not have.

    Results are memoised per text: the same line is scored, costed, re-costed
    during the budget-contract loop and finally emitted, and agent histories
    repeat lines constantly.
    """

    # Per-message overhead of the chat template (role tags, turn delimiters).
    # Used when it cannot be measured from the tokenizer.
    DEFAULT_MESSAGE_OVERHEAD = 4

    def __init__(self, tokenizer: Any | None = None, cache_size: int = 200_000):
        self._tokenizer = tokenizer
        self._cache: dict[str, int] = {}
        self._cache_size = cache_size
        self._message_overhead: int | None = None
        self.exact = tokenizer is not None

    def _encode_len(self, text: str) -> int:
        tk = self._tokenizer
        if tk is None:
            # Fallback only. Kept deliberately pessimistic (max of a
            # piece-count estimate and chars/4) because underestimating the
            # cost is the failure mode that breaks the budget contract.
            pieces = len(re.findall(r"\w+|[^\w\s]", text))
            return max(math.ceil(pieces * 1.25), math.ceil(len(text) / 4))
        try:
            return len(tk.encode(text, add_special_tokens=False))
        except TypeError:
            return len(tk.encode(text))

    def __call__(self, text: str) -> int:
        if not text:
            return 0
        hit = self._cache.get(text)
        if hit is not None:
            return hit
        value = self._encode_len(text)
        if len(self._cache) < self._cache_size:
            self._cache[text] = value
        return value

    @property
    def message_overhead(self) -> int:
        """Tokens a chat template spends per message, above its content.

        Measured once from the tokenizer's own template rather than guessed:
        a hardcoded constant is exactly the kind of guess that broke the budget
        contract before. Taken as the *difference* between a two-message and a
        one-message rendering, so the template preamble (default system block,
        BOS) cancels out and what is left is the marginal cost of one more
        message.
        """
        if self._message_overhead is not None:
            return self._message_overhead
        overhead = self.DEFAULT_MESSAGE_OVERHEAD
        tk = self._tokenizer
        if tk is not None:
            try:
                probe = "x"
                msg = {"role": "user", "content": probe}
                one = tk.apply_chat_template(
                    [msg], tokenize=False, add_generation_prompt=False
                )
                two = tk.apply_chat_template(
                    [msg, msg], tokenize=False, add_generation_prompt=False
                )
                measured = self(two) - self(one) - self(probe)
                if 0 < measured <= 64:
                    overhead = measured
            except Exception:
                logger.debug(
                    "ACE: could not measure chat-template message overhead; "
                    "falling back to %d tokens per message.",
                    overhead,
                    exc_info=True,
                )
        self._message_overhead = overhead
        return overhead


def truncate_to_tokens(text: str, max_tokens: int, count: TokenCounter) -> str:
    """Shrink one oversized line to fit, keeping head and tail and marking the
    cut. Verified against the counter rather than assumed, because the whole
    point of rule 7 is that estimates are not contracts."""
    if max_tokens <= 0:
        return LINES_OMITTED.format(n=1)
    if count(text) <= max_tokens:
        return text
    total = count(text)
    keep = max(20, int(len(text) * (max_tokens / max(total, 1)) * 0.5))
    for _ in range(8):
        head, tail = text[:keep], text[-keep:]
        cut = len(text) - len(head) - len(tail)
        candidate = f"{head} [... {cut} chars cut ...] {tail}"
        if count(candidate) <= max_tokens:
            return candidate
        keep = int(keep * 0.6)
        if keep < 8:
            break
    return LINES_OMITTED.format(n=1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AceCompressionConfig:
    """Knobs. The ones that matter are derived, not set — see ``_autotune``."""

    # Retrieval
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    rrf_k: int = 60

    # Rule 9. Demote tool results a later call on the same target overwrote.
    # Measured on a read -> edit -> re-read probe, 8 seeds: the superseded
    # version survived 8/8 with this off and 0/8 with it on, while the current
    # version survived 8/8 either way. Correctness, not efficiency.
    supersede: bool = True

    # Rule 10. Stop when the evidence runs out instead of when the tokens do.
    # Measured (8 seeds, presence of the fact — NOT answer quality):
    #
    #     budget    tokens saved    cost in facts
    #      3,000         -1%            +-0.0     <- product regime, free
    #      8,000          2%            +-0.0
    #     16,000         26%            -1.8
    #     32,000         60%            -8.9
    #
    # A TRADE, not a gift, and the bench only sees one side of it: it measures
    # whether the fact is present, not whether the answer is better. The case
    # in favour is this file's own headline (retrieving well beats the full
    # context, so filler subtracts), but until that is run with a model
    # answering, the -8.9 at 32,000 is a real number and the gain is a
    # conjecture. Said out loud rather than hidden. Turning this off restores
    # fill-to-the-brim behaviour and 98.5-99.9% budget utilisation.
    elastic: bool = True

    # Rule 11. Reserve for the FIRST messages, kept verbatim and unscored.
    # Unconditional: it used to be gated off when the budget was roomy because
    # "the assignment survives on its own there" — true only because we were
    # filling to the brim. With the elastic window it fell from 100% to 0%.
    # Its survival was an accident of the filler, not a property of having
    # room. 5% of the budget, measured not to cost anything.
    head_frac: float = 0.05

    # Rule 12. Floor for the recent window: below it, a message that does not
    # fit is truncated rather than dropped.
    tail_min_frac: float = 0.10

    # Recent window kept verbatim
    recent_turns: int = 6
    recent_frac: float | None = None  # None -> derived from pressure
    auto_min_recent_frac: float = 0.35
    auto_max_recent_frac: float = 0.80
    # Below this pressure, document order beats novelty, so the recency
    # tie-break is switched OFF. Measured neutral at 0.17 and harmful at 0.34.
    recency_pressure_threshold: float = 0.25

    # Redundancy
    dedup: bool = True
    mmr: bool = True
    mmr_lambda: float | None = None  # None -> measured from the material
    mmr_candidates: int | None = None  # None -> derived from the budget
    redundancy_sample: int = 240

    # Pinning
    pin_query_terms: bool = True
    pin_max: int = 40

    # Emission
    per_line_cap_frac: float = 0.5
    max_message_chars: int = 12_000
    # Roles never scored, never dropped: they carry the task, the security
    # constraints and the behavioural guardrails.
    protected_roles: tuple[str, ...] = ("system", "developer")

    # Semantic half (unwired by default; see module docstring)
    semantic_blocks: int = 400

    # Cost bound on the live question, applied in IDF order (see
    # ``BM25Index.rank_query_terms``). Not a stoplist.
    max_query_terms: int = 256

    # Safety valve on the budget-contract loop.
    max_fit_iterations: int = 24

    def __post_init__(self) -> None:
        if self.recent_turns < 1:
            raise ValueError("recent_turns must be >= 1")
        if not 0.0 < self.per_line_cap_frac <= 1.0:
            raise ValueError("per_line_cap_frac must be in (0, 1]")


@dataclass
class AceCompressionStats:
    """What the compressor actually did. Emitted as logs and metrics — without
    it nobody can tell whether it is working."""

    input_tokens: int = 0
    output_tokens: int = 0
    budget_tokens: int = 0
    total_lines: int = 0
    kept_lines: int = 0
    pinned_lines: int = 0
    deduped_lines: int = 0
    collapsed_messages: int = 0
    dropped_messages: int = 0
    # Rule 9/11/12: how much of the output the three unscored mechanisms
    # account for. Without these nobody can tell a stale demotion from a
    # retrieval miss.
    superseded_messages: int = 0
    head_messages: int = 0
    tail_truncated_messages: int = 0
    pressure: float = 1.0
    mmr_lambda: float = 0.0
    recency_tiebreak: bool = False
    over_budget: bool = False
    exact_tokens: bool = False
    mode: str = "lexical"

    @property
    def ratio(self) -> float:
        """Output tokens per input token. Lower is more compression."""
        if self.input_tokens <= 0:
            return 1.0
        return self.output_tokens / self.input_tokens


# ---------------------------------------------------------------------------
# Autotuning
# ---------------------------------------------------------------------------


@dataclass
class _Tuned:
    pressure: float
    recent_frac: float
    recency_tiebreak: bool


def _autotune(
    history_tokens: int, budget_tokens: int, cfg: AceCompressionConfig
) -> _Tuned:
    """
    Derive the knobs from a measurable property of *this* history.

    ``pressure`` is the fraction of the history that fits: 1 means it all fits,
    ->0 means severe squeeze. It matters most in long context, which is the
    real case: a 200k session against a 32k budget has nothing in common with a
    5k session against 3k, and one constant cannot serve both.
    """
    pressure = budget_tokens / max(history_tokens, 1)
    pressure = min(1.0, max(0.0, pressure))

    # Recent-turn reserve. With a roomy budget, keeping the last turns verbatim
    # is cheap. Under squeeze the reserve has to shrink so there is room to
    # SEARCH: if the recent turns eat the budget there is no space left to
    # bring back the line from thirty turns ago that is the one being asked
    # about.
    if cfg.recent_frac is not None:
        recent_frac = cfg.recent_frac
    else:
        recent_frac = (
            cfg.auto_min_recent_frac
            + (cfg.auto_max_recent_frac - cfg.auto_min_recent_frac) * pressure
        )

    # Rule 6. A switch, not a weight, and the sweep says where it goes.
    recency_tiebreak = pressure < cfg.recency_pressure_threshold
    return _Tuned(pressure, recent_frac, recency_tiebreak)


def _measure_redundancy(lines: Sequence[str], sample: int) -> float:
    """
    Measure the redundancy actually present, instead of assuming a lambda.

    A history of near-identical tool results needs a hard diversity penalty; a
    conversation where every line is different needs almost none, and there a
    high lambda only destroys good information.
    """
    n = len(lines)
    if n < 4:
        return 0.0
    step = max(1, n // sample)
    picked = [sim_tokens(lines[i]) for i in range(0, n, step)]
    total = 0.0
    pairs = 0
    for i in range(len(picked)):
        for j in range(i + 1, min(i + 8, len(picked))):
            total += jaccard(picked[i], picked[j])
            pairs += 1
    return total / pairs if pairs else 0.0


# ---------------------------------------------------------------------------
# Message plumbing
# ---------------------------------------------------------------------------


@dataclass
class _Line:
    msg_slot: int  # index into the compressible-message list
    line_idx: int
    text: str
    is_first: bool
    bm25: float = 0.0
    semantic: float = 0.0
    recency: float = 0.0
    score: float = 0.0
    duplicate: bool = False
    pinned: bool = False
    # Rule 9: this line belongs to a tool result that a later call on the same
    # target has overwritten. Demoted to the filler stratum, never deleted.
    stale: bool = False

    @property
    def ident(self) -> int:
        return self.msg_slot * 1_000_000 + self.line_idx


def _content_of(message: Any) -> str | None:
    """Return the plain-text content of a message, or None when it is not
    plain text (multimodal part lists, ``None``): those are passed through
    untouched because a line-level compressor has nothing to say about them."""
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    return content if isinstance(content, str) else None


def _role_of(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "")
    return str(getattr(message, "role", "") or "")


def _with_content(message: Any, content: str) -> Any:
    """Copy a message replacing only its content, preserving every other field
    (``tool_call_id``, ``tool_calls``, ``name``): dropping those breaks the
    call/result pairing that chat templates and tool parsers rely on."""
    if isinstance(message, dict):
        return {**message, "content": content}
    copied = message.model_copy(update={"content": content})  # pydantic
    return copied


def _is_structural(message: Any) -> bool:
    """True when removing the message entirely would break the conversation's
    structure — a ``tool`` result needs its assistant call and vice versa. Such
    messages are collapsed to a marker, never dropped."""
    role = _role_of(message)
    if role == "tool":
        return True
    if role == "assistant":
        if isinstance(message, dict):
            return bool(message.get("tool_calls"))
        return bool(getattr(message, "tool_calls", None))
    return False


def _tool_calls_of(message: Any) -> Sequence[Any]:
    if isinstance(message, dict):
        calls = message.get("tool_calls")
    else:
        calls = getattr(message, "tool_calls", None)
    return calls or ()


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


# Argument names that identify WHAT a call acted on. This is the vLLM-side
# equivalent of "name + mtime" in a directory watcher: the timestamp is the
# turn, and this is the name. It is a structural list (where a tool says which
# object it touched), not a content heuristic — nothing here looks at the words
# inside a result, which is what rule 1 forbids.
_TARGET_ARG_KEYS = (
    "path",
    "file_path",
    "filepath",
    "file",
    "filename",
    "notebook_path",
    "url",
    "uri",
    "command",
    "cmd",
)


def _call_target(call: Any) -> str | None:
    """The object a tool call acted on, or None when it cannot be told.

    Keyed on the TARGET, not on ``(tool name, target)``. That is deliberate and
    it is what the measured engine does: a write to a path has to supersede an
    earlier read of the same path, and keying on the tool name would put them
    in different buckets and lose exactly the case rule 9 exists for. When no
    location-style argument is present the whole call is used as its own key,
    so an identical call repeated still supersedes itself and an unrecognised
    tool is never falsely paired with a different one.
    """
    fn = _field(call, "function")
    if fn is None:
        return None
    name = _field(fn, "name")
    raw_args = _field(fn, "arguments")
    parsed: Any = None
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except (ValueError, TypeError):
            parsed = None
    elif isinstance(raw_args, dict):
        parsed = raw_args
    if isinstance(parsed, dict):
        for key in _TARGET_ARG_KEYS:
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(raw_args, str) and raw_args.strip():
        blob = raw_args
    elif parsed is not None:
        try:
            blob = json.dumps(parsed, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return None
    else:
        return None
    return f"{name}({_WS_RE.sub(' ', blob).strip()})"


def _mark_superseded(messages: Sequence[Any]) -> set[int]:
    """Indices of ``tool`` results a later call on the same target overwrote.

    Rule 9. The target is NOT in the result — the result only says which tool
    ran. It comes from the assistant call, and in the OpenAI schema the pairing
    is explicit: ``tool_calls[].id`` on the assistant message, ``tool_call_id``
    on the result. That is stronger than the browser engine's "the message
    right before", which cannot survive parallel calls; adjacency is kept only
    as a fallback for clients that omit the id, and only when the preceding
    assistant message made exactly one call, so an ambiguous pairing is never
    guessed.
    """
    target_by_call_id: dict[str, str] = {}
    adjacent: tuple[int, str] | None = None
    target_of_result: dict[int, str] = {}
    newest_for: dict[str, int] = {}

    for i, message in enumerate(messages):
        calls = _tool_calls_of(message)
        if calls:
            for call in calls:
                target = _call_target(call)
                call_id = _field(call, "id")
                if target is not None and call_id is not None:
                    target_by_call_id[str(call_id)] = target
            adjacent = None
            if len(calls) == 1:
                only = _call_target(calls[0])
                if only is not None:
                    adjacent = (i + 1, only)
            continue
        if _role_of(message) != "tool":
            adjacent = None
            continue
        call_id = _field(message, "tool_call_id")
        target = target_by_call_id.get(str(call_id)) if call_id is not None else None
        if target is None and adjacent is not None and adjacent[0] == i:
            target = adjacent[1]
        adjacent = None
        if target is None:
            continue
        target_of_result[i] = target
        newest_for[target] = i

    return {i for i, target in target_of_result.items() if newest_for[target] != i}


def _clamp_message(content: str, max_chars: int) -> str:
    """Bound a single pathological message before it is line-split, so one
    multi-megabyte tool result cannot make indexing quadratic."""
    if len(content) <= max_chars:
        return content
    head = int(max_chars * 0.7)
    tail = max(0, max_chars - head - 40)
    cut = len(content) - head - tail
    return (
        content[:head]
        + f"\n... [{cut} characters clamped by ACE] ...\n"
        + (content[-tail:] if tail else "")
    )


# ---------------------------------------------------------------------------
# The compressor
# ---------------------------------------------------------------------------


def compress_chat_history(
    messages: Sequence[Any],
    budget_tokens: int,
    *,
    tokenizer: Any | None = None,
    count_tokens: TokenCounter | None = None,
    config: AceCompressionConfig | None = None,
    embed: Callable[[list[str]], Sequence[Sequence[float]]] | None = None,
) -> tuple[list[Any], AceCompressionStats]:
    """
    Pack a chat history into ``budget_tokens``, keeping what the live question
    needs.

    Returns ``(messages, stats)``. The input list is never mutated. When the
    history already fits, the original messages are returned unchanged (same
    objects), so a no-op compression is byte-identical by construction.
    """
    cfg = config or AceCompressionConfig()
    count = count_tokens or TokenCounter(tokenizer)
    stats = AceCompressionStats(budget_tokens=budget_tokens, exact_tokens=count.exact)
    overhead = count.message_overhead

    original = list(messages)
    if not original:
        return original, stats

    def cost_of(message: Any) -> int:
        content = _content_of(message)
        if content is None:
            # Non-text content (image parts and friends). We cannot line-split
            # it and we cannot price it at this layer; charge the overhead so
            # the accounting stays monotonic and say so in the stats.
            return overhead
        return count(content) + overhead

    stats.input_tokens = sum(cost_of(m) for m in original)
    if stats.input_tokens <= budget_tokens:
        stats.output_tokens = stats.input_tokens
        return original, stats

    # --- partition -------------------------------------------------------
    # Protected roles and non-text messages are emitted verbatim wherever they
    # are. Everything else is a candidate for line-level compression.
    n = len(original)
    verbatim: set[int] = set()
    for i, message in enumerate(original):
        if _role_of(message) in cfg.protected_roles or _content_of(message) is None:
            verbatim.add(i)

    history_tokens = stats.input_tokens
    tuned = _autotune(history_tokens, budget_tokens, cfg)
    stats.pressure = tuned.pressure
    stats.recency_tiebreak = tuned.recency_tiebreak

    # The most recent turns are kept literally, but only while they fit inside
    # their reserve. A fixed recent-turn count can eat the whole budget on its
    # own (one page of tool output is easily 1.5k tokens) and return several
    # times what was asked for.
    #
    # Rule 12: the reserve is a FLOOR as well as a ceiling. Stopping at the
    # first message that does not fit turned a 38% reserve into 3-5% delivered,
    # because one large tool result in the second-to-last position blocked
    # everything before it. Below the floor, a message that does not fit is
    # truncated through the middle instead of being discarded — the model needs
    # the last turns to know where it is, and that cannot depend on the turn
    # before having been bulky.
    reserve = int(budget_tokens * tuned.recent_frac)
    floor_tokens = int(budget_tokens * cfg.tail_min_frac)
    # Content substituted for a verbatim message, index -> text. Only the tail
    # floor writes here; everything else in the verbatim set is emitted as the
    # very object that came in.
    overrides: dict[int, str] = {}
    used = 0
    recent: set[int] = set()
    taken = 0
    for i in range(n - 1, -1, -1):
        if taken >= cfg.recent_turns:
            break
        if i in verbatim:
            continue
        c = cost_of(original[i])
        newest = not recent
        # The newest turn keeps its old escape hatch: it may exceed the reserve
        # as long as it fits the budget, because a model that cannot see the
        # turn it is answering is worse off than one short of history.
        if (used + c <= reserve or newest) and used + c <= budget_tokens:
            recent.add(i)
            used += c
            taken += 1
            continue
        if newest:
            # Even the newest turn does not fit the budget. Letting it through
            # truncated would hand the whole request to head/tail cutting;
            # line-level retrieval over it is strictly better, so drop it into
            # the compressible pool and keep the marked gaps.
            break
        if used >= floor_tokens:
            # Floor already met: back to the old rule. What is left competes
            # for the rest of the budget on relevance, which is where an old
            # message can beat a bulky recent one.
            break
        if not any(j not in verbatim and j not in recent for j in range(i)):
            # Nothing older to protect. The floor exists to stop one big
            # message from blocking the turns BEFORE it; with no turns before
            # it there is nothing being blocked.
            break
        content = _content_of(original[i])
        if content is None:
            break
        room = min(reserve, floor_tokens) - used - overhead
        if room < _MIN_TAIL_ROOM:
            break
        truncated = truncate_to_tokens(content, room, count)
        recent.add(i)
        overrides[i] = truncated
        used += count(truncated) + overhead
        taken += 1
        if used >= floor_tokens:
            break
    verbatim |= recent
    stats.tail_truncated_messages = len(overrides)

    def verbatim_cost(i: int) -> int:
        override = overrides.get(i)
        if override is None:
            return cost_of(original[i])
        return count(override) + overhead

    used = sum(verbatim_cost(i) for i in verbatim)

    # --- head reserve (rule 11) ------------------------------------------
    # The FIRST messages are kept verbatim and unscored, exactly like the last
    # ones. Not aesthetic symmetry: the opening carries the assignment, which
    # the rest of the session takes as read and therefore never repeats, so
    # there is no lexical overlap for BM25 to score it on. Unconditional and
    # hard-capped so it can never compete with the search.
    head: list[int] = []
    head_reserve = int(budget_tokens * cfg.head_frac)
    head_used = 0
    for i in range(n):
        if i in verbatim:
            continue
        c = cost_of(original[i])
        if head_used + c > head_reserve:
            break
        head.append(i)
        head_used += c
    verbatim |= set(head)
    used += head_used
    stats.head_messages = len(head)

    compressible = [i for i in range(n) if i not in verbatim]
    if not compressible:
        stats.output_tokens = used
        stats.over_budget = used > budget_tokens
        if stats.over_budget:
            logger.warning(
                "ACE: verbatim messages alone need %d tokens, above the %d "
                "token budget; nothing left to compress.",
                used,
                budget_tokens,
            )
        if not overrides:
            return original, stats
        out = [
            _with_content(m, overrides[i]) if i in overrides else m
            for i, m in enumerate(original)
        ]
        stats.output_tokens = sum(cost_of(m) for m in out)
        stats.over_budget = stats.output_tokens > budget_tokens
        return out, stats

    # --- live question ---------------------------------------------------
    # The last user message. Not the opening task (rule 4). Tool results carry
    # role="tool" in the OpenAI schema, so no content sniffing is needed to
    # tell them apart — which is the point: no rule looks at the words.
    query = ""
    for message in reversed(original):
        if _role_of(message) == "user":
            content = _content_of(message)
            if content:
                query = content
                break
    if not query:
        for message in reversed(original):
            content = _content_of(message)
            if content:
                query = content
                break

    # --- staleness (rule 9) ----------------------------------------------
    # Computed over the WHOLE history, not just the compressible slice: the
    # call that supersedes an old result is usually the newest one, which lives
    # in the verbatim recent window.
    superseded = _mark_superseded(original) if cfg.supersede else set()
    stats.superseded_messages = len(superseded)

    # --- index -----------------------------------------------------------
    lines: list[_Line] = []
    slot_of_index: dict[int, int] = {}
    clamped: list[list[str]] = []
    for slot, idx in enumerate(compressible):
        slot_of_index[idx] = slot
        content = _clamp_message(
            _content_of(original[idx]) or "", cfg.max_message_chars
        )
        parts = content.split("\n")
        clamped.append(parts)
        stale = idx in superseded
        for li, text in enumerate(parts):
            lines.append(_Line(slot, li, text, li == 0, stale=stale))

    stats.total_lines = len(lines)
    index = BM25Index([line.text for line in lines], k1=cfg.bm25_k1, b=cfg.bm25_b)
    query_terms = index.rank_query_terms(
        list(dict.fromkeys(terms(query))), cfg.max_query_terms
    )
    n_slots = max(len(compressible) - 1, 1)
    bm25_scores = index.score_all(query_terms)
    for i, line in enumerate(lines):
        line.bm25 = bm25_scores[i]
        line.recency = line.msg_slot / n_slots

    mode = "lexical"
    fused: dict[int, float] | None = None
    if embed is not None:
        fused = _semantic_fusion(lines, query, cfg, embed)
        if fused is not None:
            mode = "hybrid"
    stats.mode = mode

    # --- score (rule 6: two strata, 10 apart) ----------------------------
    # Rule 9 rides on the same two strata: a superseded line is DEMOTED to the
    # filler stratum, not deleted. It comes back only if there is nothing
    # better to say — which is the honest rendering of "this is no longer
    # true" as opposed to "this never existed".
    if fused is not None:
        max_fused = max(fused.values()) if fused else 1.0
        for line in lines:
            relevant = (line.bm25 > 0.0 or line.semantic > 0.0) and not line.stale
            if relevant:
                line.score = _RELEVANT_STRATUM + (
                    fused.get(line.ident, 0.0) / (max_fused or 1.0)
                )
            else:
                line.score = line.recency if tuned.recency_tiebreak else 0.0
    else:
        max_bm = max((line.bm25 for line in lines), default=0.0) or 1.0
        for line in lines:
            if line.bm25 > 0.0 and not line.stale:
                line.score = _RELEVANT_STRATUM + line.bm25 / max_bm
            else:
                line.score = line.recency if tuned.recency_tiebreak else 0.0
    for line in lines:
        if line.is_first:
            line.score = max(line.score, _PROVENANCE_FLOOR)

    # --- dedup -----------------------------------------------------------
    # Dedup sees the HEAD reserve too: content already travelling verbatim up
    # there must not be paid for a second time down here. Reserving room and
    # then repeating the same text would spend the same budget twice.
    if cfg.dedup:
        best: dict[str, _Line] = {}
        head_sentinel = _Line(-1, -1, "", False, score=math.inf)
        for idx in head:
            for text in (_content_of(original[idx]) or "").split("\n"):
                key = dedup_key(text)
                if key:
                    best.setdefault(key, head_sentinel)
        for line in lines:
            if line.is_first:
                continue
            key = dedup_key(line.text)
            if not key:
                continue
            prev = best.get(key)
            if prev is None:
                best[key] = line
                continue
            loser, winner = (prev, line) if line.score > prev.score else (line, prev)
            loser.duplicate = True
            stats.deduped_lines += 1
            best[key] = winner

    # --- selection -------------------------------------------------------
    per_line_cap = max(40, int((budget_tokens - used) * cfg.per_line_cap_frac))
    keep: set[int] = set()

    def line_cost(line: _Line) -> int:
        return min(count(line.text), per_line_cap) + 1

    # What the question names explicitly is never evicted (bounded by pin_max
    # so it cannot eat the budget on its own).
    #
    # The pin MUST honour staleness. The superseded version of a file carries
    # the same identifiers as the current one, so a pin that only looked at
    # ``bm25 > 0`` pinned the stale copy by itself and cancelled the demotion
    # entirely — the bug that made rule 9 look like it did nothing. That the
    # question names something does not make it true.
    if cfg.pin_query_terms:
        hits = sorted(
            (ln for ln in lines if ln.bm25 > 0.0 and not ln.duplicate and not ln.stale),
            key=lambda ln: -ln.score,
        )[: cfg.pin_max]
        for line in hits:
            c = line_cost(line)
            if used + c > budget_tokens:
                break
            keep.add(line.ident)
            line.pinned = True
            used += c
            stats.pinned_lines += 1

    pool = [ln for ln in lines if not ln.duplicate and ln.ident not in keep]
    pool.sort(key=lambda ln: -ln.score)

    # Lambda measured, not assumed; candidate pool scaled to what fits, because
    # a fixed candidate count leaves most of a long history unexamined.
    if cfg.mmr_lambda is not None:
        lam = cfg.mmr_lambda
    else:
        lam = max(
            _MMR_LAMBDA_MIN,
            min(
                _MMR_LAMBDA_MAX,
                2.0
                * _measure_redundancy([ln.text for ln in pool], cfg.redundancy_sample),
            ),
        )
    stats.mmr_lambda = lam
    n_cand = (
        cfg.mmr_candidates
        if cfg.mmr_candidates is not None
        else max(400, min(8000, round(budget_tokens / 6)))
    )

    if cfg.mmr and lam > 0.0 and pool:
        used = _mmr_select(
            pool[:n_cand],
            lines,
            keep,
            used,
            budget_tokens,
            lam,
            line_cost,
            elastic=cfg.elastic,
        )

    # --- rule 10: the elastic window --------------------------------------
    # The pool is sorted by score and the two strata are 10 apart, so the first
    # line below the relevant stratum is where the evidence ends: stop there
    # instead of spending what is left on text with no word in common with the
    # question. Plain score cut-off, the oldest idea in retrieval. The head and
    # tail reserves are untouched by this — that is what makes them reserves.
    for line in pool:
        if line.ident in keep:
            continue
        if cfg.elastic and line.score < _RELEVANT_STRATUM:
            break
        c = line_cost(line)
        if used + c > budget_tokens:
            continue
        keep.add(line.ident)
        used += c
        if used >= budget_tokens:
            break

    # --- rule 7: the budget contract -------------------------------------
    # Per-line costs ignore per-message overhead and the omission markers, so
    # what has been "spent" so far is a guess. Measure what actually gets
    # emitted and hand back the worst-scoring lines until it genuinely fits.
    def render(current_keep: set[int]) -> tuple[list[Any], AceCompressionStats]:
        return _emit(
            original,
            compressible,
            slot_of_index,
            clamped,
            current_keep,
            per_line_cap,
            count,
            cfg,
            overrides,
        )

    out, emit_stats = render(keep)
    realized = sum(cost_of(m) for m in out)
    if realized > budget_tokens:
        givable = sorted(
            (ln for ln in lines if ln.ident in keep and not ln.pinned),
            key=lambda ln: ln.score,
        )
        cursor = 0
        for _ in range(cfg.max_fit_iterations):
            if realized <= budget_tokens or cursor >= len(givable):
                break
            over = realized - budget_tokens
            freed = 0
            while cursor < len(givable) and freed < over:
                line = givable[cursor]
                cursor += 1
                keep.discard(line.ident)
                freed += min(count(line.text), per_line_cap)
            out, emit_stats = render(keep)
            realized = sum(cost_of(m) for m in out)

    # The give-back overshoots, and by a lot: handing back the last line of a
    # message also removes that message's template overhead and its omission
    # markers, so one line returned can free ten tokens. Measured on an agent
    # history it landed at ~54% of the granted budget. Unused budget is unused
    # recall — the point of the measured result is that a well-spent budget
    # beats the full context — so refill with the best lines that still fit,
    # verifying against the emitted size exactly as the give-back does.
    #
    # This loop is where rule 10 bites hardest, and it exists only in the
    # server-side engine: it is the refill that the browser engine never had.
    # Gating the two selection loops and leaving this one open would have
    # produced a byte-identical result, because everything the elastic window
    # rejected upstream would walk straight back in here. With the window on,
    # under-spending the budget is the intended outcome; over-spending is still
    # forbidden, and the give-back above still enforces that.
    refillable = sorted(
        (
            ln
            for ln in lines
            if ln.ident not in keep
            and not ln.duplicate
            and not (cfg.elastic and ln.score < _RELEVANT_STRATUM)
        ),
        key=lambda ln: -ln.score,
    )
    cursor = 0
    for _ in range(cfg.max_fit_iterations):
        spare = budget_tokens - realized
        if spare <= 0 or cursor >= len(refillable):
            break
        added: list[int] = []
        spend = 0
        while cursor < len(refillable):
            line = refillable[cursor]
            c = min(count(line.text), per_line_cap) + 1
            if spend + c > spare:
                break
            cursor += 1
            keep.add(line.ident)
            added.append(line.ident)
            spend += c
        if not added:
            # The next best line does not fit; nothing smaller is worth
            # reordering the ranking for.
            break
        candidate_out, candidate_stats = render(keep)
        candidate_realized = sum(cost_of(m) for m in candidate_out)
        if candidate_realized > budget_tokens:
            for ident in added:
                keep.discard(ident)
            break
        out, emit_stats, realized = candidate_out, candidate_stats, candidate_realized

    stats.kept_lines = len(keep)
    stats.collapsed_messages = emit_stats.collapsed_messages
    stats.dropped_messages = emit_stats.dropped_messages
    stats.output_tokens = realized
    stats.over_budget = realized > budget_tokens
    if stats.over_budget:
        # Honest failure: the structural floor (protected roles, the recent
        # window, and one marker per tool-paired message) is above the budget.
        logger.warning(
            "ACE: could not reach the %d token budget; emitted %d tokens "
            "(structural floor: protected roles + recent turns + tool pairing).",
            budget_tokens,
            realized,
        )
    return out, stats


def _mmr_select(
    candidates: list[_Line],
    all_lines: list[_Line],
    keep: set[int],
    used: int,
    budget_tokens: int,
    lam: float,
    line_cost: Callable[[_Line], int],
    *,
    elastic: bool = True,
) -> int:
    """Maximal marginal relevance over the candidate pool.

    The diversity penalty is bounded by ``lam <= 0.8``, well below the stratum
    gap of 10, so it reorders within a relevance stratum and never promotes an
    irrelevant line past a relevant one (rule 6).

    The elastic window (rule 10) applies HERE as well, not only to the final
    fill. MMR orders for diversity *within* what is relevant; it is not a side
    door for filler. Gating only the fill loop measured identical, because the
    irrelevant material had already come in through this one.
    """
    tokens = {ln.ident: sim_tokens(ln.text) for ln in candidates}
    max_sim = {ln.ident: 0.0 for ln in candidates}
    for line in all_lines:
        if line.ident in keep:
            ks = sim_tokens(line.text)
            for cand in candidates:
                max_sim[cand.ident] = max(
                    max_sim[cand.ident], jaccard(tokens[cand.ident], ks)
                )
    remaining = {ln.ident for ln in candidates}
    by_id = {ln.ident: ln for ln in candidates}
    while remaining:
        best_id = max(remaining, key=lambda i: by_id[i].score - lam * max_sim[i])
        remaining.discard(best_id)
        line = by_id[best_id]
        if elastic and line.score < _RELEVANT_STRATUM:
            continue
        c = line_cost(line)
        if used + c > budget_tokens:
            continue
        keep.add(best_id)
        used += c
        chosen = tokens[best_id]
        for i in remaining:
            max_sim[i] = max(max_sim[i], jaccard(tokens[i], chosen))
        if used >= budget_tokens:
            break
    return used


def _semantic_fusion(
    lines: list[_Line],
    query: str,
    cfg: AceCompressionConfig,
    embed: Callable[[list[str]], Sequence[Sequence[float]]],
) -> dict[int, float] | None:
    """
    Score lines semantically in blocks and fuse the two rankings by position.

    Encoding line by line is prohibitive and the signal survives chunking, so
    the block SIZE follows from an encoding budget rather than the other way
    round: the number of encoder calls is bounded and blocks grow with the
    history. Falls back to lexical-only (returns None) on any failure — no
    embedder must ever be able to take the request down.
    """
    if not lines:
        return None
    block_size = max(1, math.ceil(len(lines) / max(1, cfg.semantic_blocks)))
    blocks: list[tuple[int, str]] = []
    for start in range(0, len(lines), block_size):
        chunk = lines[start : start + block_size]
        blocks.append((start, "\n".join(ln.text for ln in chunk)[:4000]))
    try:
        vectors = embed([query[:2000]] + [text for _, text in blocks])
        query_vec = vectors[0]
        for (start, _), vec in zip(blocks, vectors[1:]):
            sim = cosine(query_vec, vec)
            for line in lines[start : start + block_size]:
                line.semantic = sim
    except Exception:
        logger.warning(
            "ACE: embedding step failed; falling back to the lexical path.",
            exc_info=True,
        )
        return None

    by_lexical = [ln.ident for ln in sorted(lines, key=lambda ln: -ln.bm25)]
    by_semantic = [ln.ident for ln in sorted(lines, key=lambda ln: -ln.semantic)]
    return rrf_fuse([by_lexical, by_semantic], cfg.rrf_k)


def _emit(
    original: list[Any],
    compressible: list[int],
    slot_of_index: dict[int, int],
    clamped: list[list[str]],
    keep: set[int],
    per_line_cap: int,
    count: TokenCounter,
    cfg: AceCompressionConfig,
    overrides: dict[int, str] | None = None,
) -> tuple[list[Any], AceCompressionStats]:
    """Build the outgoing message list, with every omission marked (rule 8).

    ``overrides`` carries the tail-floor rewrites (rule 12): verbatim messages
    whose content was truncated through the middle so the recent window could
    reach its floor instead of stopping at the first message that did not fit.
    """
    stats = AceCompressionStats()
    rendered: dict[int, str | None] = {}
    for slot, idx in enumerate(compressible):
        parts = clamped[slot]
        out_lines: list[str] = []
        skipped = 0
        for li, text in enumerate(parts):
            if (slot * 1_000_000 + li) in keep:
                if skipped:
                    out_lines.append(LINES_OMITTED.format(n=skipped))
                    skipped = 0
                out_lines.append(truncate_to_tokens(text, per_line_cap, count))
            else:
                skipped += 1
        if skipped and out_lines:
            out_lines.append(LINES_OMITTED.format(n=skipped))
        if out_lines:
            rendered[idx] = "\n".join(out_lines)
        elif _is_structural(original[idx]):
            # Never dropped: a tool result without its call (or the reverse)
            # breaks the chat template and every tool parser downstream.
            rendered[idx] = LINES_OMITTED.format(n=len(parts))
            stats.collapsed_messages += 1
        else:
            rendered[idx] = None
            stats.dropped_messages += 1

    out: list[Any] = []
    notice_pending = stats.dropped_messages
    rewrites = overrides or {}
    for i, message in enumerate(original):
        if i not in slot_of_index:
            rewritten = rewrites.get(i)
            out.append(
                message if rewritten is None else _with_content(message, rewritten)
            )
            continue
        content = rendered.get(i)
        if content is None:
            continue
        if notice_pending:
            # The "N messages omitted" notice rides as an extra line inside the
            # first surviving compressed message instead of becoming a message
            # of its own: inventing a message would risk breaking strict
            # role-alternation templates and tool-call pairing.
            content = MESSAGES_OMITTED.format(n=notice_pending) + "\n" + content
            notice_pending = 0
        out.append(_with_content(message, content))

    if notice_pending:
        # Nothing in the compressed region survived. Attach the notice to the
        # first surviving plain-text message that is not a protected role: the
        # system/developer turn carries the task and the guardrails and must
        # come out of here exactly as it went in.
        for i, message in enumerate(out):
            if _role_of(message) in cfg.protected_roles:
                continue
            content = _content_of(message)
            if content is not None:
                out[i] = _with_content(
                    message,
                    MESSAGES_OMITTED.format(n=notice_pending) + "\n" + content,
                )
                notice_pending = 0
                break
    if notice_pending:
        logger.warning(
            "ACE: %d messages were omitted but no message survived that could "
            "carry the marker; the gap is unmarked.",
            notice_pending,
        )
    return out, stats


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class _AceMetrics:
    requests: Any
    input_tokens: Any
    output_tokens: Any
    ratio: Any
    over_budget: Any


_metrics: _AceMetrics | None = None
_metrics_failed = False


def _get_metrics() -> _AceMetrics | None:
    """Lazily register the Prometheus family. Never fatal: a metrics problem
    must not take down a request."""
    global _metrics, _metrics_failed
    if _metrics is not None or _metrics_failed:
        return _metrics
    try:
        from prometheus_client import Counter, Histogram

        from vllm.v1.metrics.prometheus import get_prometheus_registry

        registry = get_prometheus_registry()
        _metrics = _AceMetrics(
            requests=Counter(
                name="vllm:ace_context_compression_requests_total",
                documentation="Chat requests compressed by ACE.",
                registry=registry,
            ),
            input_tokens=Counter(
                name="vllm:ace_context_compression_input_tokens_total",
                documentation="Prompt tokens seen by ACE before compression.",
                registry=registry,
            ),
            output_tokens=Counter(
                name="vllm:ace_context_compression_output_tokens_total",
                documentation="Prompt tokens emitted by ACE after compression.",
                registry=registry,
            ),
            ratio=Histogram(
                name="vllm:ace_context_compression_ratio",
                documentation=("Output prompt tokens divided by input prompt tokens."),
                buckets=[0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
                registry=registry,
            ),
            over_budget=Counter(
                name="vllm:ace_context_compression_over_budget_total",
                documentation=(
                    "Compressions that could not reach the requested budget "
                    "because the structural floor was already above it."
                ),
                registry=registry,
            ),
        )
    except Exception:
        _metrics_failed = True
        logger.debug("ACE: Prometheus metrics unavailable.", exc_info=True)
    return _metrics


def record_compression_metrics(stats: AceCompressionStats) -> None:
    """Publish before/after token counts and the ratio. Rule: if it is not
    measured, nobody knows whether it is working."""
    logger.info(
        "ACE context compression: %d -> %d prompt tokens (ratio %.3f, "
        "budget %d, pressure %.3f, mode %s, kept %d/%d lines, "
        "pinned %d, deduped %d, collapsed %d, dropped %d msgs, "
        "superseded %d msgs, head %d msgs, tail-truncated %d msgs, "
        "lambda %.2f, recency_tiebreak %s, exact_tokens %s)",
        stats.input_tokens,
        stats.output_tokens,
        stats.ratio,
        stats.budget_tokens,
        stats.pressure,
        stats.mode,
        stats.kept_lines,
        stats.total_lines,
        stats.pinned_lines,
        stats.deduped_lines,
        stats.collapsed_messages,
        stats.dropped_messages,
        stats.superseded_messages,
        stats.head_messages,
        stats.tail_truncated_messages,
        stats.mmr_lambda,
        stats.recency_tiebreak,
        stats.exact_tokens,
    )
    metrics = _get_metrics()
    if metrics is None:
        return
    try:
        metrics.requests.inc()
        metrics.input_tokens.inc(stats.input_tokens)
        metrics.output_tokens.inc(stats.output_tokens)
        metrics.ratio.observe(stats.ratio)
        if stats.over_budget:
            metrics.over_budget.inc()
    except Exception:
        logger.debug("ACE: failed to record metrics.", exc_info=True)


# ---------------------------------------------------------------------------
# Serving entry point
# ---------------------------------------------------------------------------


def resolve_budget_tokens(
    max_model_len: int,
    reserved_output_tokens: int | None,
    budget_tokens: int | None = None,
) -> int:
    """Token budget for the compressed prompt.

    Default: whatever the model can hold minus the output reservation, i.e.
    compress only enough to make the request fit. An operator who wants the
    measured quality win (compressing well beat the full context, 28.09 vs
    22.56, on 8% of the tokens) sets an explicit, smaller budget.
    """
    if budget_tokens is not None and budget_tokens > 0:
        return budget_tokens
    reserved = reserved_output_tokens or max(256, max_model_len // 8)
    return max(256, max_model_len - reserved)


def maybe_compress_chat_messages(
    messages: Sequence[Any],
    *,
    enabled: bool,
    tokenizer: Any | None,
    max_model_len: int,
    reserved_output_tokens: int | None = None,
    budget_tokens: int | None = None,
    config: AceCompressionConfig | None = None,
    embed: Callable[[list[str]], Sequence[Sequence[float]]] | None = None,
) -> tuple[list[Any], AceCompressionStats | None]:
    """
    Apply ACE to a chat request if it is switched on.

    Disabled is a hard no-op: the same message objects come back and nothing is
    logged, measured or allocated, so the request is byte-identical to what it
    would have been without this feature compiled in.
    """
    if not enabled:
        return list(messages), None
    budget = resolve_budget_tokens(max_model_len, reserved_output_tokens, budget_tokens)
    try:
        out, stats = compress_chat_history(
            messages,
            budget,
            tokenizer=tokenizer,
            config=config,
            embed=embed,
        )
    except Exception:
        # Compression is an optimisation. A bug in it must never cost a user
        # their request.
        logger.exception("ACE: compression failed; serving the request intact.")
        return list(messages), None
    if stats.output_tokens < stats.input_tokens:
        record_compression_metrics(stats)
    return out, stats


# ---------------------------------------------------------------------------
# Backwards compatibility with the refuted implementation
# ---------------------------------------------------------------------------

_REFUTED_NAMES = frozenset(
    {
        "AttentionImportanceTracker",
        "register_tracker",
        "get_tracker",
        "release_tracker",
        "compute_line_token_spans",
        "ace_compress",
        "apply_ace_eviction",
        "_heuristic_score",
        "_BM25Scorer",
        "_bm25",
        "_extract_query",
        "_tracker_registry",
        "_registry_lock",
    }
)

_refuted_warned = False


def __getattr__(name: str) -> Any:
    """Serve the retired heuristic/attention API from its own module.

    Not re-exported eagerly on purpose: reaching for these names is the
    explicit, visible act of opting into a scorer measured at 6.07 F1 against
    a 6.62 do-nothing floor, whose content rules alone cost -3.02 F1. Nothing
    on the serving path imports them.
    """
    if name in _REFUTED_NAMES:
        global _refuted_warned
        from vllm.entrypoints import context_compression_refuted as _refuted

        if not _refuted_warned:
            _refuted_warned = True
            logger.warning(
                "vllm.entrypoints.context_compression.%s is REFUTED "
                "(heuristic/attention scorer: 6.07 F1 against a 6.62 "
                "tail-truncation floor; its content rules alone cost -3.02 "
                "F1). It is not used by the server. Import it from "
                "vllm.entrypoints.context_compression_refuted instead.",
                name,
            )
        return getattr(_refuted, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
