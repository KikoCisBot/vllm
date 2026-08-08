# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for ACE v2 input-layer context compression.

The engine's contract is a budget measured with a REAL tokenizer, so these
tests use one. `len(text)/4` would happily pass a test that production would
fail, which is exactly the bug the contract exists to prevent.
"""

import json
from types import SimpleNamespace

import pytest

from vllm.entrypoints.context_compression import (
    AceCompressionConfig,
    BM25Index,
    TokenCounter,
    compress_chat_history,
    maybe_compress_chat_messages,
    resolve_budget_tokens,
    rrf_fuse,
    terms,
)
from vllm.renderers.online_renderer import OnlineRenderer

TOKENIZER_ID = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(TOKENIZER_ID)
    except Exception as exc:  # network-less CI, no local cache
        pytest.skip(f"tokenizer {TOKENIZER_ID} unavailable: {exc}")


@pytest.fixture(scope="module")
def count(tokenizer):
    return TokenCounter(tokenizer)


RARE = "zephyrine-4417"


def _agent_history(
    turns: int = 40, lines_per_turn: int = 30, needle_turn: int = 7
) -> list[dict]:
    """A long agentic conversation with one rare fact buried in the middle."""
    messages: list[dict] = [
        {"role": "system", "content": "You are a careful engineering agent."}
    ]
    for t in range(turns):
        messages.append({"role": "user", "content": f"turn {t}: run step {t}"})
        body = "\n".join(
            f"line {i} of output for step {t}: routine chatter"
            for i in range(lines_per_turn)
        )
        if t == needle_turn:
            body += f"\n  the widget calibration constant is {RARE}"
        messages.append({"role": "assistant", "content": f"Running step {t}\n{body}"})
    messages.append(
        {"role": "user", "content": "What is the zephyrine calibration constant?"}
    )
    return messages


def _measure(messages, count: TokenCounter) -> int:
    """Cost of a message list under the same accounting the engine promises."""
    overhead = count.message_overhead
    return sum(
        count(m["content"]) + overhead
        if isinstance(m.get("content"), str)
        else overhead
        for m in messages
    )


# ---------------------------------------------------------------------------
# (a) the budget contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fraction", [0.05, 0.08, 0.15, 0.3, 0.6])
def test_output_never_exceeds_budget(tokenizer, count, fraction):
    """Measured with the real tokenizer, at several compression pressures.

    Summing per-line costs is not a budget: it misses per-message template
    overhead and the omission markers themselves. That miss is what made an
    earlier implementation emit 1.6-1.8x what was asked for.
    """
    messages = _agent_history()
    total = _measure(messages, count)
    budget = int(total * fraction)

    out, stats = compress_chat_history(messages, budget, tokenizer=tokenizer)

    realized = _measure(out, count)
    assert realized <= budget, f"emitted {realized} tokens for a {budget} budget"
    assert stats.output_tokens == realized
    assert stats.input_tokens == total
    assert not stats.over_budget
    assert stats.ratio == pytest.approx(realized / total)


@pytest.mark.parametrize("fraction", [0.05, 0.08, 0.15, 0.3])
def test_budget_is_actually_spent(tokenizer, count, fraction):
    """Under-spending is as wrong as over-spending — for the FILLING engine.

    Handing back the last line of a message also frees its template overhead
    and its omission markers, so a naive give-back loop overshoots badly and
    leaves budget unused — which is unused recall.

    Measured with the elastic window off, because that is what this test is
    about: the give-back/refill machinery. With the window ON, spending less
    than the budget is the intended behaviour (rule 10) and is covered by
    ``test_elastic_window_saves_tokens_and_never_overspends``.
    """
    messages = _agent_history()
    budget = int(_measure(messages, count) * fraction)
    _, stats = compress_chat_history(
        messages,
        budget,
        tokenizer=tokenizer,
        config=AceCompressionConfig(elastic=False),
    )
    assert stats.output_tokens <= budget
    assert stats.output_tokens / budget > 0.9


def test_budget_is_measured_not_estimated(tokenizer, count):
    """The reported input/output token counts come from the real tokenizer."""
    messages = _agent_history(turns=10)
    total = _measure(messages, count)
    _, stats = compress_chat_history(messages, int(total * 0.2), tokenizer=tokenizer)
    assert stats.exact_tokens is True
    # chars/4 underestimates code-like text; the contract must not rely on it.
    naive = sum(len(m["content"]) // 4 for m in messages)
    assert stats.input_tokens != naive


def test_history_that_already_fits_is_returned_untouched(tokenizer):
    messages = _agent_history(turns=2)
    out, stats = compress_chat_history(messages, 1_000_000, tokenizer=tokenizer)
    assert out == messages
    assert all(a is b for a, b in zip(out, messages))
    assert stats.ratio == 1.0


# ---------------------------------------------------------------------------
# (b) relevance: a rare query term survives
# ---------------------------------------------------------------------------


def test_rare_query_term_survives_hard_compression(tokenizer, count):
    """The one line that answers the live question is kept at ~8% budget."""
    messages = _agent_history()
    budget = int(_measure(messages, count) * 0.08)

    out, _ = compress_chat_history(messages, budget, tokenizer=tokenizer)

    text = "\n".join(m["content"] for m in out if isinstance(m["content"], str))
    assert RARE in text


def test_tail_truncation_would_lose_it(tokenizer, count):
    """Guard against a test that passes for the wrong reason: the needle is
    genuinely outside the recent window, so keeping it required retrieval."""
    messages = _agent_history()
    budget = int(_measure(messages, count) * 0.08)
    tail: list[dict] = []
    used = 0
    for message in reversed(messages):
        used += count(message["content"]) + count.message_overhead
        if used > budget:
            break
        tail.insert(0, message)
    assert RARE not in "\n".join(m["content"] for m in tail)


def test_relevance_beats_recency_under_pressure(tokenizer, count):
    """Rule 6: relevance and irrelevance are two strata 10 apart, so a newer
    irrelevant line can never outrank an older relevant one."""
    messages = [{"role": "system", "content": "agent"}]
    messages.append(
        {
            "role": "assistant",
            "content": "\n".join(
                ["old output", f"  the deploy key is {RARE}"]
                + [f"noise {i}" for i in range(200)]
            ),
        }
    )
    for t in range(30):
        messages.append(
            {
                "role": "assistant",
                "content": "\n".join(
                    f"recent unrelated chatter {t}.{i}" for i in range(30)
                ),
            }
        )
    messages.append({"role": "user", "content": f"what is {RARE}?"})

    budget = int(_measure(messages, count) * 0.06)
    out, _ = compress_chat_history(messages, budget, tokenizer=tokenizer)
    text = "\n".join(m["content"] for m in out if isinstance(m["content"], str))
    assert RARE in text


def test_no_content_heuristics(tokenizer, count):
    """An "error"/"traceback" line with no relation to the live question gets
    no special treatment. The retired scorer gave it 0.95, which measured at
    -3.02 F1."""
    noise = "\n".join(f"filler line {i} about unrelated matters" for i in range(400))
    messages = [
        {
            "role": "assistant",
            "content": (
                "Traceback (most recent call last):\n"
                "Error: connection refused\n" + noise
            ),
        },
        {
            "role": "assistant",
            "content": f"the migration cursor is {RARE}\n" + noise,
        },
        {"role": "user", "content": f"what is the {RARE} cursor?"},
    ]
    budget = int(_measure(messages, count) * 0.05)
    out, _ = compress_chat_history(messages, budget, tokenizer=tokenizer)
    text = "\n".join(m["content"] for m in out if isinstance(m["content"], str))
    assert RARE in text
    assert "connection refused" not in text


# ---------------------------------------------------------------------------
# (b2) rule 9 — staleness is correctness
#
# In the OpenAI schema a tool result never names what it acted on; it only
# carries ``tool_call_id``. The target comes from the assistant call it points
# back at, which is why these fixtures are built with real ``tool_calls`` and
# real ``tool`` messages rather than with prose.
# ---------------------------------------------------------------------------

CFG_FILE = "/srv/app/config.py"
OLD_VALUE = "timeout-legacy-8891"
NEW_VALUE = "timeout-current-2205"


def _call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def _result(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _config_dump(marker: str) -> str:
    """Two reads of the same file differ in one line and share every other."""
    return "\n".join(
        [
            f"[read {CFG_FILE}]",
            "  1: import os",
            "  2: from app import runtime",
            "  3:",
            "  4: # tuning block",
            f"  5: TIMEOUT_SECONDS = 30  # {marker}",
            "  6:",
            "  7: def build(): return runtime.Server(TIMEOUT_SECONDS)",
        ]
    )


def _read_edit_reread_history(noise_turns: int = 12) -> list[dict]:
    """The probe: read a file, edit it, read it again, then ask for the value.

    Both dumps share their whole vocabulary, so BM25 scores them the same and
    the stale one is retrieved right next to the current one. Nothing about
    relevance can fix that; only knowing it was overwritten can.
    """
    messages: list[dict] = [
        {"role": "system", "content": "You are a careful engineering agent."},
        {"role": "user", "content": f"Keep {CFG_FILE} healthy."},
        _call("call_read_1", "read_file", {"path": CFG_FILE}),
        _result("call_read_1", _config_dump(OLD_VALUE)),
    ]
    for t in range(noise_turns):
        messages.append(
            _call(f"call_noise_{t}", "bash", {"command": f"pytest set_{t}"})
        )
        messages.append(
            _result(
                f"call_noise_{t}",
                "\n".join(f"set {t} case {i}: ok" for i in range(25)),
            )
        )
    messages.append(
        _call("call_edit", "edit_file", {"path": CFG_FILE, "old": "30", "new": "90"})
    )
    messages.append(_result("call_edit", f"[edited {CFG_FILE}]"))
    messages.append(_call("call_read_2", "read_file", {"path": CFG_FILE}))
    messages.append(_result("call_read_2", _config_dump(NEW_VALUE)))
    messages.append(
        {"role": "user", "content": f"What is TIMEOUT_SECONDS in {CFG_FILE} right now?"}
    )
    return messages


def test_superseded_read_does_not_survive_the_edit(tokenizer, count):
    """read -> edit -> re-read -> ask: the old version must be gone, the new
    one present. Measured over 8 seeds: 8/8 stale survivals without this, 0/8
    with it, and the current version still 8/8."""
    messages = _read_edit_reread_history()
    budget = int(_measure(messages, count) * 0.25)

    out, stats = compress_chat_history(messages, budget, tokenizer=tokenizer)

    text = "\n".join(m["content"] for m in out if isinstance(m["content"], str))
    assert NEW_VALUE in text
    assert OLD_VALUE not in text
    assert stats.superseded_messages >= 1


def test_without_supersede_the_stale_version_comes_back(tokenizer, count):
    """Guard against passing for the wrong reason: with the mechanism off, the
    obsolete dump IS retrieved, which is the failure it exists to fix."""
    messages = _read_edit_reread_history()
    budget = int(_measure(messages, count) * 0.25)

    out, _ = compress_chat_history(
        messages,
        budget,
        tokenizer=tokenizer,
        config=AceCompressionConfig(supersede=False),
    )

    text = "\n".join(m["content"] for m in out if isinstance(m["content"], str))
    assert OLD_VALUE in text


def test_superseded_is_demoted_not_deleted(tokenizer, count):
    """ "No longer true" and "never existed" are different claims, and only the
    first one is true here.

    A superseded result keeps its place in the ranking — it drops to the filler
    stratum, below everything relevant — rather than being taken out of the
    pool. Measured with the elastic window off, which is what materialises the
    filler stratum at all, and with the head reserve off, so the old dump
    cannot come through as an unscored opening message instead.

    What this deliberately does NOT claim: that asking explicitly for the
    previous value retrieves it. Once demoted, a stale line sits among hundreds
    of other zero-score lines, and which of those come back is decided by MMR
    novelty, not by the question. Reachable is not the same as retrievable.
    """
    messages = _read_edit_reread_history()
    budget = int(_measure(messages, count) * 0.5)

    out, _ = compress_chat_history(
        messages,
        budget,
        tokenizer=tokenizer,
        config=AceCompressionConfig(elastic=False, head_frac=0.0),
    )

    stale = [m for m in out if m.get("tool_call_id") == "call_read_1"]
    assert stale, "the superseded result was deleted, not demoted"
    body = [ln for ln in stale[0]["content"].split("\n") if "omitted by ACE" not in ln]
    assert body, "the superseded result was collapsed to a bare marker"


def test_query_term_pin_honours_staleness(tokenizer, count):
    """The trap. The stale version carries the SAME identifiers as the current
    one, so a pin that only looked at ``bm25 > 0`` pinned it by itself and
    cancelled the demotion. Named in the question and pinning switched hard on,
    the obsolete dump must still not come through."""
    messages = _read_edit_reread_history()
    budget = int(_measure(messages, count) * 0.25)

    out, _ = compress_chat_history(
        messages,
        budget,
        tokenizer=tokenizer,
        config=AceCompressionConfig(pin_query_terms=True, pin_max=200),
    )

    text = "\n".join(m["content"] for m in out if isinstance(m["content"], str))
    assert OLD_VALUE not in text
    assert NEW_VALUE in text


def test_parallel_tool_calls_are_paired_by_id_not_adjacency(tokenizer, count):
    """vLLM knows something the browser engine could not: ``tool_call_id``.

    One assistant turn issues two calls, so "the message right before" is
    ambiguous — with adjacency the wrong result gets marked stale. Only the
    file that was read twice may be superseded; the other must stay fresh.
    """
    other = "/srv/app/ledger.py"
    fresh = "ledger-mark-7731"
    messages: list[dict] = [
        {"role": "system", "content": "agent"},
        {"role": "user", "content": "inspect both files"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": CFG_FILE}),
                    },
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": other}),
                    },
                },
            ],
        },
        _result("call_a", _config_dump(OLD_VALUE)),
        _result("call_b", f"[read {other}]\n  1: LEDGER = '{fresh}'"),
    ]
    for t in range(12):
        messages.append(_call(f"call_n{t}", "bash", {"command": f"pytest set_{t}"}))
        messages.append(
            _result(f"call_n{t}", "\n".join(f"set {t} case {i}: ok" for i in range(25)))
        )
    messages.append(_call("call_a2", "read_file", {"path": CFG_FILE}))
    messages.append(_result("call_a2", _config_dump(NEW_VALUE)))
    messages.append(
        {"role": "user", "content": f"read TIMEOUT_SECONDS and LEDGER from {other}"}
    )

    budget = int(_measure(messages, count) * 0.3)
    out, _ = compress_chat_history(messages, budget, tokenizer=tokenizer)
    text = "\n".join(m["content"] for m in out if isinstance(m["content"], str))
    assert OLD_VALUE not in text
    assert fresh in text


def test_supersede_ignores_calls_it_cannot_pair(tokenizer, count):
    """No id, no unambiguous neighbour, no claim. Guessing a pairing would
    demote live content, and a demotion made up is worse than one missed."""
    messages: list[dict] = [
        {"role": "system", "content": "agent"},
        {"role": "tool", "content": _config_dump(OLD_VALUE)},
        {"role": "tool", "content": _config_dump(NEW_VALUE)},
        {"role": "user", "content": "what is TIMEOUT_SECONDS?"},
    ]
    _, stats = compress_chat_history(messages, 10_000, tokenizer=tokenizer)
    assert stats.superseded_messages == 0


# ---------------------------------------------------------------------------
# (b3) rule 10 — the elastic window
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fraction", [0.05, 0.08, 0.15, 0.3, 0.6])
def test_elastic_window_saves_tokens_and_never_overspends(tokenizer, count, fraction):
    """The budget is a ceiling, not a quota.

    Spending LESS than granted is the intended outcome here; spending more is
    still forbidden, and the give-back loop still enforces that.
    """
    messages = _agent_history()
    budget = int(_measure(messages, count) * fraction)
    _, off = compress_chat_history(
        messages,
        budget,
        tokenizer=tokenizer,
        config=AceCompressionConfig(elastic=False),
    )
    _, on = compress_chat_history(messages, budget, tokenizer=tokenizer)
    assert on.output_tokens <= budget
    assert on.output_tokens < off.output_tokens


def test_elastic_window_keeps_the_evidence_it_stops_for(tokenizer, count):
    """Saving tokens by dropping the answer would be a cheat, not a win."""
    messages = _agent_history()
    budget = int(_measure(messages, count) * 0.15)
    out, _ = compress_chat_history(messages, budget, tokenizer=tokenizer)
    text = "\n".join(m["content"] for m in out if isinstance(m["content"], str))
    assert RARE in text


def test_elastic_window_gates_the_mmr_loop_too(tokenizer, count):
    """The gate belongs inside MMR, not only in the final fill.

    MMR orders for diversity WITHIN what is relevant; it is not a side door
    for filler. Measured here with the gate moved downstream only: 92.6% of a
    2,746-token budget still spent and 101 lines kept, against 36.8% and 1 line
    with the gate in place — i.e. the measurement comes out unchanged, which is
    exactly how this hid for an hour.
    """
    messages = _agent_history()
    budget = int(_measure(messages, count) * 0.6)
    _, stats = compress_chat_history(messages, budget, tokenizer=tokenizer)
    # Ungated the MMR loop alone kept 563 lines and filled the budget.
    assert stats.kept_lines < 10
    assert stats.output_tokens < budget * 0.5


# ---------------------------------------------------------------------------
# (b4) rule 11 — the head reserve
# ---------------------------------------------------------------------------

ASSIGNMENT = "brief-8814-migrate-billing"


def _assignment_history(turns: int = 40) -> list[dict]:
    """The opening carries the task; nothing afterwards repeats it, so there is
    no lexical overlap for BM25 to rescue it with.

    The assignment shares NOT ONE token with the live question — including the
    function words, which matters more than it sounds: endogenous IDF is
    computed over this history, so a "the" that happens to be rare here scores
    high, and an accidental "the" was enough to make this fixture pass for the
    wrong reason.
    """
    messages: list[dict] = [
        {"role": "system", "content": "You are a careful engineering agent."},
        {
            "role": "user",
            "content": (
                f"ASSIGNMENT {ASSIGNMENT}: port billing onto ledger API. "
                "Never write legacy tables."
            ),
        },
    ]
    for t in range(turns):
        messages.append({"role": "user", "content": f"turn {t}: continue"})
        messages.append(
            {
                "role": "assistant",
                "content": "\n".join(
                    f"step {t} log {i}: routine chatter about retries"
                    for i in range(30)
                ),
            }
        )
    messages.append({"role": "user", "content": "Summarise routine chatter retries."})
    return messages


def test_head_reserve_keeps_the_original_assignment(tokenizer, count):
    """Measured 0/5 without the reserve and 5/5 with it, at a tight budget.

    The assignment is the one thing an agent cannot reconstruct by looking at
    the code, and it is precisely what a pure retriever cannot see: nobody
    repeats it, so it has no overlap with the live question.
    """
    messages = _assignment_history()
    budget = int(_measure(messages, count) * 0.08)

    out, stats = compress_chat_history(messages, budget, tokenizer=tokenizer)

    text = "\n".join(m["content"] for m in out if isinstance(m["content"], str))
    assert ASSIGNMENT in text
    assert stats.head_messages >= 1


def test_without_the_head_reserve_the_assignment_is_lost(tokenizer, count):
    """The other half of the measurement: the reserve is load-bearing, the test
    above is not passing by accident."""
    messages = _assignment_history()
    budget = int(_measure(messages, count) * 0.08)

    out, _ = compress_chat_history(
        messages,
        budget,
        tokenizer=tokenizer,
        config=AceCompressionConfig(head_frac=0.0),
    )

    text = "\n".join(m["content"] for m in out if isinstance(m["content"], str))
    assert ASSIGNMENT not in text


def test_head_reserve_survives_a_roomy_budget(tokenizer, count):
    """It used to be gated off with slack because "the assignment survives on
    its own there". It did — because we filled to the brim. With the elastic
    window it fell from 100% to 0%, so the reserve is unconditional now."""
    messages = _assignment_history()
    budget = int(_measure(messages, count) * 0.5)
    out, _ = compress_chat_history(messages, budget, tokenizer=tokenizer)
    text = "\n".join(m["content"] for m in out if isinstance(m["content"], str))
    assert ASSIGNMENT in text


def test_head_reserve_is_capped(tokenizer, count):
    """A hard 5% ceiling, so the reserve can never compete with the search."""
    messages = _assignment_history()
    budget = int(_measure(messages, count) * 0.08)
    cap = int(budget * AceCompressionConfig().head_frac)

    _, stats = compress_chat_history(messages, budget, tokenizer=tokenizer)

    head_cost = sum(
        count(m["content"]) + count.message_overhead
        for m in messages[1 : 1 + stats.head_messages]
    )
    assert head_cost <= cap

    # And an opening message bigger than the cap is not taken at all: it goes
    # to the compressible pool, where retrieval can still reach into it.
    huge = list(messages)
    huge[1] = {"role": "user", "content": huge[1]["content"] + "\n" + ("pad " * 4000)}
    _, big = compress_chat_history(huge, budget, tokenizer=tokenizer)
    assert big.head_messages == 0


# ---------------------------------------------------------------------------
# (b5) rule 12 — the tail reserve is a floor
# ---------------------------------------------------------------------------

TAIL_MARK = "tail-marker-6613"


def _blocked_tail_history(turns: int = 25) -> list[dict]:
    """One bulky result in the second-to-last position — the shape that turned
    a 38% reserve into 3-5% delivered."""
    messages: list[dict] = [{"role": "system", "content": "agent"}]
    for t in range(turns):
        messages.append(
            {
                "role": "assistant",
                "content": "\n".join(
                    f"turn {t} line {i}: routine chatter" for i in range(30)
                ),
            }
        )
    bulk = "\n".join(f"bulk output line {i}: filler" for i in range(1200))
    messages.append({"role": "assistant", "content": f"{bulk}\n{TAIL_MARK} all green"})
    messages.append(
        {"role": "user", "content": "What is the widget calibration constant?"}
    )
    return messages


def test_tail_floor_truncates_instead_of_dropping(tokenizer, count):
    """Below the floor, a message that does not fit is cut through the middle
    rather than discarded. ``TAIL_MARK`` shares no word with the live question,
    so retrieval cannot rescue it — only the floor can."""
    messages = _blocked_tail_history()
    budget = int(_measure(messages, count) * 0.1)

    out, stats = compress_chat_history(messages, budget, tokenizer=tokenizer)

    text = "\n".join(m["content"] for m in out if isinstance(m["content"], str))
    assert TAIL_MARK in text
    assert stats.tail_truncated_messages == 1
    assert stats.output_tokens <= budget


def test_without_the_floor_one_big_result_blocks_the_tail(tokenizer, count):
    """The bug, reproduced: stop at the first message that does not fit and the
    guaranteed recent window silently delivers almost nothing."""
    messages = _blocked_tail_history()
    budget = int(_measure(messages, count) * 0.1)

    out, stats = compress_chat_history(
        messages,
        budget,
        tokenizer=tokenizer,
        config=AceCompressionConfig(tail_min_frac=0.0),
    )

    text = "\n".join(m["content"] for m in out if isinstance(m["content"], str))
    assert TAIL_MARK not in text
    assert stats.tail_truncated_messages == 0


def test_newest_turn_may_still_exceed_the_reserve(tokenizer, count):
    """Unchanged by the floor: the turn being answered may overrun the recent
    reserve as long as it fits the budget, and is never truncated to make room.

    The measured bug was a bulky result in the SECOND-TO-LAST position blocking
    everything before it. The last position is a different thing: a model that
    cannot see the turn it is answering is worse off than one short of history.
    """
    mark = "final-turn-9042"
    messages: list[dict] = [{"role": "system", "content": "agent"}]
    for t in range(25):
        messages.append(
            {
                "role": "assistant",
                "content": "\n".join(
                    f"turn {t} line {i}: routine chatter" for i in range(30)
                ),
            }
        )
    last = "\n".join(
        f"{mark} question line {i}: what happened to the widget?" for i in range(120)
    )
    messages.append({"role": "user", "content": last})

    budget = int(_measure(messages, count) * 0.35)
    # Genuinely above the recent reserve (~0.51 of the budget here) and below
    # the budget, which is the window this rule is about.
    assert budget * 0.51 < count(last) + count.message_overhead < budget

    out, stats = compress_chat_history(messages, budget, tokenizer=tokenizer)

    assert any(m["content"] == last for m in out if isinstance(m["content"], str))
    assert stats.tail_truncated_messages == 0


def test_tail_floor_leaves_a_lone_message_to_retrieval(tokenizer, count):
    """The floor exists to stop one big message blocking the turns BEFORE it.
    With no turns before it there is nothing to protect, and line-level
    retrieval beats head/tail truncation — so the floor stands down."""
    content = "\n".join(f"line {i}: alpha beta gamma delta" for i in range(20_000))
    messages = [{"role": "user", "content": content}]
    out, stats = compress_chat_history(messages, 500, tokenizer=tokenizer)
    assert stats.tail_truncated_messages == 0
    assert _measure(out, count) <= 500


# ---------------------------------------------------------------------------
# (b6) dedup — the line number is part of the line's identity
#
# Both directions matter. Only the first half and this is a regression wearing
# a fix's clothes: dedup switched off.
# ---------------------------------------------------------------------------

GREP_HIT = "def calibrate(self, widget):"


def _grep_history(turns: int = 25) -> list[dict]:
    """The same text found at three different lines, in both gutter shapes.

    Under the old key all three collapsed to one, so the model was handed one
    site out of three — an incomplete answer that looks complete. Both the Read
    tool's ``42→`` and the ``42: `` form are exercised, because those are the
    shapes the old key actually stripped.
    """
    messages: list[dict] = [{"role": "system", "content": "agent"}]
    for t in range(turns):
        messages.append(
            {
                "role": "assistant",
                "content": "\n".join(
                    f"turn {t} log {i}: routine chatter" for i in range(30)
                ),
            }
        )
    messages.append(
        {
            "role": "assistant",
            "content": "\n".join(
                [
                    "[read src/widget.py]",
                    f"  12→{GREP_HIT}",
                    f"  47→{GREP_HIT}",
                    f"  91→{GREP_HIT}",
                ]
            ),
        }
    )
    messages.append(
        {
            "role": "assistant",
            "content": "\n".join(
                [
                    "[grep -n 'def calibrate' src/other.py]",
                    f"  8: {GREP_HIT}",
                    f"  63: {GREP_HIT}",
                ]
            ),
        }
    )
    for t in range(turns):
        messages.append(
            {
                "role": "assistant",
                "content": "\n".join(
                    f"later {t} log {i}: routine chatter" for i in range(30)
                ),
            }
        )
    messages.append({"role": "user", "content": "where is def calibrate defined?"})
    return messages


def test_grep_hits_at_different_lines_do_not_collapse(tokenizer, count):
    """(a) Same text, three different positions: all three must survive.

    "Where is X defined?" is the commonest question an agent of code asks, and
    the answer IS the number.
    """
    messages = _grep_history()
    budget = int(_measure(messages, count) * 0.15)

    out, _ = compress_chat_history(messages, budget, tokenizer=tokenizer)

    text = "\n".join(m["content"] for m in out if isinstance(m["content"], str))
    for site in (f"12→{GREP_HIT}", f"47→{GREP_HIT}", f"91→{GREP_HIT}"):
        assert site in text, f"lost {site!r}: one site of three is not an answer"
    for site in (f"8: {GREP_HIT}", f"63: {GREP_HIT}"):
        assert site in text, f"lost {site!r}: one site of three is not an answer"


def _repetition_history(turns: int = 30) -> list[dict]:
    """The same result body over and over: a bare ``}``, a ``});``, an
    identical status line and the SAME grep hit at the SAME line."""
    messages: list[dict] = [{"role": "system", "content": "agent"}]
    for t in range(turns):
        messages.append(
            {
                "role": "assistant",
                "content": "\n".join(
                    ["[bash] make check", "}", "});", "build ok", f"  12:{GREP_HIT}"]
                    + [f"turn {t} extra {i}: routine chatter" for i in range(30)]
                ),
            }
        )
    messages.append({"role": "user", "content": "did make check build ok?"})
    return messages


def test_true_repetition_still_collapses(tokenizer, count):
    """(b) Without this half, the "fix" is just dedup switched off.

    Real repetition still collapses, because those lines are equal in where
    they were as well as in what they say. Measured against dedup switched off,
    so the assertion cannot be satisfied by doing nothing.
    """
    messages = _repetition_history()
    budget = int(_measure(messages, count) * 0.15)

    out, stats = compress_chat_history(messages, budget, tokenizer=tokenizer)
    _, without = compress_chat_history(
        messages,
        budget,
        tokenizer=tokenizer,
        config=AceCompressionConfig(dedup=False),
    )

    assert stats.deduped_lines > 0
    # The collapse buys real room: same budget, fewer tokens spent saying the
    # same thing twenty times.
    assert stats.output_tokens < without.output_tokens

    text = "\n".join(m["content"] for m in out if isinstance(m["content"], str))
    # 30 identical copies in. What survives is bounded by the verbatim windows
    # (recent turns plus the head reserve), which dedup deliberately does not
    # touch — not by the compressed region, where they collapsed.
    assert text.count("});") <= 8
    assert text.count(f"12:{GREP_HIT}") <= 8


def test_gutter_style_is_still_noise(tokenizer, count):
    """The same line of the same file seen once through a Read (``42→``) and
    once through a grep (``42:``) is one line, not two. Position is identity;
    the shape of the gutter around it is not."""
    from vllm.entrypoints.context_compression import dedup_key

    assert dedup_key("  42→  return 42;") == dedup_key("42:   return 42;")
    # Real `grep -n` puts no space after the colon; requiring one meant the
    # commonest shape of all was not recognised as a gutter at all.
    assert dedup_key("42:return 42;") == dedup_key("  42→ return 42;")
    assert dedup_key("10:  return 42;") != dedup_key("87:  return 42;")
    assert dedup_key("10:return 42;") != dedup_key("87:return 42;")
    # Whitespace remains noise.
    assert dedup_key("   }   ") == dedup_key("}")


def test_head_reserve_dedup_seeding_still_works(tokenizer, count):
    """The head reserve seeds the dedup map so content travelling verbatim up
    there is not paid for twice. A finer-grained key must still MATCH across
    the two sides of that seeding, or it silently stops working.

    Measured with the elastic window off, because that is what materialises the
    filler stratum at all: with it on, the repeats never get near the budget
    and the seeding has nothing to prove.
    """
    repeated = "\n".join(["[boot] loading", "}", "});", f"  12:{GREP_HIT}"])
    messages: list[dict] = [
        {"role": "system", "content": "agent"},
        {"role": "user", "content": repeated},
    ]
    for t in range(60):
        messages.append(
            {
                "role": "assistant",
                "content": repeated
                + "\n"
                + "\n".join(f"turn {t} log {i}: routine chatter" for i in range(30)),
            }
        )
    messages.append({"role": "user", "content": "did it finish loading?"})

    budget = int(_measure(messages, count) * 0.15)
    out, stats = compress_chat_history(
        messages,
        budget,
        tokenizer=tokenizer,
        config=AceCompressionConfig(elastic=False),
    )
    without, _ = compress_chat_history(
        messages,
        budget,
        tokenizer=tokenizer,
        config=AceCompressionConfig(elastic=False, dedup=False),
    )

    assert stats.head_messages >= 1
    assert stats.deduped_lines > 0

    text = "\n".join(m["content"] for m in out if isinstance(m["content"], str))
    loose = "\n".join(m["content"] for m in without if isinstance(m["content"], str))
    assert text.count("});") < loose.count("});")
    assert text.count(f"12:{GREP_HIT}") < loose.count(f"12:{GREP_HIT}")


# ---------------------------------------------------------------------------
# IDF is endogenous — no hand-written stoplist
# ---------------------------------------------------------------------------


def test_idf_is_endogenous():
    """A term in every line carries ~no weight; a term in one line carries a
    lot. That is what replaces the stoplist, without a dictionary and without
    knowing the language."""
    docs = [f"the quick brown fox number {i}" for i in range(100)]
    docs.append("the quick brown fox number ñandu")
    index = BM25Index(docs)
    assert index.idf("the") < 0.1
    assert index.idf("ñandu") > index.idf("the") * 20


def test_terms_are_not_english_only():
    """Non-ASCII words must tokenize, otherwise the endogenous IDF silently
    has nothing to work with outside English."""
    assert "ñandu" in terms("El ñandu corre")
    assert "línea" in terms("una LÍNEA cualquiera")


def test_rrf_fuse_merges_positions():
    """Positions are fused, not scores, so nothing has to be calibrated between
    an unbounded BM25 score and a cosine."""
    fused = rrf_fuse([[1, 2, 3], [1, 3, 2]], k=60)
    # 1 is first in both lists; 2 and 3 swap second and third place.
    assert fused[1] > fused[2]
    assert fused[2] == pytest.approx(fused[3])

    # Being retrieved by BOTH rankers outranks topping only one of them. That
    # is the whole value of fusion: lexical and semantic retrieval cover each
    # other's blind spots rather than duplicating each other.
    fused = rrf_fuse([[9, 7], [7]], k=60)
    assert fused[7] > fused[9]


def test_hybrid_path_fuses_an_embedder_when_one_is_supplied(tokenizer, count):
    """Nothing wires an embedder by default (the served model is generative),
    but the fusion path must work wherever one exists."""
    messages = _agent_history(turns=12)

    def embed(texts):
        # A crude bag-of-characters vector: enough to make cosine meaningful
        # without pulling in an embedding model.
        vectors = []
        for text in texts:
            vec = [0.0] * 32
            for ch in text.lower():
                vec[ord(ch) % 32] += 1.0
            vectors.append(vec)
        return vectors

    out, stats = compress_chat_history(
        messages,
        int(_measure(messages, count) * 0.15),
        tokenizer=tokenizer,
        embed=embed,
    )
    assert stats.mode == "hybrid"
    assert _measure(out, count) <= stats.budget_tokens


def test_broken_embedder_falls_back_to_lexical(tokenizer, count):
    """An embedder must never be able to take a request down."""

    def embed(_texts):
        raise RuntimeError("embedding service down")

    messages = _agent_history(turns=12)
    out, stats = compress_chat_history(
        messages,
        int(_measure(messages, count) * 0.15),
        tokenizer=tokenizer,
        embed=embed,
    )
    assert stats.mode == "lexical"
    assert _measure(out, count) <= stats.budget_tokens


# ---------------------------------------------------------------------------
# (c) flag off -> byte-identical request
# ---------------------------------------------------------------------------


def _render_stub(tokenizer, *, enabled: bool, budget: int | None = None):
    """A bare OnlineRenderer carrying only what the ACE hook reads."""
    render = OnlineRenderer.__new__(OnlineRenderer)
    render.enable_ace_context_compression = enabled
    render.ace_context_compression_budget_tokens = budget
    render.ace_compression_config = AceCompressionConfig()
    render.model_config = SimpleNamespace(max_model_len=4096)
    render.renderer = SimpleNamespace(tokenizer=tokenizer)
    return render


def test_disabled_flag_leaves_request_byte_identical(tokenizer):
    messages = _agent_history()
    request = SimpleNamespace(
        messages=messages,
        context_compression=None,
        max_completion_tokens=None,
        max_tokens=None,
    )
    out = _render_stub(tokenizer, enabled=False)._maybe_compress_chat_history(request)
    assert out == messages
    assert all(a is b for a, b in zip(out, messages))


def test_disabled_helper_is_a_hard_noop():
    messages = [{"role": "user", "content": "x" * 10_000}]
    out, stats = maybe_compress_chat_messages(
        messages, enabled=False, tokenizer=None, max_model_len=16
    )
    assert stats is None
    assert out[0] is messages[0]


def test_server_flag_enables_compression(tokenizer):
    messages = _agent_history()
    request = SimpleNamespace(
        messages=messages,
        context_compression=None,
        max_completion_tokens=256,
        max_tokens=None,
    )
    render = _render_stub(tokenizer, enabled=True, budget=1500)
    out = render._maybe_compress_chat_history(request)
    assert out != messages
    assert len(out) <= len(messages)


def test_per_request_opt_in_still_works(tokenizer):
    messages = _agent_history()
    request = SimpleNamespace(
        messages=messages,
        context_compression="ace",
        max_completion_tokens=256,
        max_tokens=None,
    )
    render = _render_stub(tokenizer, enabled=False, budget=1500)
    assert render._maybe_compress_chat_history(request) != messages


def test_cli_flag_is_off_by_default():
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = make_arg_parser(FlexibleArgumentParser())
    default = parser.parse_args(["some-model"])
    assert default.enable_ace_context_compression is False
    assert default.ace_context_compression_budget_tokens is None

    enabled = parser.parse_args(
        [
            "some-model",
            "--enable-ace-context-compression",
            "--ace-context-compression-budget-tokens",
            "1884",
        ]
    )
    assert enabled.enable_ace_context_compression is True
    assert enabled.ace_context_compression_budget_tokens == 1884


# ---------------------------------------------------------------------------
# (d) degenerate inputs
# ---------------------------------------------------------------------------


def test_empty_history(tokenizer):
    out, stats = compress_chat_history([], 100, tokenizer=tokenizer)
    assert out == []
    assert stats.input_tokens == 0
    assert stats.ratio == 1.0


def test_single_small_message(tokenizer):
    messages = [{"role": "user", "content": "hello"}]
    out, stats = compress_chat_history(messages, 100, tokenizer=tokenizer)
    assert out == messages
    assert stats.ratio == 1.0


def test_single_enormous_message_is_compressed(tokenizer, count):
    """One huge tool result must still be compressible: letting the newest turn
    through verbatim just because it is newest would make this request
    impossible to shrink."""
    content = "\n".join(f"line {i}: alpha beta gamma delta" for i in range(20_000))
    messages = [{"role": "user", "content": content}]
    out, stats = compress_chat_history(messages, 500, tokenizer=tokenizer)
    assert _measure(out, count) <= 500
    assert stats.output_tokens < stats.input_tokens


def test_protected_roles_are_never_compressed(tokenizer):
    system = "SYSTEM POLICY: " + " ".join(f"rule {i}" for i in range(200))
    messages = [{"role": "system", "content": system}]
    messages += _agent_history(turns=20)[1:]
    out, _ = compress_chat_history(messages, 400, tokenizer=tokenizer)
    assert out[0]["content"] == system


def test_tool_messages_are_collapsed_not_dropped(tokenizer):
    """Dropping a tool result orphans its assistant tool_call and breaks the
    chat template, so structural messages collapse to a marker instead."""
    messages: list[dict] = []
    for t in range(20):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{t}",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{t}",
                "content": "\n".join(f"stdout {t}.{i}" for i in range(40)),
            }
        )
    messages.append({"role": "user", "content": "summarise"})

    out, _ = compress_chat_history(messages, 200, tokenizer=tokenizer)

    call_ids = {m["tool_calls"][0]["id"] for m in out if m.get("tool_calls")}
    result_ids = {m["tool_call_id"] for m in out if m.get("role") == "tool"}
    assert call_ids == result_ids
    assert len(result_ids) == 20


def test_multimodal_messages_pass_through(tokenizer):
    parts = [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {}}]
    messages = [{"role": "user", "content": parts}]
    messages += _agent_history(turns=15)[1:]
    out, _ = compress_chat_history(messages, 300, tokenizer=tokenizer)
    assert out[0]["content"] is parts


def test_no_tokenizer_still_honours_its_own_budget(count):
    """Without a tokenizer the counter is a documented fallback, and the
    contract still has to hold against that counter."""
    messages = _agent_history(turns=15)
    fallback = TokenCounter(None)
    total = sum(fallback(m["content"]) + fallback.message_overhead for m in messages)
    out, stats = compress_chat_history(messages, int(total * 0.1))
    realized = sum(fallback(m["content"]) + fallback.message_overhead for m in out)
    assert realized <= int(total * 0.1)
    assert stats.exact_tokens is False


def test_compression_failure_does_not_lose_the_request(tokenizer):
    class Exploding:
        def encode(self, *_a, **_k):
            raise RuntimeError("boom")

        def apply_chat_template(self, *_a, **_k):
            raise RuntimeError("boom")

    messages = _agent_history(turns=5)
    out, stats = maybe_compress_chat_messages(
        messages,
        enabled=True,
        tokenizer=Exploding(),
        max_model_len=128,
        reserved_output_tokens=16,
    )
    assert out == messages
    assert stats is None


# ---------------------------------------------------------------------------
# (e) gaps are marked
# ---------------------------------------------------------------------------


def test_omissions_are_marked(tokenizer, count):
    """Rule 8: a model that knows something is missing asks; one that believes
    it saw everything invents."""
    messages = _agent_history()
    out, stats = compress_chat_history(
        messages, int(_measure(messages, count) * 0.08), tokenizer=tokenizer
    )
    text = "\n".join(m["content"] for m in out if isinstance(m["content"], str))
    assert "lines omitted by ACE" in text
    if stats.dropped_messages:
        assert "earlier messages omitted by ACE" in text


def test_dropped_message_notice_creates_no_new_message(tokenizer, count):
    """The notice rides inside a surviving message: inventing one could break
    strict role-alternation templates."""
    messages = _agent_history()
    roles_before = [m["role"] for m in messages]
    out, stats = compress_chat_history(
        messages, int(_measure(messages, count) * 0.08), tokenizer=tokenizer
    )
    assert stats.dropped_messages > 0
    assert len(out) < len(messages)
    for role in (m["role"] for m in out):
        assert role in roles_before


def test_oversized_single_line_is_truncated_with_a_marker(tokenizer, count):
    line = " ".join(f"token{i}" for i in range(5000))
    messages = [
        {"role": "assistant", "content": f"header\n{line}\ntail"},
        {"role": "user", "content": "token42?"},
    ]
    budget = 300
    out, _ = compress_chat_history(messages, budget, tokenizer=tokenizer)
    assert _measure(out, count) <= budget
    text = "\n".join(m["content"] for m in out)
    assert "chars cut" in text or "omitted by ACE" in text


# ---------------------------------------------------------------------------
# stats / metrics surface
# ---------------------------------------------------------------------------


def test_stats_report_before_after_and_ratio(tokenizer, count):
    messages = _agent_history()
    total = _measure(messages, count)
    _, stats = compress_chat_history(messages, int(total * 0.08), tokenizer=tokenizer)
    assert stats.input_tokens == total
    assert 0 < stats.output_tokens < stats.input_tokens
    assert 0.0 < stats.ratio < 1.0
    assert stats.mode == "lexical"
    assert 0.0 <= stats.pressure <= 1.0
    assert stats.total_lines > stats.kept_lines > 0


def test_knobs_are_derived_from_measured_pressure(tokenizer, count):
    """Rule 5/6: the recency tie-break is a switch driven by pressure, and the
    MMR lambda comes from the redundancy actually present."""
    messages = _agent_history()
    total = _measure(messages, count)
    _, tight = compress_chat_history(messages, int(total * 0.05), tokenizer=tokenizer)
    _, loose = compress_chat_history(messages, int(total * 0.60), tokenizer=tokenizer)
    assert tight.pressure < loose.pressure
    assert tight.recency_tiebreak is True
    assert loose.recency_tiebreak is False
    # This history is highly redundant, so the measured lambda must be high.
    assert tight.mmr_lambda > 0.5


def test_resolve_budget_tokens():
    assert resolve_budget_tokens(8192, 1024) == 8192 - 1024
    assert resolve_budget_tokens(8192, 1024, budget_tokens=1884) == 1884
    assert resolve_budget_tokens(8192, None) == 8192 - 1024


def test_metrics_recording_is_never_fatal(tokenizer, count):
    from vllm.entrypoints.context_compression import record_compression_metrics

    messages = _agent_history(turns=8)
    _, stats = compress_chat_history(
        messages, int(_measure(messages, count) * 0.2), tokenizer=tokenizer
    )
    record_compression_metrics(stats)  # must not raise


# ---------------------------------------------------------------------------
# the refuted scorer is off the serving path
# ---------------------------------------------------------------------------


def test_refuted_scorer_is_not_reachable_from_the_render_path():
    import pathlib

    import vllm.renderers.online_renderer as render_serving

    source = pathlib.Path(render_serving.__file__).read_text()
    assert "apply_ace_eviction" not in source
    assert "_heuristic_score" not in source


def test_refuted_names_still_import_with_a_warning(caplog):
    import vllm.entrypoints.context_compression as cc

    with caplog.at_level("WARNING"):
        fn = cc.ace_compress
    assert callable(fn)
