"""Exercise 4 — normalising by impressions does not fix exposure bias.

    Port the fusion scorer to include a `user_feedback` dimension (thumbs-up
    on retrieved records). How do you prevent gaming (the agent only returns
    records it already liked)?

Reading of the exercise: gaming here is a loop, not a lie -- only retrieved
records can be rated, so a rated record outranks an unrated one and stays
retrievable. The test is therefore whether a *better* record that arrives
later can ever reach the top. Recency is weighted to zero so the experiment
does not depend on when it runs; everything else is the lesson's own fusion.

**ANSWER: a fourth term, and the loop it creates.** An incumbent with no
query relevance but high importance scores **0.24** and wins rounds 1-24,
collecting **24** thumbs-up. A better record arrives at round 25 scoring
**0.30** on the base fusion -- and under a naive count it never reaches the
top: **0** of the remaining **26** rounds.

**FINDING: normalising by impressions does not help.** The incumbent's rate
is **1.0** because every impression it has ever had ended in a thumbs-up, and
the newcomer's is **0.0** because it has had none. Rate fixes volume bias
between two *rated* records; it cannot fix the bias between a rated record
and an unrated one, and it leaves the lock-in at **0/26**.

**FINDING: capping the term below the merit gap is what works.** The base
gap is **0.06**; with the feedback contribution capped at **0.05** the
newcomer takes the top slot on the round it arrives and holds it for all
**26** remaining rounds. The cap is a statement that feedback may break ties
and may not overturn evidence.

**FINDING: exploration alone fixes discovery, not ranking.** Forcing an
unrated record into the top slot every **5th** round surfaces the newcomer at
round **25** -- and it wins exactly **1** of the remaining **26**. One
impression makes it rated, after which it re-enters a ranking the incumbent's
**24** thumbs-up still dominate. Exploration answers "was it ever seen"; only
the cap answers "may feedback outvote evidence".

Structure: `run()` plays the same 50 rounds under four policies; the only
thing that varies between them is `feedback_term`.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "09-hybrid-memory-mem0"
QUERY = "pipeline throttling incident on friday"
ROUNDS, ARRIVAL, EXPLORE_EVERY, CAP = 50, 25, 5, 0.05
W_FEEDBACK = 0.2
INCUMBENT = ("the user likes concise citation heavy summaries", 0.6)
NEWCOMER = ("the friday queue drained after a throttling event upstream", 0.5)
FILLERS = (("a note about the weekly digest", 0.2),
           ("the sales org asked for a dashboard", 0.2),
           ("an unrelated remark about invoices", 0.2))


def relevance(mem, query, rid):
    for score, record in mem.vector.search(query, top_k=99):
        if record.rid == rid:
            return score
    return 0.0


def feedback_term(policy, ups, impressions):
    if policy == "count":
        return min(1.0, ups / 10)
    if policy == "rate":
        return ups / impressions if impressions else 0.0
    if policy == "capped":
        return min(CAP / W_FEEDBACK, ups / 10)
    return min(1.0, ups / 10)


def build(ref):
    config = ref.Mem0Config(w_relevance=0.6, w_importance=0.4, w_recency=0.0)
    mem = ref.Mem0(config)
    ids = {"incumbent": mem.add(INCUMBENT[0], user_id="ava", importance=INCUMBENT[1])}
    for index, (text, importance) in enumerate(FILLERS):
        ids[f"filler{index}"] = mem.add(text, user_id="ava", importance=importance)
    return mem, config, ids


def base_scores(mem, config, ids):
    return {name: round(config.w_relevance * relevance(mem, QUERY, rid)
                        + config.w_importance * mem.vector._records[rid].importance, 3)
            for name, rid in ids.items()}


def pick(policy, turn, base, ups, seen):
    """Who is returned this round, under one policy."""
    scored = {name: base[name] + W_FEEDBACK * feedback_term(policy, ups[name], seen[name])
              for name in base}
    unrated = [name for name in base if seen[name] == 0]
    if policy == "explore" and turn % EXPLORE_EVERY == 0 and unrated:
        return max(unrated, key=lambda name: base[name])
    return max(scored, key=lambda name: scored[name])


def run(ref, policy):
    mem, config, ids = build(ref)
    ups = {name: 0 for name in ids}
    seen = dict(ups)
    winners = []
    for turn in range(1, ROUNDS + 1):
        if turn == ARRIVAL:
            ids["newcomer"] = mem.add(NEWCOMER[0], user_id="ava", importance=NEWCOMER[1])
            ups["newcomer"], seen["newcomer"] = 0, 0
        top = pick(policy, turn, base_scores(mem, config, ids), ups, seen)
        winners.append(top)
        ups[top], seen[top] = ups[top] + 1, seen[top] + 1
    return {"winners": winners, "base": base_scores(mem, config, ids), "ups": ups,
            "seen": seen}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {policy: run(ref, policy) for policy in ("count", "rate", "capped", "explore")}
    after = {policy: row["winners"][ARRIVAL - 1:] for policy, row in rows.items()}
    counted = rows["count"]
    return {
        "base": counted["base"], "rounds": ROUNDS, "arrival": ARRIVAL,
        "gap": round(counted["base"]["newcomer"] - counted["base"]["incumbent"], 3),
        "ups_before": ARRIVAL - 1,
        "newcomer_wins": {policy: rows_after.count("newcomer")
                          for policy, rows_after in after.items()},
        "remaining": len(after["count"]),
        "rate_incumbent": round(counted["ups"]["incumbent"] / counted["seen"]["incumbent"], 2),
        "rate_newcomer": rows["rate"]["seen"]["newcomer"],
        "first_capped_win": after["capped"].index("newcomer") + ARRIVAL,
        "first_explore_win": after["explore"].index("newcomer") + ARRIVAL,
        "cap": CAP,
    }


def verify(result):
    wins = result["newcomer_wins"]
    return [
        practice.Check(
            "ANSWER: a better record arrives at round 25 and never reaches the top",
            all([result["base"]["incumbent"] == 0.24, result["base"]["newcomer"] == 0.3,
                 wins["count"] == 0, result["remaining"] == 26,
                 result["ups_before"] == 24]),
            f"an incumbent with no query relevance scores "
            f"{result['base']['incumbent']} and wins rounds 1-24, collecting "
            f"{result['ups_before']} thumbs-up. The record that arrives at round "
            f"{result['arrival']} scores {result['base']['newcomer']} on the base fusion "
            f"and wins {wins['count']} of the remaining {result['remaining']} rounds",
        ),
        practice.Check(
            "FINDING: normalising by impressions does not help",
            all([wins["rate"] == 0, result["rate_incumbent"] == 1.0,
                 result["rate_newcomer"] == 0]),
            f"the incumbent's rate is {result['rate_incumbent']} because every impression "
            f"it ever had ended in a thumbs-up, and the newcomer has "
            f"{result['rate_newcomer']} impressions to divide by. Rate fixes volume bias "
            f"between two rated records and leaves the lock-in at {wins['rate']}/"
            f"{result['remaining']}",
        ),
        practice.Check(
            "FINDING: capping the term below the merit gap is what works",
            all([result["gap"] == 0.06, result["cap"] == 0.05,
                 wins["capped"] == 26, result["first_capped_win"] == ARRIVAL]),
            f"the base gap is {result['gap']} and the cap is {result['cap']}, so the "
            f"newcomer takes the top slot on round {result['first_capped_win']} -- the "
            f"round it arrives -- and holds it for all {wins['capped']} remaining rounds. "
            "The cap says feedback may break ties and may not overturn evidence",
        ),
        practice.Check(
            "FINDING: exploration alone fixes discovery, not ranking",
            all([wins["explore"] == 1, result["first_explore_win"] == ARRIVAL,
                 wins["explore"] < wins["capped"]]),
            f"forcing an unrated record into the top slot every {EXPLORE_EVERY}th round "
            f"surfaces the newcomer at round {result['first_explore_win']} and it wins "
            f"{wins['explore']} of the remaining {result['remaining']}. One impression "
            f"makes it rated, after which the incumbent's {result['ups_before']} "
            "thumbs-up dominate again -- exploration fixes discovery, not ranking",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
