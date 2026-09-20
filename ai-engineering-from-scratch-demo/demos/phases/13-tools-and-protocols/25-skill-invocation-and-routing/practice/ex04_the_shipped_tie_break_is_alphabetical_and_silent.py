"""Exercise 4 — the shipped tie-break is alphabetical and silent.

    Add an ambiguity margin between the top two routing scores. Return `ask`
    when the margin is too small.

Reading of the exercise: a margin needs two scores, so the first thing the
rule has to decide is what a one-candidate catalog means -- and the only
defensible answer is that an unopposed winner is not ambiguous, which makes
`ask` a statement about the runner-up rather than about confidence. Building
that then shows what the shipped code does with the same situation, which is
to pick one and say nothing.

**ANSWER: `ask` when the top two are within the margin, `activate`
otherwise, and never `ask` with one candidate.** Two skills scoring
**0.3125** and **0.2353** are **0.0772** apart and return `ask`; a query
naming the verb one of them owns returns `activate` at a margin of
**0.1889**. A single eligible skill returns `activate` at every margin,
because there is no second score to be close to.

**FINDING: the shipped tie-break is alphabetical and silent.**
`max(scored, key=lambda item: (item[0], item[1].name))` breaks an exact tie
by the later name, so two skills at **0.5** each resolve to
`release-review` over `release-audit` with `activated=True` and a reason
that mentions only the eligibility policy. The caller cannot distinguish a
confident match from a coin toss that spelling won.

**FINDING: an absolute margin is the wrong shape at low scores.** The same
**0.03** gap is **8%** of a 0.36 score and **60%** of a 0.05 one. At the demo
threshold of 0.15 both of those pairs are routable, so one absolute margin
either asks constantly near the floor or never asks near the top.

**FINDING: `ask` has nowhere to live in the return type.**
`InvocationDecision` carries `activated: bool`, so a third outcome has to be
smuggled into `mode` and read by a caller that knows to look. **3** outcomes,
**2** states, and the field that distinguishes them is a string.

Structure: `route_with_margin()` wraps the shipped scoring rather than
replacing it, so the two differ only in what they do with the runner-up.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "25-skill-invocation-and-routing"
MARGIN = 0.1


def catalog(ref):
    """Two skills whose descriptions differ by one verb, so ties are reachable."""
    return (ref.SkillMetadata("release-audit",
                              "Audit merged pull request summaries before a release."),
            ref.SkillMetadata("release-review",
                              "Review merged pull request summaries before a release."))


def ranked(ref, skills, query, adapter, actor):
    eligible = [skill for skill in sorted(skills, key=lambda item: item.name)
                if adapter.allows(skill, actor, None)[0]]
    scored = sorted(((round(ref.relevance_score(query, skill), 4), skill.name)
                     for skill in eligible), reverse=True)
    return scored


def route_with_margin(ref, skills, query, adapter, margin=MARGIN, actor=None):
    """The shipped ranking, plus one decision about the runner-up."""
    actor = actor or ref.Actor.MODEL
    scored = ranked(ref, skills, query, adapter, actor)
    if not scored:
        return {"mode": "deny", "skill": None, "score": 0.0, "margin": None}
    top, runner_up = scored[0], scored[1] if len(scored) > 1 else None
    if top[0] < adapter.policy.model_threshold:
        return {"mode": "deny", "skill": top[1], "score": top[0], "margin": None}
    gap = None if runner_up is None else round(top[0] - runner_up[0], 4)
    if gap is not None and gap < margin:
        return {"mode": "ask", "skill": None, "score": top[0], "margin": gap,
                "between": sorted([top[1], runner_up[1]])}
    return {"mode": "activate", "skill": top[1], "score": top[0], "margin": gap}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    adapter = ref.CorePolicyAdapter(ref.InvocationPolicy(model_threshold=0.15))
    skills = catalog(ref)
    close = "merged pull request summaries with an audit trail for the quarterly board pack"
    clear = "audit the merged pull request summaries before a release"
    tied = "merged pull request summaries"

    ambiguous = route_with_margin(ref, skills, close, adapter)
    decided = route_with_margin(ref, skills, clear, adapter)
    single = route_with_margin(ref, (skills[0],), close, adapter)
    shipped = ref.route_request(skills, ref.InvocationRequest(ref.Actor.MODEL, tied), adapter)
    tie_scores = ranked(ref, skills, tied, adapter, ref.Actor.MODEL)
    return {
        "ambiguous": ambiguous, "decided": decided, "single": single,
        "close_scores": ranked(ref, skills, close, adapter, ref.Actor.MODEL),
        "tie_scores": tie_scores, "tied": tie_scores[0][0] == tie_scores[1][0],
        "shipped_activated": shipped.activated, "shipped_skill": shipped.skill_name,
        "shipped_reason": shipped.reason, "shipped_score": shipped.score,
        "alphabetical": max(name for _, name in tie_scores),
        "relative_high": round(0.0303 / 0.3636, 2), "relative_low": round(0.03 / 0.05, 2),
        "threshold": adapter.policy.model_threshold,
        "decision_fields": [name for name in
                            vars(ref.InvocationDecision)["__dataclass_fields__"]],
        "outcomes": sorted({ambiguous["mode"], decided["mode"], single["mode"], "deny"}),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: ask inside the margin, activate outside it, never ask with one candidate",
            all([result["ambiguous"]["mode"] == "ask",
                 result["ambiguous"]["margin"] == 0.0772,
                 result["ambiguous"]["between"] == ["release-audit", "release-review"],
                 result["decided"]["mode"] == "activate",
                 result["decided"]["margin"] >= MARGIN,
                 result["single"]["mode"] == "activate",
                 result["single"]["margin"] is None]),
            f"two skills at {result['close_scores'][0][0]} and "
            f"{result['close_scores'][1][0]} are {result['ambiguous']['margin']} apart and "
            f"return ask between {result['ambiguous']['between']}; a query favouring one "
            f"returns activate at a margin of {result['decided']['margin']}. One eligible "
            "skill activates at every margin, because there is no second score",
        ),
        practice.Check(
            "FINDING: the shipped tie-break is alphabetical and silent",
            all([result["tied"], result["shipped_activated"],
                 result["shipped_skill"] == result["alphabetical"],
                 result["shipped_skill"] == "release-review",
                 "policy" in result["shipped_reason"]]),
            f"the two skills score {result['tie_scores'][0][0]} each and max(..., "
            f"key=(score, name)) hands the tie to {result['shipped_skill']!r}, activated, "
            f"with the reason {result['shipped_reason']!r} -- the eligibility policy and "
            "nothing about the tie. A confident match and a coin toss won on spelling look "
            "identical to the caller",
        ),
        practice.Check(
            "FINDING: an absolute margin is the wrong shape at low scores",
            all([result["relative_high"] == 0.08, result["relative_low"] == 0.6,
                 result["threshold"] == 0.15]),
            f"the same 0.03 gap is {result['relative_high']:.0%} of a 0.36 score and "
            f"{result['relative_low']:.0%} of a 0.05 one. Both pairs clear a "
            f"{result['threshold']} threshold, so one absolute margin either asks "
            "constantly near the floor or never asks near the top -- the rule has to be "
            "relative or it has to be two rules",
        ),
        practice.Check(
            "FINDING: ask has nowhere to live in the return type",
            all(["activated" in result["decision_fields"],
                 "mode" in result["decision_fields"], len(result["outcomes"]) == 3]),
            f"InvocationDecision carries {result['decision_fields']}, and activated is a "
            f"bool. The {len(result['outcomes'])} outcomes {result['outcomes']} have to be "
            "smuggled through mode and read by a caller that knows to look -- three "
            "outcomes, two states, and a string doing the distinguishing",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
