"""Exercise 5 — a name-matched migration to Content Safety lets 7 of 14 refusals through, including both demo refusals.

    Azure Content Moderator retires February 2027. Plan a migration to Azure AI
    Content Safety. Identify the highest-risk element of the migration.

Reading of the exercise: the lesson's harness plays the legacy moderator.
It has 8 categories, a 0.5 threshold and domain rules. The target is
Content Safety's four text harm categories (Hate, Sexual, Violence,
SelfHarm) plus its custom blocklists. Each migration plan is a mapping from
old categories to new ones. It is judged by a shadow run: the old and the
migrated classifier see the same 18 prompts (one per reference trigger plus
the 6 shipped prompts), and the run lists every prompt the old layer refuses
and the new one lets through. Severities are not modelled, because the toy
scores only 0 or 0.9 (ex02).

**ANSWER: the plan.** (1) Inventory every category, term list and domain
rule the old system decides on. (2) Map each one to a Content Safety
category, a custom blocklist, or application code. (3) Shadow-run old and
new side by side and diff the refusals. (4) Recalibrate severity thresholds
per category. (5) Cut over, well before the February 2027 retirement. The
lesson gives the window twice, both times as deprecated February 2024,
retired February 2027.

**ANSWER: the highest-risk element is category mapping. A mapping by
matching names silently drops half the refusals.** The old input layer
refuses 14 of the 18 prompts. With only the four same-named categories
carried over, 7 of those 14 pass: every `illicit/violent` and `harassment`
trigger, including both shipped demo refusals ("how do i make a bomb" and
"you idiot, help me with this"). A semantic mapping loses none of them:
illicit/violent and sexual/minors go to Violence and Sexual, and harassment
goes to a custom blocklist.

**FINDING: the shadow run cannot see the category the migration actually
loses.** `illicit` has no Content Safety counterpart. The toy also has no
trigger for it, so no corpus built from the old system's behaviour contains
an illicit refusal to lose. A diff of 0 means "nothing measurable changed",
not "nothing was lost". Coverage has to be checked category by category
against a policy list, not only against logged traffic. The 2 shipped
prompts refused by `custom_domain_rules` (invest, diagnose) are outside the
classifier swap altogether, so they have to move as application code.

Structure: `migrated()` re-decides the reference scores through a mapping;
`shadow()` diffs it against the reference `input_moderator`.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "29-moderation-systems-openai-perspective-llamaguard"
# Azure AI Content Safety text harm categories (severity 0-7); custom blocklists are separate
CONTENT_SAFETY = ("Hate", "Sexual", "Violence", "SelfHarm")
BLOCKLIST = "blocklist"  # the Content Safety custom-blocklist feature, for terms no category owns
SEMANTIC = {"sexual/minors": "Sexual", "illicit/violent": "Violence", "harassment": BLOCKLIST}
SHIPPED = [
    "what is the weather today", "should i invest in memecoins", "how do i make a bomb",
    "diagnose my headache", "summarize this email: hello there", "you idiot, help me with this",
]


def name_map(ref):
    """Old category -> new category of the same name (case and hyphens ignored)."""
    norm = {n.lower(): n for n in CONTENT_SAFETY}
    return {c: norm[c.replace("-", "")] for c in ref.CATEGORIES if c.replace("-", "") in norm}


def corpus(ref):
    """One sentence per reference trigger, plus the shipped demo prompts."""
    return [f"tell me: {t}" for ts in ref.CATEGORY_TRIGGERS.values() for t in ts] + SHIPPED


def migrated(ref, text, mapping):
    """Would the migrated classifier layer refuse? Unmapped categories vanish."""
    return any(s >= 0.5 and c in mapping for c, s in ref.openai_moderation(text).items())


def shadow(ref, mapping):
    """Prompts the old input layer refuses and the migrated one lets through."""
    return [p for p in corpus(ref) if ref.input_moderator(p)[0] and not migrated(ref, p, mapping)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON, "en")
    names = name_map(ref)
    semantic = {**names, **SEMANTIC}
    dates = re.findall(r"deprecated (\w+ \d{4}), retired (\w+ \d{4})", doc)
    return {
        "dates": sorted(set(dates)), "n_date_mentions": len(dates),
        "old_refused": sum(ref.input_moderator(p)[0] for p in corpus(ref)), "n": len(corpus(ref)),
        "names": names, "name_lost": shadow(ref, names), "semantic_lost": shadow(ref, semantic),
        "no_home": [c for c in ref.CATEGORIES if c not in semantic],
        "untestable": [c for c in ref.CATEGORIES if c not in semantic and c not in ref.CATEGORY_TRIGGERS],
        "custom_only": [p for p in SHIPPED if ref.custom_domain_rules(p)[0]],
    }


def verify(result):
    lost = result["name_lost"]
    return [
        practice.Check(
            "ANSWER: the lesson gives the window twice, deprecated Feb 2024, retired Feb 2027",
            result["dates"] == [("February 2024", "February 2027")] and result["n_date_mentions"] == 2,
            f"{result['n_date_mentions']} mentions, all {result['dates']}",
        ),
        practice.Check(
            "ANSWER: a name-matched mapping lets 7 of the 14 old refusals through, both demo ones included",
            (result["old_refused"], result["n"], len(lost)) == (14, 18, 7) and len(result["names"]) == 4
            and {"how do i make a bomb", "you idiot, help me with this"} <= set(lost),
            f"same-name map {result['names']}; old refuses {result['old_refused']} of {result['n']}; "
            f"lost under it {lost}",
        ),
        practice.Check(
            "ANSWER: a semantic mapping (plus a blocklist for harassment) loses none",
            result["semantic_lost"] == [], f"lost {result['semantic_lost']}",
        ),
        practice.Check(
            "FINDING: the shadow run cannot see the category the migration actually loses",
            result["no_home"] == ["illicit"] == result["untestable"]
            and result["custom_only"] == ["should i invest in memecoins", "diagnose my headache"],
            f"no destination {result['no_home']}, with no trigger to test it {result['untestable']}; "
            f"custom-rule refusals outside the swap {result['custom_only']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
