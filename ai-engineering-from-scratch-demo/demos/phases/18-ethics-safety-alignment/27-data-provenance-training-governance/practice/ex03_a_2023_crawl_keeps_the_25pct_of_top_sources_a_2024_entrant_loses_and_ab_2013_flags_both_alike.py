"""Exercise 3 — a 2023 crawl keeps the 25% of top sources a 2024 entrant loses, and AB 2013 flags both alike.

    Read the Data Provenance Initiative's "Consent in Crisis" (July 2024).
    Describe the three fastest-restricting content categories and argue one
    economic consequence.

Reading of the exercise: the categories come from the paper, because the
lesson names none. The paper (Longpre, Mahari, Lee et al., arXiv 2407.14933)
finds news sites restricting fastest in robots.txt and Terms of Service. The
head of C4 that is restricting is mostly news, social media / forums and
encyclopedias. The economic argument is made runnable. A 20-source panel of
top sources ([SRC-01]..[SRC-20]) gets a 2023 and a 2024 robots.txt snapshot.
The share that restricts in 2024 is read from the lesson's own "about 25%",
and restrictions name AI crawlers the way the paper describes (GPTBot,
CCBot). The stdlib `urllib.robotparser` then decides who may crawl what.

**ANSWER: news, social media / forums, and encyclopedias. The consequence is
an incumbency moat.** robots.txt is not retroactive. An incumbent that
crawled in 2023 may fetch all 20 panel sources and keeps the copy. A
compliant entrant crawling in 2024 under its own named user-agent may fetch
15. The same 25% a year compounds to 75.0%, 56.2% and 42.2% of the panel
open after 1, 2 and 3 years, a half-life of 2.41 years. Data freshness then
has a price only new entrants pay, as licences, synthetic data, or a crawl
that ignores the signal.

**FINDING: the moat rewards the crawler that is not named.** An entrant
crawling in 2024 as NewAIBot, a user-agent no panel site lists, is allowed on
20 of 20 sources. Restrictions keyed to known crawlers tax the developers
who identify themselves.

**FINDING: the AB 2013 disclosure cannot show the difference.** Summaries of
the incumbent's and the entrant's crawls differ only in item 10 (collection
period), and the reference's `flag_followups` returns the same follow-ups
for both. The lesson says the compliance window is at collection time.
Nothing in the 12 fields records what that window allowed.

**FINDING: the lesson names no content category.** Before its Exercises
section, docs/en.md never says "news", "forum" or "encyclopedia", and it
states the paper's result only as "about 25%" of top sources.

Structure: `panel()` writes the robots.txt snapshots; `open_sources()` asks
robotparser per crawler and year; `crawl_summary()` fills the reference's 12
fields for either crawl.
"""

from __future__ import annotations

import math
import re
import urllib.robotparser

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "27-data-provenance-training-governance"
SOURCES, AI_AGENTS = 20, ("GPTBot", "CCBot")
CATEGORIES = ("news", "forum", "encyclop")


def lesson_rate():
    """The lesson's 'about N%' share of top sources that added a restriction, 2023 -> 2024."""
    return int(re.search(r"about (\d+)% of the top training sources", parity.doc_text(PHASE, LESSON)).group(1)) / 100


def panel(rate):
    """{year: [robots.txt per source]}: in 2024 the first rate*SOURCES sources block AI crawlers."""
    open_file = "User-agent: *\nAllow: /\n"
    blocked = "".join(f"User-agent: {a}\nDisallow: /\n\n" for a in AI_AGENTS) + open_file
    k = round(rate * SOURCES)
    return {2023: [open_file] * SOURCES, 2024: [blocked] * k + [open_file] * (SOURCES - k)}


def open_sources(files, agent):
    count = 0
    for i, text in enumerate(files, 1):
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(text.splitlines())
        count += parser.can_fetch(agent, f"https://src-{i:02d}.example/")
    return count


def crawl_summary(ref, period):
    values = ["top web sources [SRC-01]..[SRC-20]", "general-purpose pretraining", "20 sources",
              "web text; unlabeled", "Y (third-party web text)", "N", "Y", "N",
              "deduplicated to reduce memorisation", period, "2024-09", "N"]
    return dict(zip(ref.AB_2013_FIELDS, values))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rate = lesson_rate()
    snaps = panel(rate)
    inc, ent = crawl_summary(ref, "2023-01 to 2023-06; not ongoing"), crawl_summary(ref, "2024-01 to 2024-06; not ongoing")
    body = parity.doc_text(PHASE, LESSON).split("## Exercises")[0].lower()
    return {
        "rate": rate,
        "incumbent": open_sources(snaps[2023], "GPTBot"),
        "entrant_named": open_sources(snaps[2024], "GPTBot"),
        "entrant_unnamed": open_sources(snaps[2024], "NewAIBot"),
        "compound": [round((1 - rate) ** n, 3) for n in (1, 2, 3)],
        "half_life": round(math.log(0.5) / math.log(1 - rate), 2),
        "diff_items": [i for i, f in enumerate(ref.AB_2013_FIELDS, 1) if inc[f] != ent[f]],
        "flags": (ref.flag_followups(inc), ref.flag_followups(ent)),
        "named_in_lesson": [c for c in CATEGORIES if c in body],
    }


def verify(result):
    inc_flags, ent_flags = result["flags"]
    return [
        practice.Check(
            "ANSWER: news, social media / forums, encyclopedias; the consequence is an incumbency moat",
            (result["rate"], result["incumbent"], result["entrant_named"]) == (0.25, 20, 15)
            and result["compound"] == [0.75, 0.562, 0.422] and result["half_life"] == 2.41,
            f"lesson rate {result['rate']:.0%}; sources open to a 2023 crawl {result['incumbent']}/20, "
            f"to a named 2024 crawl {result['entrant_named']}/20; open share after 1-3 years "
            f"{result['compound']}, half-life {result['half_life']} years",
        ),
        practice.Check(
            "FINDING: the moat rewards the crawler that is not named",
            result["entrant_unnamed"] == SOURCES > result["entrant_named"],
            f"a 2024 crawl as NewAIBot may fetch {result['entrant_unnamed']}/20",
        ),
        practice.Check(
            "FINDING: the AB 2013 disclosure cannot show the difference",
            result["diff_items"] == [10] and inc_flags == ent_flags and len(inc_flags) == 2,
            f"the two summaries differ in items {result['diff_items']}; follow-ups for both: {inc_flags}",
        ),
        practice.Check(
            "FINDING: the lesson names no content category",
            result["named_in_lesson"] == [],
            f"category words in docs/en.md before Exercises: {result['named_in_lesson']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
