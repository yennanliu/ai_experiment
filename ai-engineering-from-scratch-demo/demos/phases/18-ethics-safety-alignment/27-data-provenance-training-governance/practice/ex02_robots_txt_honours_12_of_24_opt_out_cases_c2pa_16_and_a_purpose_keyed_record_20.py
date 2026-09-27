"""Exercise 2 — robots.txt honours 12 of 24 opt-out cases, C2PA 16, and a purpose-keyed record 20.

    The EU Copyright Directive TDM opt-out is machine-readable. Propose a
    standard format for the opt-out signal and compare it to robots.txt and
    C2PA "No AI Training."

Reading of the exercise: a format is compared by what it can express and
where it survives, so each signal is run against one rightholder intent,
"no AI training, search indexing is fine, for any crawler, wherever the copy
lives". There are 24 cases: 2 crawlers (one named in robots.txt, one new), 2
purposes (train, search), 2 asset types (an HTML page, an image), and 3
places (origin site, byte-identical mirror, stripped copy: image re-encoded
or text extracted). robots.txt is evaluated by the stdlib
`urllib.robotparser`, C2PA as a purpose-level assertion embedded in media
files only, and the proposal as written below.

**ANSWER: the proposal is one purpose-keyed, crawler-independent record,
published twice.** A site file at `/.well-known/tdmrep.json`
(`{"tdm-reservation": 1, "applies-to": "*", "purposes": {"train":
"reserved", "search-index": "allowed"}, "policy": <licence URL>}`) and the
same record embedded in each asset (`<meta name="tdm-reservation"
content="train=reserved; search-index=allowed">` in HTML, the metadata block
in media). It honours 20 of the 24 cases. robots.txt honours 12 and C2PA 16.
The proposal's 4 misses are the stripped copies asked to train, which no
embedded signal survives.

**FINDING: robots.txt opts a work into training by omission.** With
GPTBot and CCBot disallowed, `robotparser` lets a crawler named NewAIBot
fetch everything. It has no purpose field, so blocking GPTBot to stop
training also blocks GPTBot's search use. On a mirror, the mirror's
robots.txt governs, not the rightholder's.

**FINDING: C2PA is right about purpose and wrong about coverage.** It is
crawler-independent and travels with the file, so it honours all 4 image
cases per crawler at origin and on mirrors. HTML text carries no manifest,
so it misses all 6 HTML train cases, at every place and for both crawlers.

**FINDING: the lesson's generator records no opt-out at all.** None of the
12 AB 2013 fields names robots, opt-out or TDM, and `flag_followups` asks
for opt-out respect only when item 5 starts with "Y". A crawled dataset
declared public domain (item 5 = "N") gets no opt-out check. The lesson
spells the TDMRep signal "TDM.Reservation", and the property is
`tdm-reservation`.

Structure: `by_robots()`, `by_c2pa()` and `by_proposal()` return each signal's
allow/deny for one case;
`score()` counts the cases where that matches the intent.
"""

from __future__ import annotations

import itertools
import json
import re
import urllib.robotparser

from harness import parity, practice

URL = "https://site-a.example/work"
PHASE, LESSON = "18-ethics-safety-alignment", "27-data-provenance-training-governance"
ORIGIN_ROBOTS = "User-agent: GPTBot\nDisallow: /\n\nUser-agent: CCBot\nDisallow: /\n\nUser-agent: *\nAllow: /\n"
MIRROR_ROBOTS = "User-agent: *\nAllow: /\n"
SITE_FILE = json.dumps({"tdm-reservation": 1, "applies-to": "*", "policy": "https://[RIGHTHOLDER]/licence",
                        "purposes": {"train": "reserved", "search-index": "allowed"}})
META_TAG = "train=reserved; search-index=allowed"
C2PA = {"c2pa.ai_training": "notAllowed", "c2pa.data_mining": "allowed"}
CASES = list(itertools.product(("GPTBot", "NewAIBot"), ("train", "search"), ("html", "image"),
                               ("origin", "mirror", "stripped")))


def robots(text):
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(text.splitlines())
    return parser


def parse_meta(content):
    return dict(part.strip().split("=") for part in content.split(";"))


def by_robots(agent, purpose, asset, place):
    text = ORIGIN_ROBOTS if place == "origin" else MIRROR_ROBOTS
    return robots(text).can_fetch(agent, URL)


def by_c2pa(agent, purpose, asset, place):
    carried = asset == "image" and place != "stripped"
    key = "c2pa.ai_training" if purpose == "train" else "c2pa.data_mining"
    return not carried or C2PA[key] == "allowed"


def by_proposal(agent, purpose, asset, place):
    if place == "stripped":
        return True
    record = json.loads(SITE_FILE)["purposes"] if place == "origin" else parse_meta(META_TAG)
    return record["train" if purpose == "train" else "search-index"] == "allowed"


SIGNALS = {"robots.txt": by_robots, "C2PA": by_c2pa, "proposal": by_proposal}


def c2pa_image_hits():
    """C2PA cases honoured for one crawler on images at origin and mirror."""
    cases = [c for c in CASES if c[0] == "NewAIBot" and c[2] == "image" and c[3] != "stripped"]
    return sum(by_c2pa(*c) == (c[1] == "search") for c in cases)


def score(signal):
    misses = [c for c in CASES if SIGNALS[signal](*c) != (c[1] == "search")]
    return len(CASES) - len(misses), misses


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    scores = {s: score(s) for s in ("robots.txt", "C2PA", "proposal")}
    origin = robots(ORIGIN_ROBOTS)
    public = dict(ref.TOY_EXAMPLE, **{ref.AB_2013_FIELDS[0]: "crawled from [SITE-A]"})
    doc = parity.doc_text(PHASE, LESSON)
    return {
        "honoured": {s: v[0] for s, v in scores.items()},
        "proposal_misses": scores["proposal"][1],
        "fetch": {a: origin.can_fetch(a, URL) for a in ("GPTBot", "CCBot", "NewAIBot")},
        "c2pa_img": c2pa_image_hits(),
        "c2pa_html_miss": sum(c[2] == "html" for c in scores["C2PA"][1]),
        "optout_fields": [f for f in ref.AB_2013_FIELDS if re.search("opt|robots|tdm", f.lower())],
        "public_flags": ref.flag_followups(public),
        "spelling": ("TDM.Reservation" in doc, "tdm-reservation" in doc),
    }


def verify(result):
    misses = result["proposal_misses"]
    return [
        practice.Check(
            "ANSWER: one purpose-keyed, crawler-independent record, published twice",
            result["honoured"] == {"robots.txt": 12, "C2PA": 16, "proposal": 20}
            and all(c[1] == "train" and c[3] == "stripped" for c in misses),
            f"cases honoured out of 24: {result['honoured']}; the proposal misses {misses}",
        ),
        practice.Check(
            "FINDING: robots.txt opts a work into training by omission",
            result["fetch"] == {"GPTBot": False, "CCBot": False, "NewAIBot": True},
            f"urllib.robotparser on the origin robots.txt: may fetch {result['fetch']}",
        ),
        practice.Check(
            "FINDING: C2PA is right about purpose and wrong about coverage",
            result["c2pa_img"] == 4 and result["c2pa_html_miss"] == 6,
            f"C2PA honours {result['c2pa_img']} of 4 image cases per crawler at origin and "
            f"mirror; it misses {result['c2pa_html_miss']} HTML cases (every train case)",
        ),
        practice.Check(
            "FINDING: the lesson's generator records no opt-out at all",
            result["optout_fields"] == [] and not any("TDM" in f for f in result["public_flags"])
            and result["spelling"] == (True, False),
            f"opt-out fields among the 12: {result['optout_fields']}; flags for a crawl declared "
            f"public domain: {result['public_flags']}; 'TDM.Reservation' / 'tdm-reservation' in "
            f"the lesson: {result['spelling']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
