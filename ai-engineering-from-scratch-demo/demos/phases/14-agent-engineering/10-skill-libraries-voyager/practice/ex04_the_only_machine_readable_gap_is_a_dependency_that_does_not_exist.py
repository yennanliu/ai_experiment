"""Exercise 4 — the only machine-readable gap is a dependency that does not
exist.

    Add a "curriculum" agent: given the current library and a domain
    description, propose 5 missing skills. Call it weekly.

Reading of the exercise: a proposal needs evidence, and the library offers
exactly two kinds. Dangling `depends_on` names are gaps the library states
about itself; everything else has to come from the domain description the
caller supplies. What the library cannot offer is the third kind an agent
would most want -- what was tried and failed -- and that absence is the
measurement.

**ANSWER: five proposals, ranked with the library's own evidence first.**
Over a **6**-skill library and an **8**-capability domain, the agent proposes
**5**: the **2** dangling dependencies other skills already name, then the
**3** highest-priority uncovered capabilities. Every proposal is a name no
skill in the library has.

**FINDING: a dangling dependency is the only self-reported gap, and it
surfaces at run time.** `topo_order` returns the unknown name without
complaint and `execute` only notices on arrival, so today those **2** gaps
are discoverable by running every skill and reading the logs. Reading
`depends_on` directly turns a run-time failure into a weekly report.

**FINDING: the library cannot say what failed.** `Skill` has **8** fields
and **0** of them record an attempt, and `execute` puts `failed` into a
context dictionary the caller throws away. A weekly agent therefore proposes
the same skill every week until someone writes it, with no way to know it was
attempted and abandoned on Tuesday.

**FINDING: duplicate detection is by name, because retrieval reads prose.**
Proposing `extract_minerals` against a library containing `mine_ore` passes
the name check, and `search("extract minerals")` returns **0** candidates:
the two share no token, and tokens are all the retrieval has. A synonym is
proposed as missing, and the library grows a second skill for the same job.

Structure: `propose()` reads `depends_on` and the domain list; `duplicate()`
asks the lesson's own `search` whether a proposal is already covered.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "10-skill-libraries-voyager"
DOMAIN = (
    ("smelt_ore", 9), ("build_shelter", 8), ("find_water", 7), ("trade_goods", 4),
    ("tame_animal", 3), ("mine_ore", 9), ("gather_sticks", 6), ("place_table", 5),
)
LIBRARY = (
    ("mine_ore", "mine iron ore from nearby rock formations", ()),
    ("gather_sticks", "gather sticks from tree or broken planks", ()),
    ("place_table", "place a crafting table at current position", ()),
    ("craft_pickaxe", "craft an iron pickaxe from ore and sticks",
     ("mine_ore", "gather_sticks", "smelt_ore")),
    ("build_camp", "set up a camp for the night",
     ("place_table", "build_shelter")),
    ("explore_cave", "explore a nearby cave system", ("mine_ore",)),
)


def build(ref):
    lib = ref.SkillLibrary()
    for name, description, deps in LIBRARY:
        lib.register(ref.Skill(name=name, description=description, code=f"{name}()",
                               fn=lambda ctx: "ok", depends_on=deps))
    return lib


def dangling(lib):
    """Gaps the library states about itself."""
    known = set(lib.list_names())
    return sorted({dep for name in known for dep in lib.get(name).depends_on
                   if dep not in known})


def propose(lib, domain, count=5):
    known = set(lib.list_names())
    gaps = dangling(lib)
    uncovered = [name for name, _ in sorted(domain, key=lambda row: -row[1])
                 if name not in known and name not in gaps]
    return (gaps + uncovered)[:count]


def duplicate(lib, proposal):
    """What the shipped retrieval says about a proposed name."""
    query = proposal.replace("_", " ")
    return [skill.name for _, skill in lib.search(query, top_k=3)]


def run_to_find_gaps(lib):
    """The gaps a weekly agent would find by executing everything instead."""
    found = []
    for name in lib.list_names():
        context = lib.execute(name)
        found += [line.split(": ")[1] for line in context["log"]
                  if line.startswith("missing skill")]
    return sorted(set(found))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    lib = build(ref)
    proposals = propose(lib, DOMAIN)
    gaps = dangling(lib)
    failed = lib.execute("craft_pickaxe")
    return {
        "library": len(lib.list_names()), "domain": len(DOMAIN),
        "proposals": proposals, "gaps": gaps,
        "novel": sum(1 for name in proposals if lib.get(name) is None),
        "by_running": run_to_find_gaps(lib),
        "skill_fields": list(ref.Skill.__dataclass_fields__),
        "failure_fields": [f for f in ref.Skill.__dataclass_fields__
                           if "fail" in f or "attempt" in f],
        "execute_reports": sorted(k for k in failed if k in ("failed", "log")),
        "synonym_hits": duplicate(lib, "extract_minerals"),
        "synonym_known": lib.get("extract_minerals") is None,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: five proposals, the library's own evidence first",
            all([result["proposals"] == ["build_shelter", "smelt_ore", "find_water",
                                         "trade_goods", "tame_animal"],
                 result["gaps"] == ["build_shelter", "smelt_ore"],
                 result["novel"] == 5, result["library"] == 6,
                 result["domain"] == 8]),
            f"over a {result['library']}-skill library and a {result['domain']}-"
            f"capability domain the agent proposes {result['proposals']}: the "
            f"{len(result['gaps'])} dangling dependencies {result['gaps']} first, then "
            f"the highest-priority uncovered capabilities. All {result['novel']} are "
            "names no skill in the library has",
        ),
        practice.Check(
            "FINDING: a dangling dependency is the only self-reported gap",
            all([result["by_running"] == result["gaps"],
                 len(result["by_running"]) == 2]),
            f"executing every skill and reading the logs finds "
            f"{result['by_running']} -- the same {len(result['gaps'])} gaps that reading "
            "depends_on finds directly. Today they are discoverable only by running "
            "the library; a weekly report can read them off the declarations",
        ),
        practice.Check(
            "FINDING: the library cannot say what failed",
            all([len(result["skill_fields"]) == 8, result["failure_fields"] == [],
                 result["execute_reports"] == ["failed", "log"]]),
            f"Skill carries {len(result['skill_fields'])} fields and "
            f"{len(result['failure_fields'])} of them record an attempt, while execute "
            f"reports {result['execute_reports']} in a context dictionary the caller "
            "throws away. The agent proposes the same skill every week with no way to "
            "know it was attempted and abandoned",
        ),
        practice.Check(
            "FINDING: duplicate detection is by name, because retrieval reads prose",
            all([result["synonym_hits"] == [], result["synonym_known"] is True]),
            f"proposing 'extract_minerals' against a library containing 'mine_ore' "
            f"passes the name check ({result['synonym_known']}) and "
            f"search('extract minerals') returns "
            f"{result['synonym_hits']} -- no candidates at all. A synonym is proposed as "
            "missing, and the library grows a second skill for the same job",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
