"""Exercise 4 — CAISI keeps the testing-contact role but deletes "pre-deployment" and "safety", and 3 of 6 duties name foreign actors.

    CAISI's "pro-growth" framing is a departure from the 2022-2024 AI safety
    institute model. Identify two measurable policy shifts that would follow
    from this framing.

Reading of the exercise: a shift is measurable only against a baseline, so
both mandates are measured as written. The baseline is the US AISI *Strategic
Vision* (NIST, 21 May 2024): its seven project bullets, its point-of-contact
sentence and its risk-scope sentence. The departure is Secretary Lutnick's
CAISI statement (Commerce, 3 June 2025): its six "CAISI will" duties, its
point-of-contact sentence and its risk-scope sentence. Both are US government
works, quoted verbatim and read 2026-09-27. Each shift is stated as a metric
that can be tracked on CAISI's future outputs, and first measured on the
mandate text itself.

**ANSWER, shift 1: evaluations move to national-security risks and foreign
models.** Metric: the share of CAISI's published evaluations that cover
foreign-developed models, and the share that cover rights harms (bias,
privacy). In the mandates, duties naming foreign or adversary actors go from
0 of 7 to 3 of 6. The risk scope goes from three classes (individual rights,
national security, public safety) to "demonstrable risks" whose three
examples (cyber, bio, chemical) are all national security. "Rights" appears 0
times in the 2025 text, against once in the 2024 scope sentence.

**ANSWER, shift 2: the international role flips from building a safety
network to resisting foreign rules.** Metric: US sign-ons to multilateral AI
safety statements, and US positions in standards bodies. The 2024 bullet
reads "Lead an inclusive, international network on the science of AI
safety". The 2025 duty is to "guard against burdensome and unnecessary
regulation of American technologies by foreign governments". Bullets naming
a network go from 1 of 7 to 0 of 6, and duties naming regulation from 0 of 7
to 1 of 6. This is the lesson's "domestic counterweight to EU AI Act's
regulatory posture", now written into the mandate.

**FINDING: the lesson's "reduced emphasis on pre-deployment evaluation" is a
word deletion inside a testing role that survives.** The point-of-contact
role stays, but its sentence loses "pre-deployment", "post-deployment" and
"safety", and gains "industry's" and "commercial". Evaluation is not cut: 4 of 6 duties mention
it, against 4 of 7 bullets in 2024. "Safety" falls from 4 of 7 bullets to 0
of 6 duties.

Structure: `V2024`/`V2025` hold the quoted mandates; `measure()` computes the
same counts on each; `lesson_claim()` reads the page's CAISI section.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "24-regulatory-frameworks-eu-us-uk-korea"
V2024 = {  # NIST, US AISI Strategic Vision, 21 May 2024 (project bullets, first sentence each)
    "items": [
        "Perform and coordinate technical research to improve or create needed safety guidelines and "
        "technical safety tools and techniques, such as techniques for detection of synthetic content, "
        "best practices for model security, and technical safeguards and mitigations at the level of "
        "models, systems, and agents.",
        "Conduct pre-deployment TEVV of advanced models, systems, and agents to assess potential and "
        "emerging risks.",
        "Conduct TEVV of advanced AI models, systems, and agents to develop scientific understanding and "
        "documentation of the range of existing risks.",
        "Build and publish specific metrics, evaluation tools, methodological guidelines, protocols, and "
        "benchmarks for assessing risks of advanced AI across different domains and deployment contexts.",
        "Develop and publish risk-based mitigation guidelines and safety mechanisms to support the "
        "responsible design, development, deployment, use, and governance of advanced AI models, systems,"
        " and agents.",
        "Promote adoption of AISI guidelines, evaluations, and recommended AI safety measures and risk "
        "mitigations.",
        "Lead an inclusive, international network on the science of AI safety."],
    "contact": "we will serve as the primary U.S. government point of contact with advanced model developers for"
        " potential pre-deployment and post-deployment AI safety testing.",
    "scope": "evaluations and mitigations for existing harms and potential and emerging risks, including to "
        "individual rights, national security and public safety.",
}
V2025 = {  # Commerce, Lutnick statement on CAISI, 3 June 2025 ("CAISI will:" duties, verbatim)
    "items": [
        "Work with NIST organizations to develop guidelines and best practices to measure and improve the"
        " security of AI systems, and work with the NIST Information Technology Laboratory and other NIST"
        " organizations to assist industry to develop voluntary standards.",
        "Establish voluntary agreements with private sector AI developers and evaluators, and lead "
        "unclassified evaluations of AI capabilities that may pose risks to national security.",
        "Lead evaluations and assessments of capabilities of U.S. and adversary AI systems, the adoption "
        "of foreign AI systems, and the state of international AI competition.",
        "Lead evaluations and assessments of potential security vulnerabilities and malign foreign "
        "influence arising from use of adversaries' AI systems, including the possibility of backdoors "
        "and other covert, malicious behavior.",
        "Coordinate with other federal agencies and entities, including the Department of Defense, the "
        "Department of Energy, the Department of Homeland Security, the Office of Science and Technology "
        "Policy, and the Intelligence Community, to develop evaluation methods, as well as conduct "
        "evaluations and assessments.",
        "Represent U.S. interests internationally to guard against burdensome and unnecessary regulation "
        "of American technologies by foreign governments and collaborate with the NIST Information "
        "Technology Laboratory to ensure U.S. dominance of international AI standards."],
    "contact": "CAISI will serve as industry's primary point of contact within the U.S. Government to facilitate"
        " testing and collaborative research related to harnessing and securing the potential of "
        "commercial AI systems.",
    "scope": "In conducting these evaluations, CAISI will focus on demonstrable risks, such as cybersecurity, "
        "biosecurity, and chemical weapons.",
}
RISK_CLASSES = {"individual rights": r"rights", "public safety": r"public safety",
                "national security": r"national security|cyber|biosecurity|chemical"}
ITEM_TERMS = {"foreign": r"adversar|foreign", "safety": r"safety", "evaluation": r"evaluat|TEVV",
              "network": r"network", "regulation": r"regulat"}


def measure(v):
    text = " ".join(v["items"] + [v["scope"], v["contact"]])
    counts = {k: (sum(bool(re.search(rx, i)) for i in v["items"]), len(v["items"]))
              for k, rx in ITEM_TERMS.items()}
    return {**counts, "classes": [c for c, rx in RISK_CLASSES.items() if re.search(rx, v["scope"])],
            "rights": len(re.findall(r"\brights\b", text)), "contact": "point of contact" in v["contact"]}


def lesson_claim():
    doc = parity.doc_text(PHASE, LESSON)
    section = doc.split("### US CAISI")[1].split("\n### ")[0]
    return (re.search(r"(Reduced emphasis on [^;]+);", section).group(1),
            re.search(r"(Domestic counterweight to [^.]+)\.", section).group(1))


def solve():
    old, new = measure(V2024), measure(V2025)
    ow, nw = (set(re.findall(r"[\w'-]+", v["contact"].lower())) for v in (V2024, V2025))
    return {"old": old, "new": new, "dropped": sorted(w for w in ow - nw if "deploy" in w or w == "safety"),
            "added": sorted(w for w in nw - ow if w in ("commercial", "industry's")),
            "claim": lesson_claim()[0], "counterweight": lesson_claim()[1]}


def verify(result):
    old, new = result["old"], result["new"]
    return [
        practice.Check(
            "ANSWER, shift 1: evaluations move to national-security risks and foreign models",
            (old["foreign"], new["foreign"], old["rights"], new["rights"], old["classes"], new["classes"])
            == ((0, 7), (3, 6), 1, 0, list(RISK_CLASSES), ["national security"]),
            f"duties naming foreign/adversary actors {old['foreign']} -> {new['foreign']}; risk "
            f"classes {old['classes']} -> {new['classes']}; 'rights' {old['rights']} -> {new['rights']}",
        ),
        practice.Check(
            "ANSWER, shift 2: the international role flips from a safety network to resisting rules",
            (old["network"], new["network"], old["regulation"], new["regulation"], result["counterweight"])
            == ((1, 7), (0, 6), (0, 7), (1, 6), "Domestic counterweight to EU AI Act's regulatory posture"),
            f"network {old['network']} -> {new['network']}, regulation {old['regulation']} -> "
            f"{new['regulation']}; lesson: {result['counterweight']!r}",
        ),
        practice.Check(
            "FINDING: the pre-deployment cut is a word deletion inside a surviving testing role",
            (result["claim"], old["contact"], new["contact"], result["dropped"], result["added"],
             old["evaluation"], new["evaluation"], old["safety"], new["safety"])
            == ("Reduced emphasis on pre-deployment evaluation", True, True,
                ["post-deployment", "pre-deployment", "safety"], ["commercial", "industry's"],
                (4, 7), (4, 6), (4, 7), (0, 6)),
            f"lesson: {result['claim']!r}; contact sentence drops {result['dropped']}, adds {result['added']}; "
            f"evaluation {old['evaluation']} -> {new['evaluation']}, safety {old['safety']} -> {new['safety']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
