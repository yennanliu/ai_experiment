"""Exercise 2 — the lesson's allowlist breaks IAM-role auth and Weaviate gRPC and still passes a DNS tunnel.

    Design the network egress policy for a vLLM-on-EKS deployment calling
    OpenAI + Anthropic + Weaviate.

Reading of the exercise: the deployment is three workloads -- `vllm` serving
the local model, `app` doing retrieval against Weaviate Cloud, and `gateway`
holding the provider keys, as the lesson's AI-gateway pattern puts them. A
design is judged on two lists: the flows the deployment needs, including the
ones the lesson's own advice creates (IAM roles need STS, the vault needs its
endpoint), and the flows it must drop. There is no cluster here, so the policy
is modelled as data and both lists are run through it; the real objects are
CiliumNetworkPolicies, shown in the README.

**ANSWER: per-pod FQDN egress with per-pod DNS -- 0 of 8 needed flows
broken, 0 of 6 unwanted passed.** The DNS proxy answers each pod only for
its own names.
`gateway` alone reaches api.openai.com, api.anthropic.com, Secrets Manager
and STS; `app` reaches the Weaviate REST host and its `grpc-` twin; `vllm`
reaches only its S3 weights bucket and STS, so start-up pulls from
huggingface.co and the default-on usage stats to stats.vllm.ai are dropped.
Kubernetes NetworkPolicy peers are pod selectors, namespace selectors and
CIDR blocks, with no hostname peer, so this needs an FQDN-aware layer
(Cilium `toFQDNs` here).

**FINDING: the lesson's list, applied as written, breaks 4 needed flows.**
One namespace-wide allowlist of "api.openai.com, api.anthropic.com, vector DB
endpoints, vault endpoints" drops STS for both pods (the IAM-role auth the
lesson recommends over static keys), the S3 weights and Weaviate's gRPC
host: the v4 Python client derives `grpc-<cluster>` from the cluster URL and
connects there on 443.

**FINDING: it still passes 2 of the 6 unwanted flows.** DNS to kube-dns
answering every name -- what Cilium's own getting-started policy does --
carries a tunnel however tight the TCP rules are. And a domain allowlist
cannot tell our OpenAI account from an attacker's: any pod that can reach
api.openai.com can post data under a key of its own. Only topology stops
that -- one gateway egresses, and it replaces the credential with the vault's.

**FINDING: a strict DNS allowlist plus ndots:5 breaks musl images.**
ndots:5 is the Kubernetes default. api.openai.com has 2 dots, so the resolver tries 3
search-domain names first. glibc sends 4 queries and succeeds after 3
REFUSED; musl stops at the first REFUSED and fails. Setting ndots:1 in the
pod's `dnsConfig` makes it one query.

Structure: `audit()` runs both flow lists through a policy function;
`lookups()` walks the resolver search list.
"""

from __future__ import annotations

from harness import practice

WV = "llm-rag.c0.us-east1.gcp.weaviate.cloud"
OPENAI, ANTHROPIC = "api.openai.com", "api.anthropic.com"
STS, SM = "sts.us-east-1.amazonaws.com", "secretsmanager.us-east-1.amazonaws.com"
S3 = "llm-weights.s3.us-east-1.amazonaws.com"
SEARCH = ("llm.svc.cluster.local", "svc.cluster.local", "cluster.local")

# (pod, destination, why) -- the flows the deployment needs
NEEDED = [
    ("gateway", OPENAI, "chat completions"), ("gateway", ANTHROPIC, "messages"),
    ("gateway", SM, "provider keys, pulled per request"), ("gateway", STS, "IRSA web identity"),
    ("app", WV, "Weaviate REST"), ("app", "grpc-" + WV, "Weaviate v4 client gRPC"),
    ("vllm", S3, "pre-staged weights"), ("vllm", STS, "IRSA web identity"),
]
# (pod, destination, why) -- flows that must be dropped
UNWANTED = [
    ("vllm", "huggingface.co", "default weight pull at start-up"),
    ("vllm", "stats.vllm.ai", "usage stats, on by default"),
    ("app", "attacker.example", "plain exfil"), ("app", "203.0.113.7", "exfil by raw IP"),
    ("app", "c2V.t.attacker.example", "DNS tunnel through the resolver"),
    ("app", OPENAI, "exfil to a provider account holding the attacker's key"),
]
LESSON_LIST = {OPENAI, ANTHROPIC, WV, SM}  # 'api.openai.com, api.anthropic.com, vector DB, vault'
DESIGN = {"gateway": {OPENAI, ANTHROPIC, SM, STS}, "app": {WV, "grpc-" + WV}, "vllm": {S3, STS}}


def lesson_policy(pod, dest):
    """One namespace-wide allowlist of the lesson's hosts; DNS to kube-dns answers anything."""
    return dest in LESSON_LIST, True


def design_policy(pod, dest):
    """Per-pod FQDN egress, and a DNS proxy that refuses names outside the pod's list."""
    allowed = DESIGN.get(pod, set())
    return dest in allowed, dest in allowed


def leaks(policy, pod, dest, why):
    egress, dns = policy(pod, dest)
    return dns if why.startswith("DNS tunnel") else egress


def audit(policy):
    broken = [(p, d) for p, d, _ in NEEDED if not policy(p, d)[0]]
    passed = [(p, why) for p, d, why in UNWANTED if leaks(policy, p, d, why)]
    return broken, passed


def lookups(name, ndots, allowed, stop_on_refused):
    """Resolver search-list walk: (queries sent, refused, resolved)."""
    names = [f"{name}.{s}" for s in SEARCH] if name.count(".") < ndots else []
    queries = refused = 0
    for candidate in names + [name]:
        queries += 1
        if candidate == name and name in allowed:
            return queries, refused, True
        refused += candidate not in allowed
        if stop_on_refused:
            return queries, refused, False
    return queries, refused, False


def solve():
    allow = DESIGN["gateway"]
    return {
        "lesson": audit(lesson_policy), "design": audit(design_policy),
        "glibc5": lookups(OPENAI, 5, allow, False), "musl5": lookups(OPENAI, 5, allow, True),
        "musl1": lookups(OPENAI, 1, allow, True),
    }


def verify(result):
    (lb, lp), (db, dp) = result["lesson"], result["design"]
    return [
        practice.Check(
            "ANSWER: per-pod FQDN egress with per-pod DNS -- 0 of 8 needed flows broken, "
            "0 of 6 unwanted passed",
            (db, dp, len(NEEDED), len(UNWANTED)) == ([], [], 8, 6),
            f"design broke {db}, passed {dp}",
        ),
        practice.Check(
            "FINDING: the lesson's list, applied as written, breaks 4 needed flows",
            sorted(d for _, d in lb) == sorted([STS, STS, S3, "grpc-" + WV]),
            f"blocked {lb}",
        ),
        practice.Check(
            "FINDING: it still passes 2 of the 6 unwanted flows",
            [why.split(" ")[0] for _, why in lp] == ["DNS", "exfil"],
            f"passed {lp}",
        ),
        practice.Check(
            "FINDING: a strict DNS allowlist plus ndots:5 breaks musl images",
            (result["glibc5"], result["musl5"], result["musl1"])
            == ((4, 3, True), (1, 1, False), (1, 0, True)),
            f"(queries, refused, resolved): glibc ndots:5 {result['glibc5']}, musl ndots:5 "
            f"{result['musl5']}, musl ndots:1 {result['musl1']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
