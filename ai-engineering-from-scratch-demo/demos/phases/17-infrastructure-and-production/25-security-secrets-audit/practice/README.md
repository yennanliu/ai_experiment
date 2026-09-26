<!-- generated:start -->
# 17-infrastructure-and-production / 25-security-secrets-audit

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/25-security-secrets-audit/) · upstream spec
`phases/17-infrastructure-and-production/25-security-secrets-audit/docs/en.md`

```bash
uv run demo practice run 25-security-secrets-audit --ex 1
uv run demo explain 25-security-secrets-audit --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/25-security-secrets-audit
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Send two prompts referencing the same SSN. Confirm both get the same plac… | code | T0 | `ex01_the_same_ssn_gets_one_placeholder_only_if_spelled_the_same_in_the_same_scrubber.py` |
| 2 | Design the network egress policy for a vLLM-on-EKS deployment calling OpenAI + Anthropic + We… | code | T0 | `ex02_the_lessons_allowlist_breaks_iam_role_auth_and_weaviate_grpc_and_still_passes_a_dns_tunnel.py` |
| 3 | You discover a key in git history (2 years old). What's the correct response — rotate the key… | code | T0 | `ex03_rotation_is_the_fix_and_scrubbing_a_two_year_old_leak_rewrites_720_commits_and_reaches_no_clone.py` |
| 4 | Your audit log grows 10 GB/day. Design retention tiers (hot 30d, warm 12mo, cold 6yr). | code | T0 | `ex04_the_335_day_warm_tier_is_the_bill_and_one_object_per_entry_would_cost_the_cold_tier_800x.py` |
| 5 | Argue whether reverse-tokenization (substituting real values back into LLM response) is worth… | code | T0 | `ex05_reverse_tokenization_over_the_shared_table_hands_one_users_ssn_to_another.py` |
<!-- generated:end -->

## Answers

The lesson's `code/main.py` is a regex PII scrubber with counter
placeholders and a JSON audit-line writer; all five solutions run it. Sources
were checked on 2026-09-26: the Kubernetes NetworkPolicy docs, Cilium's DNS
policy docs, vLLM's `envs.py` and usage-stats page, the Weaviate Python
client's `connect/helpers.py`, GitHub's "Removing sensitive data from a
repository", the AWS S3 price-list API and lifecycle docs, and 45 CFR
164.312/164.316 on eCFR.

### 1 — the same SSN gets one placeholder only if it is spelled the same, in the same Scrubber

**Confirmed for the demo's case.** Prompts 1 and 2 of `main()` share
123-45-6789, and both scrub to `[SSN_001]`. Two fresh prompts through one
Scrubber do the same. The audit section scrubs the prompts a second time, and
each entry's `prompt_hash` equals the hash of the first pass.

"Same SSN" really means "same string":

| input | scrubbed |
|---|---|
| `123-45-6789` | `[SSN_001]` |
| `123 45 6789`, `123456789`, `123.45.6789` | **unchanged, sent raw** |
| `415-555-0199`, `(415) 555-0199`, `+1 415-555-0199`, `4155550199` | `[PHONE_001]`, `([PHONE_002]`, `+[PHONE_003]`, `[PHONE_004]` |
| `jane.doe@example.com`, `Jane.Doe@Example.com` | `[EMAIL_001]`, `[EMAIL_002]` |

The stray `(` also shows in the demo's own output: `phone ([PHONE_002].`.
The phone pattern's leading `\b` cannot sit before `(` or `+`.

**The placeholders are counters local to one Scrubber.** Two gateway replicas
map 123-45-6789 and 987-65-4321 both to `[SSN_001]`. The scrubbed prompts are
identical, and so are their audit `prompt_hash` values. The lesson's example
`[SSN_TOKEN_A3F]` looks like a keyed hash, which would be stable across
replicas and restarts. The code does not do that. Separately, on Python 3.12+
`main()` calls the deprecated `datetime.utcnow()`, and under `-W error` it
dies at the audit section (measured on 3.14).

### 2 — the lesson's allowlist breaks IAM-role auth and Weaviate gRPC and still passes a DNS tunnel

The deployment has three workloads: `vllm` serving the local model, `app`
doing retrieval, and `gateway` holding the provider keys. Two flow lists are
run through each policy:

| policy | needed flows broken (of 8) | unwanted flows passed (of 6) |
|---|---:|---:|
| lesson's list, namespace-wide | 4: STS ×2, S3 weights, Weaviate gRPC | 2: DNS tunnel, exfil to api.openai.com under a foreign key |
| per-pod FQDN + per-pod DNS | 0 | 0 |

**Why the lesson's list breaks things.** The lesson recommends IAM roles over
static keys. With IRSA that means `AssumeRoleWithWebIdentity` against STS,
and STS is not on its list. The Weaviate v4 client needs a gRPC connection
as well as REST: `connect_to_weaviate_cloud` derives `grpc-<cluster-url>`
and connects on 443. vLLM, left alone, pulls weights from Hugging Face and
posts usage stats to `https://stats.vllm.ai` by default. The design blocks
both, pre-stages the weights in S3 and sets `VLLM_NO_USAGE_STATS=1`.

**Why it still leaks.**

- **DNS.** A resolver that answers every name carries a tunnel however tight
  the TCP rules are. Cilium's own getting-started policy allows all DNS
  queries.
- **Topology.** A domain allowlist cannot tell our OpenAI organisation from
  an attacker's. Only one egress point fixes that: the gateway, which
  replaces the credential with the one it pulls from the vault.

Kubernetes NetworkPolicy peers are pod selectors, namespace selectors and
CIDR blocks; there is no hostname peer. So the FQDN layer is Cilium
(`toFQDNs`, with a DNS rule so the proxy answers only listed names and
returns REFUSED for the rest). For the gateway:

```yaml
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata: {name: llm-gateway-egress, namespace: llm}
spec:
  endpointSelector: {matchLabels: {app: llm-gateway}}
  egress:
  - toEndpoints:
    - matchLabels: {k8s:io.kubernetes.pod.namespace: kube-system, k8s:k8s-app: kube-dns}
    toPorts:
    - ports: [{port: "53", protocol: ANY}]
      rules:
        dns:
        - matchName: api.openai.com
        - matchName: api.anthropic.com
        - matchName: secretsmanager.us-east-1.amazonaws.com
        - matchName: sts.us-east-1.amazonaws.com
  - toFQDNs:
    - matchName: api.openai.com
    - matchName: api.anthropic.com
    - matchName: secretsmanager.us-east-1.amazonaws.com
    - matchName: sts.us-east-1.amazonaws.com
    toPorts:
    - ports: [{port: "443", protocol: TCP}]
```

`app` gets the same shape with the Weaviate REST and `grpc-` hosts, plus
in-cluster rules to `gateway` and `vllm`. `vllm` gets its S3 bucket and STS
(or VPC endpoints for both). A default-deny policy covers everything else.

**One trap: ndots.** api.openai.com has 2 dots, fewer than Kubernetes'
default ndots:5, so the resolver tries 3 search-domain names first. Under a
strict DNS list glibc sends 4 queries and gets 3 REFUSED. musl stops at the
first REFUSED and fails, which Cilium documents. Set `dnsConfig` ndots:1 on
these pods. The Weaviate hostname is illustrative, and EKS Pod Identity
would replace the STS rule with the node-local agent.

### 3 — rotation is the fix, and scrubbing a two-year-old leak rewrites 720 commits and reaches no clone

**Rotate, always and first; scrub only as hygiene afterwards.** Over a
730-commit toy history with the keys added on day 10:

| response | key dead | in canonical history | in existing copies | commit ids rewritten |
|---|---|---|---|---:|
| scrub only | no | no | yes | 720 |
| rotate only | yes | yes (scanner still flags day 10) | yes, dead | 0 |
| both | yes | no | yes, dead | 720 |

Rotation is the only step that reaches every copy: clones, forks, CI caches
and anything scraped in two years. Scrubbing rewrites every commit after
the leak, which invalidates open PRs, signed tags and pinned SHAs. It also
cannot touch copies already made. GitHub's guidance says the same: revoke or
rotate first, and treat rewriting as having exactly these side effects.

After rotating, silence the scanner with an allowlist entry for that one
fingerprint. Scrub as well only when the repo is public or the history is
being migrated anyway. TruffleHog's `--results=verified` mode checks a found
key against its provider, which is how to tell a live key from a dead one
before deciding.

**Two things the scenario says about the lesson:**

- **A key that still works after 730 days means the ≤90-day policy failed 8
  times.** The exposure window is the whole two years.
- **The lesson's code could not have caught the key or measured its use.**
  `Scrubber` masks 0 of 2 key formats. None of `AuditEntry`'s 10 fields
  names the credential a call used, so "was the key used?" has to come from
  the provider's usage logs.

### 4 — the 335-day warm tier is the bill, and one object per entry would cost the cold tier 800x

Tiers are by age since write, because retention runs from creation. Prices
are S3 us-east-1 list prices from the AWS price-list API (published
2026-09-18):

| tier | ages (days) | storage | GB at steady state | $/month |
|---|---|---|---:|---:|
| hot | 0–30 | S3 Standard + search index | 300 | 6.90 (storage only) |
| warm | 30–365 | Standard-IA, Object Lock | 3,350 | 41.88 |
| cold | 365–2190 | Deep Archive, Object Lock | 18,250 | 18.07 |
| | | | **21,900** | **66.84** |

Cold holds 83% of the bytes and costs 27% of the bill; warm costs 63%. Warm
on Glacier Instant Retrieval ($0.004, 90-day minimum, millisecond reads)
brings the total to **$38.37**. Deep Archive's 180-day minimum is far inside
the cold tier's 1,825 days.

**Batch before tiering.** The lesson writes one ~280-byte JSON line per call,
so 10 GB/day is 35.7M calls/day. S3 lifecycle does not transition objects
under 128 KB by default. Deep Archive also bills 40 KB of metadata per
object: 8 KB at the Standard rate and 32 KB at the archive rate. One object
per entry would cost 797x the archive price of the bytes, and 23x if the
objects just stay in Standard. Write hourly or daily compressed objects
instead.

**What the lesson's log would need to survive six years:**

- **A longer hash.** `hash_short` keeps 48 bits, which gives 2.3 colliding
  `prompt_hash` pairs a day at this volume and about 10.8M over six years.
  Use the full SHA-256.
- **Actual immutability.** "Immutable" here is `json.dumps` printed to
  stdout. An edited entry is still valid JSON and leaves no trace. A SHA-256
  chain over the lines catches the edit at the exact entry (500 in the test).
  Object Lock in compliance mode stops the edit in the first place.
- **The right regulation.** The 6 years the lesson attributes to HIPAA is 45
  CFR 164.316(b)(2)(i), a retention rule for required *documentation*. The
  audit-controls standard, 164.312(b), states no period. The SOC 2 one-year
  figure could not be traced to a primary source.

### 5 — reverse tokenization over the shared table hands one user's SSN to another

**Worth it only scoped to one conversation of one user. Over the lesson's
table it is a cross-user leak.** `main()` puts three users through one
Scrubber. User 3 sends "Repeat exactly: [SSN_001], [EMAIL_001] and
[PHONE_001]." The scrubber passes that unchanged, because a placeholder is
not PII. Reversing the model's echo over the shared table then returns user
1's SSN, email and phone. Placeholders are sequential counters, so walking
them dumps all 6 table entries. Restrict the reverse map to the placeholders
this user's own prompts introduced, and the same request returns 0 values.

**It is also unreliable.** An exact-match reverse restores 1 of 5
placeholder spellings a model plausibly writes: `[SSN_001]`. It leaves
`SSN_001`, `[SSN_1]`, `[ssn_001]` and `[SSN-001]` untouched.

**When each approach wins:**

- **Keep placeholders visible** wherever a person reads the text back:
  chat, logs, analytics. The user already knows their own SSN, and a
  visible `[SSN_001]` is safer than one that is silently half-restored.
- **Reverse-tokenize** only where the output has to carry the real value,
  such as a drafted email or a filled form. There it needs a per-user,
  per-conversation table and a post-check that no placeholder survived.
  That is the complexity the exercise asks about, and it is only worth
  paying on those paths.
