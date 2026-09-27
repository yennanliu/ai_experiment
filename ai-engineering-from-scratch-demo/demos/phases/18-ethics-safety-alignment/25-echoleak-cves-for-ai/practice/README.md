<!-- generated:start -->
# 18-ethics-safety-alignment / 25-echoleak-cves-for-ai

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/25-echoleak-cves-for-ai/) · upstream spec
`phases/18-ethics-safety-alignment/25-echoleak-cves-for-ai/docs/en.md`

```bash
uv run demo practice run 25-echoleak-cves-for-ai --ex 1
uv run demo explain 25-echoleak-cves-for-ai --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/25-echoleak-cves-for-ai
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Report the exfiltrated data with and without the scope-separation defense. | code | T0 | `ex01_naive_leaks_the_whole_mfa_email_in_the_url_and_the_defense_leaks_nothing_because_it_never_reads_the_inbox.py` |
| 2 | The EchoLeak attack bypasses CSP because it exfiltrates via a Microsoft-signed URL. Design a… | code | T0 | `ex02_dropping_the_signed_proxy_host_stops_the_reference_leak_at_15pct_false_positives.py` |
| 3 | Aim Labs' Scope Violation framework has three boundaries: retrieval, scope, output. Construct… | code | T0 | `ex03_an_integrity_scope_violation_needs_no_output_boundary_so_output_controls_stop_none_of_it.py` |
| 4 | Microsoft's CamoLeak fix disabled image rendering entirely. Propose a partial fix that preser… | code | T0 | `ex04_render_only_provenance_trusted_images_keeps_60pct_of_renders_iff_provenance_is_unforgeable.py` |
| 5 | Responsible disclosure for AI vulnerabilities is evolving. Sketch a disclosure protocol that… | code | T0 | `ex05_the_cvss_calculator_reproduces_the_docs_9_3_and_the_mfa_demonstration_is_a_2_1_point_swing.py` |
<!-- generated:end -->

## Answers

The lesson's `code/main.py` reconstructs the EchoLeak trace: a naive Copilot
that renders an attacker email into a `render_image` call, and a
"scope-separated" defense. Every payload below uses a placeholder instruction
and the reference's own `[hidden:` marker, never a real exploit string. CVSS
figures are computed with a CVSS 3.1 base-score calculator; the EchoLeak metric
vector is from NVD (`nvd.nist.gov`, read 2026-09-27).

### 1 — the naive agent leaks the whole MFA email in the URL; the defense leaks nothing because it never reads the inbox

**Without the defense the exfiltration URL carries `your MFA code: 382914`; with
it, nothing.** The naive agent issues one `render_image` call to
`https://signed.microsoft.com/img?data=your MFA code: 382914` — the entire body
of the security email, label and spaces included, unencoded. The defended agent
issues 0 tool calls and renders only "Summary of attacker@external.example: Hi
team, Q4 update attached." The colleague's "lunch plans" email leaks in neither.

**The hijack is the literal string `[hidden:`, not the instruction.** Swap the
hidden text for a placeholder that never mentions MFA codes and the same code
still leaks; keep the lesson's instruction verbatim but drop the marker and
nothing leaks. `retrieve()` ignores the prompt and returns exactly the one
attacker email; the private inbox and the `signed.microsoft.com` host are
hard-coded in the agent, not taken from the email.

**The defense has no tool call to block and no inbox to read.** Over the four
{marker present/absent} × {trusted/untrusted} cases it leaks 0 every time, while
the naive agent leaks in both marker cases regardless of trust. Its only action
on content is `split("[hidden:")` — an input filter on retrieved content, which
the lesson's own `skill-cve-review.md` lists under "Hard rejects".

### 2 — dropping the signed proxy host stops the reference leak at 15% false positives

Graded on a labelled 100-render fixture of legitimate assistant traffic
(tenant thumbnails, Office CDN icons, web images, and 15 diagrams drawn through
the signed proxy), plus the reference's own leak URL:

| allowlist | legit renders blocked | reference leak URL |
|---|---:|---|
| vendor CSP (domain suffixes) | 0% | allowed |
| exact hosts, no signed proxy | 15% | blocked |
| tenant host only | 60% | blocked |
| no rendering (the CamoLeak fix) | 100% | blocked |

**Narrow the allowlist to exact hosts and drop the signed proxy: 15% cost, leak
blocked.** The vendor CSP lets the leak through because `signed.microsoft.com`
matches the `microsoft.com` suffix. **The leak URL and a legitimate diagram are
the same URL shape** — host, path and query key `('signed.microsoft.com',
'/img', 'data')` are identical, only the data differs — so a destination rule
can only keep or drop the endpoint, and the 15% is exactly the legitimate
traffic on it.

### 3 — an integrity scope violation needs no output boundary, so output controls stop none of it

EchoLeak is retrieval → scope-*read* → output. The fourth class is retrieval →
scope-*write*: untrusted retrieved content makes the agent change privileged
state, modelled abstractly as a placeholder instruction and a boolean "wrote
config" effect (the shape of CVE-2025-53773). Over the 8 on/off combinations of
three independent gates:

| gates on | EchoLeak leaks | integrity attack writes config |
|---|---|---|
| none | yes | yes |
| output only | no | yes |
| any set containing retrieval or scope | no | no |

**The integrity attack passes 2 of 8 gate sets (none, and output-only);
EchoLeak passes only 1 (none).** Because the integrity attack never renders,
an output control — the ex02 allowlist or CamoLeak's no-rendering fix — stops
none of it; only the retrieval or scope gate does. **The reference cannot
express this class:** the only tool in either agent is `render_image` and
neither has a write path, so the trace reconstruction can show confidentiality
leaks only. "All three boundaries must be prevented independently" is really a
claim about the *next* attack class, not this CVE, for which any single gate
suffices.

### 4 — render only provenance-trusted images; it keeps 60% of renders iff provenance is unforgeable

A partial alternative to Microsoft's total no-rendering fix: render an image
only when its URL came from a trusted content channel (system, user, a
first-party CDN), never from retrieved untrusted content. On the same fixture:

| policy | legit rendered | reference leak |
|---|---:|---|
| no rendering (Microsoft's fix) | 0% | blocked |
| provenance-trusted, unforgeable | 60% | blocked |
| provenance-trusted, forgeable | 60% | rendered |

**It keeps 60% of legitimate renders and blocks the leak — but only if
provenance is unforgeable by retrieved content.** The 40% dropped are the
diagrams the agent composes from content, which carry no trusted provenance.
The authentication assumption *is* the fix: with forgeable provenance it admits
exactly what the vendor CSP admits, the leak to `signed.microsoft.com`
included, so provenance must be bound by the platform (a signed capability on
the render call), not asserted in the content — the same trust boundary
EchoLeak crossed.

### 5 — the CVSS calculator reproduces the doc's 9.3, and the MFA demonstration is a 2.1-point swing

A CVSS 3.1 base-score calculator, run on EchoLeak's metric vector
`AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:L/A:N`:

| confidentiality impact | base score |
|---|---:|
| High (demonstrated MFA exfiltration) | 9.3 |
| Low ("information disclosure only") | 7.2 |
| None | 5.8 |

**The calculator reproduces the doc's 9.3 exactly from the vector, and the same
bug rated "information disclosure only" (C:L) scores 7.2.** So the disclosure
protocol's AI-specific evidence — the reproducible MFA-code exfiltration Aim
Labs demonstrated — is worth a 2.1-point High→Critical swing, and that
demonstration is what justifies the 9.3 over the initial lower rating. A
disclosure protocol for an AI CVE should therefore score severity only from a
demonstrated, reproducible proof-of-concept, scoped to the model version tested.

### Three corrections to the lesson text (sourced)

The lesson's front-matter and prose state three things the public records
contradict. Sources read 2026-09-27:

- **CamoLeak has a CVE number and was not found by Aim Labs.** The doc says "CVE
  undisclosed number (Microsoft's choice)". It is **CVE-2025-59145**, CVSS 9.6,
  reported by **Omer Mayraz of Legit Security** (not Aim Labs) via HackerOne in
  June 2025; GitHub disabled Copilot Chat image rendering on 2025-08-14 and it
  was publicly disclosed in October 2025.
  [Legit Security](https://www.legitsecurity.com/blog/camoleak-critical-github-copilot-vulnerability-leaks-private-source-code),
  [The Register](https://www.theregister.com/2025/10/09/github_copilot_chat_vulnerability/).
- **CVE-2025-53773 is a local RCE via workspace-file writes, not a
  "code-suggestion surface" bug.** It is CVSS 7.8 (vector Local), where a
  prompt injection makes Copilot write `"chat.tools.autoApprove": true` into
  `.vscode/settings.json` ("YOLO mode"), leading to command execution; fixed in
  the August 2025 Patch Tuesday.
  [CVE.org](https://www.cve.org/CVERecord?id=CVE-2025-53773),
  [Embrace The Red](https://embracethered.com/blog/posts/2025/github-copilot-remote-code-execution-via-prompt-injection/).
- **EchoLeak's CVSS vector.** The doc quotes only the 9.3 score; the NVD base
  vector is `AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:L/A:N`, which ex05's calculator
  reproduces.
  [NVD CVE-2025-32711](https://nvd.nist.gov/vuln/detail/CVE-2025-32711).
