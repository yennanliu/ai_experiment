"""Exercise 4 — the SSH config is sound, but its HostName is a documentation-only address.

    If you have access to a remote machine, set up Remote SSH and open a folder on it

Reading of the exercise: the exercise is conditional on a remote machine, and CI
has none, so it is read as checking the setup the lesson hands over -- the
`~/.ssh/config` block, the `ssh-keygen` / `ssh-copy-id` commands and the Remote
SSH extension -- with stdlib parsers (`ipaddress` for the host), so that a
learner who does have a box knows which lines to keep and which to replace.

**ANSWER: the block is valid and consistent; replace one line.** `Host gpu-box`
carries 4 known `ssh_config` keywords; `IdentityFile ~/.ssh/id_ed25519` is the
file `ssh-keygen -t ed25519` writes by default; Remote SSH is in the
recommendations. Keep it, set `HostName` to the real box, then run
"Remote-SSH: Connect to Host > gpu-box" and "File > Open Folder".

**FINDING: `HostName 203.0.113.50` cannot connect, by design.** It is in
`203.0.113.0/24`, TEST-NET-3 of RFC 5737, reserved for documentation and never
routed (`ipaddress` reports it not global). The doc says that after this block
"`gpu-box` connects instantly" and never says to replace it; a verbatim copy
times out.

**FINDING: `ForwardAgent yes` hands the box your keys while you are connected.**
`ssh_config(5)` warns that anyone who can bypass file permissions on the remote
host can use the forwarded agent. The doc aims this at "cloud VMs, lab servers,
Lambda, Vast.ai" -- rented and shared machines, where root is someone else.

**CONTROL: the other placeholders are reserved names too.** The `ssh-keygen -C`
email is at `example.com` (RFC 2606), so the doc is consistent in using
non-resolvable examples; only the config block lacks a "replace this" note.

Structure: `ssh_block` reads the doc's config; the checks classify each line.
"""

from __future__ import annotations

import ipaddress
import json
import re

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "08-editor-setup"
KEYWORDS = {"hostname", "user", "identityfile", "forwardagent", "port", "proxyjump"}
TEST_NET_3 = ipaddress.ip_network("203.0.113.0/24")
WARNING = "Agent forwarding should be enabled with caution."  # ssh_config(5), ForwardAgent


def ssh_block(doc):
    """Host name and keyword -> value from the doc's `Host ...` config block."""
    block = re.search(r"```\n(Host .*?)```", doc, re.S).group(1)
    lines = [ln.split(None, 1) for ln in block.splitlines() if ln.strip()]
    return lines[0][1], {k.lower(): v for k, v in lines[1:]}


def solve():
    doc = parity.doc_text(PHASE, LESSON)
    host, cfg = ssh_block(doc)
    vscode = parity.lesson_dir(PHASE, LESSON) / "code" / "vscode"
    recommended = json.loads((vscode / "extensions.json").read_text())["recommendations"]
    keygen = re.search(r"ssh-keygen -t (\S+) -C \"([^\"]+)\"", doc)
    addr = ipaddress.ip_address(cfg["hostname"])
    return {
        "host": host,
        "cfg": cfg,
        "unknown": sorted(set(cfg) - KEYWORDS),
        "key_type": keygen.group(1),
        "email_domain": keygen.group(2).split("@")[1],
        "remote_ssh": "ms-vscode-remote.remote-ssh" in recommended,
        "test_net": addr in TEST_NET_3,
        "is_global": addr.is_global,
        "connects_instantly": "connects instantly" in doc,
        "shared_hosts": [h for h in ("lab servers", "Vast.ai", "cloud VMs") if h in doc],
    }


def verify(result):
    cfg = result["cfg"]
    return [
        practice.Check(
            "ANSWER: the block is valid and consistent; only HostName must change",
            result["host"] == "gpu-box"
            and not result["unknown"]
            and cfg["identityfile"] == f"~/.ssh/id_{result['key_type']}"
            and result["remote_ssh"],
            f"Host {result['host']} with {sorted(cfg)}, none unknown; IdentityFile "
            f"{cfg['identityfile']} is ssh-keygen -t {result['key_type']}'s default; "
            "Remote SSH is recommended",
        ),
        practice.Check(
            "FINDING: HostName 203.0.113.50 is TEST-NET-3, never routable",
            result["test_net"] and not result["is_global"] and result["connects_instantly"],
            f"{cfg['hostname']} in {TEST_NET_3} (RFC 5737), is_global={result['is_global']}; "
            "the doc says gpu-box then 'connects instantly' without saying to replace it",
        ),
        practice.Check(
            "FINDING: ForwardAgent yes exposes your agent on shared GPU boxes",
            cfg["forwardagent"] == "yes" and len(result["shared_hosts"]) == 3,
            f"ForwardAgent {cfg['forwardagent']}; ssh_config(5): '{WARNING}'; the doc aims it "
            f"at {', '.join(result['shared_hosts'])}",
        ),
        practice.Check(
            "CONTROL: the keygen email is a reserved example domain too",
            result["email_domain"] == "example.com",
            f"ssh-keygen -C uses @{result['email_domain']} (RFC 2606)",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
