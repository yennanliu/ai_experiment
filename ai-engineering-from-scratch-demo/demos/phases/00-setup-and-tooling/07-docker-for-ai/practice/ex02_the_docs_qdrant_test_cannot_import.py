"""Exercise 2 — Qdrant is reachable, but the doc's Python test cannot import its client.

    Start the docker-compose stack and verify Qdrant is accessible from the AI container at `http://qdrant:6333/collections`

Reading of the exercise: with no docker daemon in CI, "start and verify" is read
as resolving what `docker compose up` would build from the lesson's own
`code/docker-compose.yml` -- the network, the service name, the container port,
the mounts -- and checking each link a request from `ai-dev` to
`http://qdrant:6333/collections` needs, plus the tools inside `ai-dev` that
could make that request. The YAML is read with the harness's stdlib subset
parser after its one flow list (`[gpu]`) is rewritten into block form; that
parser reads an unquoted `- a:/b` item as a one-key mapping, which
`mount_source` undoes.

**ANSWER: yes, by `curl` from inside `ai-dev`.** Neither service declares
`networks:`, so both join the project's default network (`code_default`, since
the project name defaults to the directory `code`), where the service name
`qdrant` resolves; Qdrant's REST port is the container side of `6333:6333`.
`docker compose exec ai-dev curl -s http://qdrant:6333/collections` works
because `curl` is in the image's apt list.

**FINDING: the lesson's own Python test fails with ModuleNotFoundError.** The
doc verifies the link with `from qdrant_client import QdrantClient`, but none of
the 15 pip packages in the Dockerfile is `qdrant-client`.

**FINDING: `../../../` mounts `phases/`, not the repository.** Resolved from
`phases/00-setup-and-tooling/07-docker-for-ai/code/`, the bind mount lands on
`phases`, while the doc's `docker run -v $(pwd):/workspace` from the repo root
mounts the root: `/workspace` means two different trees depending on which of
the lesson's two launch paths is used.

**CONTROL: the file is what the doc shows.** The doc's ```yaml block equals
`code/docker-compose.yml`, Qdrant is pinned to `v1.12.5`, and the GPU request
asks for `count: all`. There is no `depends_on`, so nothing orders the start-up.

Structure: `compose` parses the file; the checks walk its services.
"""

from __future__ import annotations

import posixpath
import re

from harness import parity, practice, yamlite

PHASE, LESSON = "00-setup-and-tooling", "07-docker-for-ai"
CODE_DIR = f"phases/{PHASE}/{LESSON}/code"


def block_flow(text):
    """Rewrite `key: [a, b]` as a block sequence, the one construct yamlite rejects."""

    def expand(m):
        pad, key, items = m.groups()
        return f"{pad}{key}:\n" + "".join(f"{pad}  - {i.strip()}\n" for i in items.split(","))

    return re.sub(r"(?m)^( *)([\w-]+): \[(.*)\]\n", expand, text)


def mount_source(item):
    """Host side of a volume; yamlite reads an unquoted `a:/b` item as {a: /b}."""
    return next(iter(item)) if isinstance(item, dict) else item.split(":")[0]


def compose(root):
    text = (root / "code" / "docker-compose.yml").read_text(encoding="utf-8")
    return text, yamlite.loads(block_flow(text))


def solve():
    root = parity.lesson_dir(PHASE, LESSON)
    text, spec = compose(root)
    services = spec["services"]
    dockerfile = (root / "code" / "Dockerfile").read_text(encoding="utf-8")
    doc = parity.doc_text(PHASE, LESSON)
    mount = mount_source(services["ai-dev"]["volumes"][0])
    return {
        "services": sorted(services),
        "networks": [s for s in services.values() if "networks" in s],
        "qdrant_ports": [p.split(":")[1] for p in services["qdrant"]["ports"]],
        "image": services["qdrant"]["image"],
        "gpu": services["ai-dev"]["deploy"]["resources"]["reservations"]["devices"][0],
        "depends": [s for s in services.values() if "depends_on" in s],
        "has_curl": re.search(r"\bcurl \\", dockerfile) is not None,
        "client_in_image": "qdrant" in dockerfile.lower(),
        "doc_imports_client": "from qdrant_client import QdrantClient" in doc,
        "mount": posixpath.normpath(posixpath.join(CODE_DIR, mount)),
        "project": posixpath.basename(CODE_DIR),
        "doc_equals_file": re.search(r"```yaml\n(.*?)```", doc, re.S).group(1).strip()
        == text.strip(),
    }


def verify(result):
    net = f"{result['project']}_default"
    return [
        practice.Check(
            "ANSWER: reachable as qdrant:6333 on the default network, via curl",
            result["services"] == ["ai-dev", "qdrant"]
            and not result["networks"]
            and "6333" in result["qdrant_ports"]
            and result["has_curl"],
            f"services {result['services']}, none declares networks:, so both join {net}; "
            f"qdrant listens on container ports {result['qdrant_ports']}; curl is in the "
            "ai-dev apt list, so `docker compose exec ai-dev curl -s "
            "http://qdrant:6333/collections` is the check that works",
        ),
        practice.Check(
            "FINDING: the doc's Python test cannot import qdrant_client in this image",
            result["doc_imports_client"] and not result["client_in_image"],
            "docs/en.md tests the link with `from qdrant_client import QdrantClient`; the "
            "Dockerfile never installs qdrant-client, so that test raises ModuleNotFoundError",
        ),
        practice.Check(
            "FINDING: the ../../../ bind mount is phases/, not the repository root",
            result["mount"] == "phases",
            f"{CODE_DIR}/../../../ resolves to '{result['mount']}', while the doc's "
            "`docker run -v $(pwd):/workspace` from the repo root mounts the root",
        ),
        practice.Check(
            "CONTROL: the doc block is the file; Qdrant is pinned; GPUs are all reserved",
            result["doc_equals_file"]
            and result["image"].endswith(":v1.12.5")
            and result["gpu"]["count"] == "all",
            f"doc ```yaml == code/docker-compose.yml: {result['doc_equals_file']}; image "
            f"{result['image']}; gpu {result['gpu']}; services with depends_on: "
            f"{len(result['depends'])}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
