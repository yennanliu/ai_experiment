"""A strict YAML *subset* parser.

`DESIGN §4` requires `harness/` to run on a bare Python with nothing installed,
so a real YAML dependency is not affordable. The price is that the subset must
be small enough to be obviously correct, and strict enough that anything outside
it is a loud error rather than a silent misparse.

Supported, and nothing else:

    key: scalar            mappings, 2-space indent
    key:                   nested block (mapping or sequence)
    - scalar               sequences of scalars
    - key: v               sequences of mappings
    key: |                 literal block scalar (newlines kept)
    key: >                 folded block scalar (newlines become spaces)
    "quoted"  'quoted'     quoted scalars; everything else is bare
    123  1.5  true  null   int / float / bool / null, else str
    # comment              whole-line and trailing on bare scalars

Deliberately absent: anchors, aliases, tags, flow collections, multi-document
streams, complex keys. Each raises `YamlError` naming the line.
"""

from __future__ import annotations

INDENT = 2


class YamlError(ValueError):
    """A construct outside the supported subset, or malformed input."""


def _scalar(raw: str, line_no: int):
    text = raw.strip()
    if not text:
        return ""
    if text[0] in "\"'":
        quote = text[0]
        if len(text) < 2 or text[-1] != quote:
            raise YamlError(f"line {line_no}: unterminated {quote} string")
        return text[1:-1]
    if text[0] in "[{":
        raise YamlError(f"line {line_no}: flow collections are not supported")
    if text[0] in "&*!":
        raise YamlError(f"line {line_no}: anchors, aliases and tags are not supported")
    if "#" in text:                       # trailing comment on a bare scalar
        text = text.split("#", 1)[0].strip()
    low = text.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "~", ""):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


class _Reader:
    def __init__(self, text: str):
        self.lines = []                   # (line_no, indent, content)
        for n, raw in enumerate(text.splitlines(), 1):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            if raw.strip() == "---":
                continue
            if "\t" in raw[: len(raw) - len(raw.lstrip())]:
                raise YamlError(f"line {n}: tab in indentation")
            indent = len(raw) - len(raw.lstrip())
            if indent % INDENT:
                raise YamlError(f"line {n}: indent {indent} is not a multiple of {INDENT}")
            self.lines.append((n, indent, raw.strip()))
        self.raw = text.splitlines()
        self.i = 0

    def peek(self):
        return self.lines[self.i] if self.i < len(self.lines) else None

    def block_scalar(self, style: str, parent_indent: int) -> str:
        """Consume the raw source lines of a `|` or `>` block."""
        start_src = self.lines[self.i - 1][0]          # the `key: |` line itself
        body, src = [], start_src                      # walk the *raw* text
        while src < len(self.raw):
            line = self.raw[src]
            if line.strip() and (len(line) - len(line.lstrip())) <= parent_indent:
                break
            body.append(line)
            src += 1
        while body and not body[-1].strip():
            body.pop()
        if not body:
            return ""
        strip = min(len(l) - len(l.lstrip()) for l in body if l.strip())
        text = [l[strip:] if l.strip() else "" for l in body]
        # skip the consumed lines in the logical stream
        while self.i < len(self.lines) and self.lines[self.i][0] <= src:
            self.i += 1
        return "\n".join(text) + "\n" if style == "|" else " ".join(
            l for l in text if l
        )


def _parse_block(r: _Reader, indent: int):
    item = r.peek()
    if item is None or item[1] < indent:
        return None
    return _parse_seq(r, indent) if item[2].startswith("- ") or item[2] == "-" else _parse_map(r, indent)


def _parse_map(r: _Reader, indent: int) -> dict:
    out = {}
    while True:
        item = r.peek()
        if item is None or item[1] < indent:
            return out
        line_no, ind, content = item
        if ind > indent:
            raise YamlError(f"line {line_no}: unexpected indent")
        if content.startswith("- "):
            raise YamlError(f"line {line_no}: sequence item where a mapping key was expected")
        if ":" not in content:
            raise YamlError(f"line {line_no}: expected 'key: value', got {content!r}")
        key, _, rest = content.partition(":")
        key = key.strip()
        if key.startswith(("\"", "'")):
            key = key[1:-1]
        if not key:
            raise YamlError(f"line {line_no}: empty key")
        if key in out:
            raise YamlError(f"line {line_no}: duplicate key {key!r}")
        r.i += 1
        rest = rest.strip()
        if rest in ("|", ">", "|-", ">-"):
            out[key] = r.block_scalar(rest[0], ind)
        elif rest == "":
            nested = _parse_block(r, ind + INDENT)
            out[key] = nested if nested is not None else None
        else:
            out[key] = _scalar(rest, line_no)


def _parse_seq(r: _Reader, indent: int) -> list:
    out = []
    while True:
        item = r.peek()
        if item is None or item[1] < indent:
            return out
        line_no, ind, content = item
        if not (content.startswith("- ") or content == "-"):
            return out
        if ind != indent:
            raise YamlError(f"line {line_no}: misaligned sequence item")
        body = content[2:].strip() if content != "-" else ""
        r.i += 1
        if body == "":
            nested = _parse_block(r, indent + INDENT)
            out.append(nested)
        elif ":" in body and not body.startswith(("\"", "'")):
            # inline first key of a mapping item: "- key: value"
            key, _, rest = body.partition(":")
            entry = {key.strip(): _scalar(rest, line_no) if rest.strip() else None}
            rest_stripped = rest.strip()
            if rest_stripped in ("|", ">", "|-", ">-"):
                entry[key.strip()] = r.block_scalar(rest_stripped[0], ind)
            more = r.peek()
            if more and more[1] == ind + INDENT and not more[2].startswith("- "):
                nested = _parse_map(r, ind + INDENT)
                for k, v in nested.items():
                    if k in entry:
                        raise YamlError(f"line {more[0]}: duplicate key {k!r}")
                    entry[k] = v
            out.append(entry)
        else:
            out.append(_scalar(body, line_no))


def loads(text: str):
    """Parse the supported subset. Raises `YamlError` on anything else."""
    r = _Reader(text)
    if not r.lines:
        return None
    if r.lines[0][1] != 0:
        raise YamlError(f"line {r.lines[0][0]}: document must start at column 0")
    value = _parse_block(r, 0)
    if r.peek() is not None:
        raise YamlError(f"line {r.peek()[0]}: trailing content")
    return value


def load(path) -> object:
    import pathlib
    return loads(pathlib.Path(path).read_text(encoding="utf-8"))
