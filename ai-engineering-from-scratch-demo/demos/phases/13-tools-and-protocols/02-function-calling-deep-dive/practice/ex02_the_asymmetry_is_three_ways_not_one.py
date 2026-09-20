"""Exercise 2 — the asymmetry is three ways, not one.

    Add a `ListToolsResponse` parser for each provider that extracts the tool
    list a model returns after a `list_tools` or discovery call. OpenAI does not
    have one natively; note this asymmetry.

Reading of the exercise: the parser is written for each provider, which means
first finding the response shape it should parse -- and none of the three has
one. The exercise singles out OpenAI, so the other two are checked against the
same standard before the asymmetry is accepted. A parser is still built, because
the shape that does exist is MCP's, and writing it against MCP shows what the
other three are missing.

**ANSWER: none of the three has a tool-list response, so all three parsers are
the same parser.** In OpenAI, Anthropic and Gemini the tool list travels in the
*request* -- `tools=[...]`, and the lesson's own `to_openai` / `to_anthropic` /
`to_gemini` are the three writers for it. A model cannot return a tool list
because it never held one; it was handed one. Discovery is **0 of 3**.

**FINDING: the asymmetry the exercise names is real and points the wrong way.**
"OpenAI does not have one natively" implies the other two do. Searching the
lesson's module for any list-shaped response finds **0** of them, and the three
`parse_*` functions it does ship all read *calls* -- `tool_calls`, `tool_use`,
`functionCall` -- never declarations. The asymmetry is not OpenAI against
Anthropic and Gemini; it is all three against MCP.

**FINDING: which is the whole reason MCP exists.** MCP's `tools/list` is a
response shape, so the registry can change between turns without the host
redeploying. The three provider APIs bind the registry at request time, so the
set of callable tools is fixed for the duration of one call and is the caller's
responsibility to know. That is the difference Lesson 06 opens with, and it is
visible here as a parser that has nothing to parse.

**FINDING: the round trip recovers every name and only two of three schemas.**
Feeding each provider's declaration back through a reader recovers
`get_weather` from **3 of 3**, so a `ListToolsResponse` parser is writable -- it
just has to read the request the host sent rather than a response the model
returned. But only **2 of 3** schemas come back identical: OpenAI's and
Anthropic's are byte-for-byte, and Gemini's is not, because `_gemini_schema`
rewrites it. Discovery over these APIs would be lossy for one provider even once
the shape existed.

Structure: `mcp_list_tools` is the shape the other three lack, `parse_mcp_list`
is the parser the exercise asks for, `read_back` recovers a tool from each
provider's declaration, and `call_parsers_read` checks what the shipped parsers
actually extract.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "02-function-calling-deep-dive"
LIST_MARKERS = ("list_tools", "tools/list", "ListTools", "listTools")
CALL_MARKERS = {"parse_openai": "tool_calls", "parse_anthropic": "tool_use",
                "parse_gemini": "functionCall"}


def mcp_list_tools(ref, tool):
    """The response shape the three provider APIs do not have."""
    return {"jsonrpc": "2.0", "id": 1,
            "result": {"tools": [{"name": tool.name,
                                  "description": tool.description,
                                  "inputSchema": tool.input_schema}]}}


def parse_mcp_list(response):
    """The ListToolsResponse parser the exercise asks for, against the shape that has one."""
    return [(entry["name"], entry["inputSchema"])
            for entry in response["result"]["tools"]]


def read_back(ref, tool):
    """Recover (name, schema) from each provider's *request-side* declaration."""
    openai = ref.to_openai(tool)["function"]
    anthropic = ref.to_anthropic(tool)
    gemini = ref.to_gemini(tool)["functionDeclarations"][0]
    return {"openai": (openai["name"], openai["parameters"]),
            "anthropic": (anthropic["name"], anthropic["input_schema"]),
            "gemini": (gemini["name"], gemini["parameters"])}


def list_shapes(ref):
    """Any name in the lesson's module that looks like a tool-list response."""
    return [name for name in dir(ref)
            if any(marker.lower() in name.lower() for marker in LIST_MARKERS)]


def call_parsers_read(ref):
    """What each shipped parse_* function keys off."""
    return {name: marker in inspect.getsource(getattr(ref, name))
            for name, marker in CALL_MARKERS.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tool = ref.WEATHER
    recovered = read_back(ref, tool)
    parsed = parse_mcp_list(mcp_list_tools(ref, tool))
    return {
        "list_shapes": list_shapes(ref), "discovery_count": len(list_shapes(ref)),
        "call_parsers": call_parsers_read(ref),
        "call_parsers_all_read_calls": all(call_parsers_read(ref).values()),
        "mcp_parsed": parsed,
        "mcp_name": parsed[0][0],
        "mcp_schema_matches": parsed[0][1] == tool.input_schema,
        "recovered_names": sorted({name for name, _ in recovered.values()}),
        "recovered_count": len(recovered),
        "schemas_match": sum(schema == tool.input_schema
                             for _, schema in recovered.values()),
        "lossless": sorted(name for name, (_, schema) in recovered.items()
                           if schema == tool.input_schema),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: none of the three has a tool-list response -- discovery is 0 of 3",
            all([result["list_shapes"] == [], result["discovery_count"] == 0,
                 result["call_parsers_all_read_calls"]]),
            f"searching the lesson's module for {list(LIST_MARKERS)} finds "
            f"{result['discovery_count']} shapes. In all three APIs the tool list travels in "
            f"the request, which is what to_openai / to_anthropic / to_gemini write. A model "
            "cannot return a tool list because it never held one; it was handed one",
        ),
        practice.Check(
            "FINDING: the asymmetry is real and points the wrong way",
            all([result["discovery_count"] == 0,
                 result["call_parsers"] == {"parse_openai": True, "parse_anthropic": True,
                                            "parse_gemini": True}]),
            f"'OpenAI does not have one natively' implies the other two do. The three parse_* "
            f"functions the lesson ships all read calls rather than declarations -- "
            f"{result['call_parsers']}, keyed off tool_calls, tool_use and functionCall -- and "
            "no list shape exists for any provider. The asymmetry is not OpenAI against the "
            "other two; it is all three against MCP",
        ),
        practice.Check(
            "FINDING: which is the whole reason MCP exists",
            all([result["mcp_name"] == "get_weather", result["mcp_schema_matches"],
                 len(result["mcp_parsed"]) == 1]),
            f"MCP's tools/list is a response shape, so the registry can change between turns "
            f"without the host redeploying; parse_mcp_list recovers "
            f"{result['mcp_parsed'][0][0]} with its schema intact. The three provider APIs "
            "bind the registry at request time, so the callable set is fixed for one call and "
            "is the caller's to know -- visible here as a parser with nothing to parse",
        ),
        practice.Check(
            "FINDING: the round trip recovers every name and only two of three schemas",
            all([result["recovered_names"] == ["get_weather"],
                 result["recovered_count"] == 3, result["schemas_match"] == 2,
                 result["lossless"] == ["anthropic", "openai"]]),
            f"feeding each provider's declaration back through a reader recovers "
            f"{result['recovered_names']} from all {result['recovered_count']}, so a "
            f"ListToolsResponse parser is writable -- it just has to read the request the "
            f"host sent rather than a response the model returned. But only "
            f"{result['schemas_match']} of 3 schemas come back identical: "
            f"{result['lossless']} are byte-for-byte, and Gemini's is not, because "
            "_gemini_schema rewrites it. Discovery over these APIs would be lossy for one "
            "provider even once the shape existed",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
