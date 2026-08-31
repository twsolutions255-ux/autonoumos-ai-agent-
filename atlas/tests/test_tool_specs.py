"""Every tool spec must be something the API will actually accept.

Atlas ran for over an hour without completing a single cycle. /health said
ok: true the whole time, because the process was alive and only the work was
failing. Every fifteen minutes:

    anthropic.BadRequestError: 400 - tools.45.custom: For 'object' type,
    'additionalProperties: object' is not supported. Please set
    'additionalProperties' to false

One tool -- snapshot_metrics -- declared a nested object with
`additionalProperties: {"type": "number"}`, a free-form name-to-number map.
Under strict mode every object in the tree must forbid extra properties. And a
malformed tool rejects the WHOLE request, so all forty-five went down together.

Two things had to be true for that to ship, and both are fixed:

1. Tool.spec() normalised only the TOP level of each schema. A normaliser was
   written, it looked like it covered the problem, and it covered exactly one
   level of it.

2. Nothing tested the tool specs. Forty-five tools, no check that any of them
   was well-formed, and the only feedback was a 400 from a live API on a
   schedule -- landing in a log nobody was reading, while the health endpoint
   stayed green.

This file is the check that was missing. It validates the specs OFFLINE, so
the next malformed schema fails here rather than at 15-minute intervals in
production.
"""
import pytest

# Importing the registry alone leaves it empty -- the tool modules are what
# populate it, through their decorators. Same import the brain loop does.
from atlas.tools import clientcare, comms, growth, money, observe, reflect  # noqa: F401
from atlas.tools.registry import registry, _strictify


def all_specs():
    return [registry.get(name).spec() for name in registry.names()]


def walk_objects(schema, path="root"):
    """Yield (path, subschema) for every object-typed node in the tree."""
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object" or "properties" in schema:
        yield path, schema
        for key, sub in (schema.get("properties") or {}).items():
            yield from walk_objects(sub, "%s.%s" % (path, key))
    if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        yield from walk_objects(schema["items"], "%s[]" % path)
    for combinator in ("anyOf", "oneOf", "allOf"):
        for i, sub in enumerate(schema.get(combinator) or []):
            yield from walk_objects(sub, "%s.%s[%d]" % (path, combinator, i))


def test_there_are_tools_to_check():
    specs = all_specs()
    assert len(specs) > 20, "expected the full tool set, got %d" % len(specs)


def test_every_object_in_every_tool_forbids_extra_properties():
    """The exact rule the API enforces, checked at every depth.

    Top level only was the bug. A nested object is where it actually bit.
    """
    problems = []
    for spec in all_specs():
        for path, node in walk_objects(spec["input_schema"], spec["name"]):
            ap = node.get("additionalProperties")
            if ap is not False:
                problems.append("%s: additionalProperties is %r, must be False"
                                % (path, ap))
    assert not problems, "\n".join(problems)


def test_no_tool_declares_a_free_form_map():
    """The specific shape that broke it. A map of unknown keys cannot be
    expressed under strict mode -- it has to be name/value pairs."""
    for spec in all_specs():
        for path, node in walk_objects(spec["input_schema"], spec["name"]):
            ap = node.get("additionalProperties")
            assert not isinstance(ap, dict), (
                "%s declares a free-form map. Use an array of "
                "{name, value} pairs instead." % path)


def test_every_object_declares_properties_and_required():
    for spec in all_specs():
        for path, node in walk_objects(spec["input_schema"], spec["name"]):
            assert isinstance(node.get("properties"), dict), \
                "%s has no properties map" % path
            assert isinstance(node.get("required"), list), \
                "%s has no required list" % path


def test_required_names_actually_exist():
    """A required field that is not in properties is rejected under strict
    mode, and is a typo nobody would otherwise catch."""
    for spec in all_specs():
        for path, node in walk_objects(spec["input_schema"], spec["name"]):
            props = set((node.get("properties") or {}).keys())
            for name in node.get("required") or []:
                assert name in props, \
                    "%s requires %r which is not one of its properties" % (path, name)


def test_snapshot_metrics_takes_pairs_not_a_map():
    """The tool that took every cycle down, pinned by name."""
    spec = next(s for s in all_specs() if s["name"] == "snapshot_metrics")
    metrics = spec["input_schema"]["properties"]["metrics"]
    assert metrics["type"] == "array", "metrics must be pairs, not a map"
    item = metrics["items"]
    assert set(item["properties"]) == {"name", "value"}
    assert item["additionalProperties"] is False


# ---------------------------------------------------------------- the fixer

def test_strictify_reaches_every_level():
    """Proving the normaliser itself, not just its output on today's tools."""
    got = _strictify({
        "type": "object",
        "properties": {
            "nested": {"type": "object", "properties": {"a": {"type": "string"}}},
            "listed": {"type": "array",
                       "items": {"type": "object",
                                 "properties": {"b": {"type": "number"}}}},
        },
    })
    assert got["additionalProperties"] is False
    assert got["properties"]["nested"]["additionalProperties"] is False, \
        "a nested object was left open -- this is the original bug"
    assert got["properties"]["listed"]["items"]["additionalProperties"] is False, \
        "an object inside an array was left open"


def test_strictify_replaces_a_map_rather_than_keeping_it():
    got = _strictify({"type": "object",
                      "properties": {"m": {"type": "object",
                                           "additionalProperties": {"type": "number"}}}})
    assert got["properties"]["m"]["additionalProperties"] is False


def test_strictify_handles_combinators():
    got = _strictify({
        "type": "object",
        "properties": {
            "either": {"anyOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}},
                {"type": "string"},
            ]},
        },
    })
    branch = got["properties"]["either"]["anyOf"][0]
    assert branch["additionalProperties"] is False


def test_strictify_leaves_non_objects_alone():
    assert _strictify({"type": "string"}) == {"type": "string"}
    assert _strictify("not a schema") == "not a schema"
