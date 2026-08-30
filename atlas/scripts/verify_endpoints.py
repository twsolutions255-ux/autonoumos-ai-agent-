#!/usr/bin/env python3
"""Prove every endpoint Atlas calls actually exists in the TWS app.

An autonomous agent that calls a path which 404s does not crash loudly — it
records a failed tool call, reasons around it, and quietly stops being able to
do that thing. Nobody notices for weeks. This script makes that class of bug
impossible to merge.

It reads the real route table out of the app's server.py and every client call
out of Atlas's tool modules, then reports paths Atlas uses that the app does
not serve. Run it in CI, and after any TWS deploy.

    python scripts/verify_endpoints.py [--server /path/to/server.py]
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_SERVER = pathlib.Path("/home/user/tws-app/backend/server.py")

ROUTE_RE = re.compile(r'@api_router\.(get|post|put|patch|delete)\(\s*"([^"]+)"')
# Matches ctx.client.get("/path"...) / .post("/path", ...) and f-string paths.
CALL_RE = re.compile(r'\.(get|post|put|patch|delete)\(\s*f?"(/[^"]*)"')
#: Job paths are held in a dict literal and dispatched through
#: run_internal_job(), so the verb-prefixed regex above never sees them. They
#: are all POSTs, and they are exactly the calls most worth checking: an
#: /internal/* typo fails as a silent 401 that looks like a missing secret.
JOB_RE = re.compile(r'"(/internal/[a-z0-9/\-]+)"')

#: Path segments that are parameters. The app declares them as {name}; Atlas
#: interpolates them, so both sides are normalised to a single placeholder
#: before comparison.
PARAM_RE = re.compile(r"\{[^}]+\}")


def normalise(path: str) -> str:
    path = PARAM_RE.sub("{}", path)
    return path.rstrip("/") or "/"


def app_routes(server: pathlib.Path) -> set:
    if not server.exists():
        print(f"! server.py not found at {server} — cannot verify.", file=sys.stderr)
        sys.exit(2)
    text = server.read_text(errors="replace")
    return {(m.group(1).upper(), normalise(m.group(2))) for m in ROUTE_RE.finditer(text)}


def atlas_calls() -> list:
    """Every HTTP call Atlas makes, with where it is written."""
    out = []
    for py in sorted((ROOT / "atlas").rglob("*.py")):
        # The client module defines the verbs; it makes no business calls of
        # its own except /auth/login, which is checked explicitly below.
        if py.name == "client.py":
            continue
        for i, line in enumerate(py.read_text(errors="replace").splitlines(), 1):
            # Atlas's own FastAPI route decorators (@app.get("/health")) look
            # exactly like client calls to this regex. They describe the routes
            # Atlas SERVES, not the ones it calls, so skip them.
            if line.lstrip().startswith("@app.") or line.lstrip().startswith("@router."):
                continue
            for m in CALL_RE.finditer(line):
                verb, path = m.group(1).upper(), m.group(2)
                # An f-string path with an interpolated segment becomes {}.
                path = re.sub(r"\{[^}]*\}", "{}", path)
                out.append((verb, normalise(path), f"{py.relative_to(ROOT)}:{i}"))
    return out


def matches(call_path: str, route_path: str) -> bool:
    """Does a concrete path Atlas calls satisfy a route pattern?

    Exact match first. Failing that, a literal segment Atlas hardcodes may fill
    a parameter slot the app declares — Atlas calling
    /chat/channels/general/messages is a legitimate use of
    /chat/channels/{channel}/messages, and must not be reported as missing.
    A parameter slot never absorbs more than one segment.
    """
    if call_path == route_path:
        return True
    a, b = call_path.strip("/").split("/"), route_path.strip("/").split("/")
    if len(a) != len(b):
        return False
    return all(seg_b == "{}" or seg_a == seg_b for seg_a, seg_b in zip(a, b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", type=pathlib.Path, default=DEFAULT_SERVER)
    args = ap.parse_args()

    routes = app_routes(args.server)
    by_path: dict = {}
    for verb, path in routes:
        by_path.setdefault(path, set()).add(verb)

    calls = atlas_calls()
    calls.append(("POST", "/auth/login", "atlas/tws/client.py:login"))
    for py in sorted((ROOT / "atlas").rglob("*.py")):
        for i, line in enumerate(py.read_text(errors="replace").splitlines(), 1):
            for m in JOB_RE.finditer(line):
                calls.append(("POST", normalise(m.group(1)),
                              f"{py.relative_to(ROOT)}:{i}"))

    missing, wrong_verb, ok = [], [], 0
    for verb, path, where in calls:
        candidates = [rp for rp in by_path if matches(path, rp)]
        if not candidates:
            missing.append((verb, path, where))
            continue
        # Prefer the most specific route (fewest parameter slots) when a
        # literal path could satisfy more than one pattern.
        candidates.sort(key=lambda rp: rp.count("{}"))
        allowed = set()
        for rp in candidates:
            allowed |= by_path[rp]
        if verb not in allowed:
            wrong_verb.append((verb, path, where, sorted(allowed)))
        else:
            ok += 1

    print(f"app routes found:   {len(routes)}")
    print(f"atlas calls found:  {len(calls)}")
    print(f"verified OK:        {ok}")

    if missing:
        print(f"\nMISSING — Atlas calls {len(missing)} path(s) the app does not serve:")
        for verb, path, where in missing:
            print(f"  {verb:6} {path:50} {where}")
    if wrong_verb:
        print(f"\nWRONG METHOD — {len(wrong_verb)} call(s):")
        for verb, path, where, have in wrong_verb:
            print(f"  {verb:6} {path:50} {where}  (app has {', '.join(have)})")

    if not missing and not wrong_verb:
        print("\nEvery endpoint Atlas calls exists in the app, with the right method.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
