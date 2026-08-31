"""The rung said it would brief the owner, and the gate said no.

AUTONOMY_DESCRIPTIONS is not documentation. It is quoted verbatim into Atlas's
own system prompt, so it is the agent's self-description, and the owner reads it
in the console when deciding how much rope to give. When it disagrees with the
gate, one of them is lying to somebody who is making a decision on it.

It disagreed. 'recommend' says:

    "Read everything, and write a plan and a briefing to the owner."

Both the morning and evening cycle instructions end with `brief_owner`.
`brief_owner` is INTERNAL_COMMS, which requires 'assist'. So Atlas was told
twice a day to brief the owner, tried, and was refused every time -- and a
policy denial comes back as a readable refusal the model absorbs and moves on
from, not an exception anybody sees. The Briefings tab was empty and nothing
anywhere said why.

Fixed with a per-tool `requires="recommend"` override rather than by raising the
rung, because raising it would also have unlocked post_to_channel, announce,
direct_message, ask_ai_employee and wake_ai_employees -- five tools that write
to the TEAM. brief_owner writes into Atlas's own briefs collection, which only
the owner reads. Same risk class, entirely different blast radius.

The test that matters longest is the last section: every capability a rung's
description PROMISES must actually be executable at that rung. That is what
keeps the prompt honest as tools are added.
"""
import pytest

# Importing the registry alone leaves it empty -- the tool modules populate it
# through their decorators. Same import the brain loop does.
from atlas.tools import clientcare, comms, growth, money, observe, reflect  # noqa: F401
from atlas.tools.registry import registry
from atlas.config import AUTONOMY_DESCRIPTIONS, AUTONOMY_LEVELS, Settings
from atlas.guardrails.policy import Policy, RateLimiter, Risk


def _policy(level, *, sandbox=True, kill_switch=False):
    s = Settings(autonomy=level, sandbox=sandbox, kill_switch=kill_switch)
    return Policy(s, RateLimiter())


def _allowed(level, name, *, sandbox=True, kill_switch=False):
    tool = registry.get(name)
    assert tool is not None, "%s is not registered" % name
    return _policy(level, sandbox=sandbox,
                   kill_switch=kill_switch).check(name, tool.policy, {}).allowed


# ------------------------------------------------------- the specific bug

def test_the_owner_gets_briefed_at_the_rung_that_promises_it():
    """The regression. Atlas ships at 'recommend' with sandbox on, which is
    exactly the posture this was refused in."""
    assert _allowed("recommend", "brief_owner"), \
        "the rung promises a briefing and the gate refuses it"


def test_briefing_works_in_the_sandbox_too():
    """Sandbox is documented as 'Atlas reads, reasons, plans and briefs, and
    nothing else happens'. A sandboxed Atlas that cannot brief is not
    evaluable, which is the one thing sandbox exists for."""
    assert _allowed("recommend", "brief_owner", sandbox=True)


def test_observe_still_cannot_brief():
    """'observe' says "Write no messages to anyone." It has to mean it."""
    assert not _allowed("observe", "brief_owner")


def test_the_fix_did_not_unlock_the_tools_that_write_to_the_team():
    """The whole reason for a per-tool override instead of raising the rung."""
    for name in ("post_to_channel", "announce", "direct_message",
                 "ask_ai_employee", "wake_ai_employees"):
        if registry.get(name) is None:
            continue
        assert not _allowed("recommend", name), \
            "%s became reachable at 'recommend'; only brief_owner should have" % name


def test_brief_owner_keeps_its_real_risk_class():
    """The override changes which rung is needed, not what the tool IS. A human
    does read it, and the console groups by risk."""
    assert registry.get("brief_owner").policy.risk is Risk.INTERNAL_COMMS


def test_the_kill_switch_still_stops_it():
    """A briefing is not a safety action, so nothing about it may outrank the
    kill switch."""
    assert not _allowed("recommend", "brief_owner", kill_switch=True)


# --------------------------------------------- the test that generalises

#: What each rung's own words commit to, read off AUTONOMY_DESCRIPTIONS as
#: (tool name, the rung whose text promises it). Add a line whenever a
#: description does.
RUNG_PROMISES = [
    ("set_plan", "recommend"),      # "write a plan"
    ("brief_owner", "recommend"),   # "and a briefing to the owner"
    ("remember", "recommend"),      # "the only writes are into Atlas's own memory"
    ("post_to_channel", "assist"),  # "talk to the team"
    ("ask_ai_employee", "assist"),  # "and the AI employees"
]


@pytest.mark.parametrize("name,rung", RUNG_PROMISES)
def test_every_promise_a_rung_makes_is_actually_executable(name, rung):
    """A promise the gate refuses is the agent being lied to about itself, and
    the owner being lied to about what they authorised."""
    if registry.get(name) is None:
        pytest.skip("%s is not registered in this build" % name)
    assert _allowed(rung, name, sandbox=False), \
        "AUTONOMY_DESCRIPTIONS[%r] promises %s and the gate refuses it" % (rung, name)


def test_the_ladder_is_still_a_ladder():
    """Each rung must permit everything the rung below it does.

    A per-tool override that lowered a requirement could otherwise punch a hole
    rather than move a step -- reachable at 'recommend' and 'operate' but not
    'assist' would make the console's ladder a lie.
    """
    for name in registry.names():
        tool = registry.get(name)
        reachable = [lvl for lvl in AUTONOMY_LEVELS
                     if _policy(lvl, sandbox=False).check(name, tool.policy, {}).allowed]
        if not reachable:
            continue
        first = AUTONOMY_LEVELS.index(reachable[0])
        assert reachable == list(AUTONOMY_LEVELS[first:]), \
            "%s is reachable at %s -- not a contiguous ladder" % (name, reachable)


def test_every_rung_named_in_descriptions_is_a_real_rung():
    assert set(AUTONOMY_DESCRIPTIONS) == set(AUTONOMY_LEVELS)
