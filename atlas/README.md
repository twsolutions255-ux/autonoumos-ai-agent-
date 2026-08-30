# Atlas — the autonomous scaling agent for TW Solutions

Atlas is a Chief of Staff for one company. It wakes on its own clock, reads the
real numbers, decides what matters most today, does something about it, and
tells you what it did. It talks to your human team, your four AI employees and
the AI Operator, and it acts through the app's own API — as a superadmin, over
HTTP, exactly like a person would.

It has one job: **grow TW Solutions' profitable revenue.** Not to be generally
helpful, not to answer questions. If it cannot connect an action to revenue or
to keeping revenue, it should not be doing it.

---

## Why this exists

The app already had every part of a growth machine and nothing driving it.

You can scan a market, audit a business's website, build them a free site,
draft outreach citing the specific faults you found, stage a call batch,
follow up, and convert. Four AI employees watch pipeline, lead flow, client
health and coaching. All of it works. All of it waits for somebody to press
the buttons in the right order.

The AI Manager already in the app is not that somebody. It is reactive and
advisory by design: it answers when you type, keeps twenty messages of
history, and is explicitly forbidden from placing a call or touching a
campaign. It is a good assistant. It is not an operator.

Atlas is the missing half:

| | AI Manager (existing) | Atlas |
|---|---|---|
| Runs | when you type | on its own clock, all day |
| Remembers | last 20 messages | every cycle, decision and outcome, with supersession |
| Can act | advisory only | the whole pipeline, under a policy gate |
| Has a goal | no | a written plan with measured objectives |
| Reports | in the chat | morning plan, evening review, and when something needs you |

Both stay. They do different jobs.

---

## How it thinks

Every cycle has the same shape, because a process that varies cannot be
audited after the fact:

```
open a cycle record  →  read the live business  →  recall what it learned
                     →  reason and act through gated tools
                     →  record what happened, and what it now believes
```

Three cycles a day, plus work ticks between them:

- **morning** — read everything, decide today's one priority, tell the team, brief you
- **work** — advance the plan by one real step (every 15 minutes by default)
- **evening** — compare against the morning plan, record what was learned, brief you

Everything is written to `atlas_cycles` and `atlas_actions` as it happens, not
at the end, so a cycle that dies halfway still leaves a complete account of
what it did.

---

## What stops it doing something stupid

Four gates, checked in this order before **every** tool call. The order
matters: a stopped agent burns no budget deciding it is stopped.

0. **Safety actions** — anything that makes the situation *safer* (today: the
   cold-call brake) passes every gate, including the kill switch. A brake you
   have to be senior enough to pull is not a brake.
1. **Kill switch** — one flag, effective mid-cycle. Reads still work.
2. **Autonomy** — is this class of action unlocked at all?
3. **Rate limit** — has this channel already done enough this hour or day?
4. **Approval** — is this one action big enough to need you?

Every tool declares its risk class next to its handler, so there is no path
from "new endpoint" to "the agent can call it" that skips the decision.

### The autonomy ladder

| Level | What it may do |
|---|---|
| `observe` | Look. Change nothing, say nothing. |
| `recommend` | Plan and brief you. Nothing in the business changes — the only writes go into Atlas's own memory. **(ships here)** |
| `assist` | Change data in the app (scan markets, build sites, run analyses), talk to the team and the AI employees, queue work for a human to release. |
| `operate` | Run the acquisition pipeline end to end. |
| `autopilot` | Money-adjacent and irreversible actions, each still capped. |

Tools above the current level are never offered to the model, so it cannot
waste a turn reaching for one.

### Sandbox

On by default. Nothing changes and nothing reaches the outside world — no
calls, no staged batches, no money, and no edits to the app's data either.
Atlas still reads, reasons, plans and briefs, so you can judge its decisions
from the record before you trust it with any of them.

That last part is load-bearing and I got it wrong the first time. Twelve tools
that write to the *app's* database were classed as "internal" — a word the code
reserved for Atlas's own memory — which made them reachable at the shipped
default, invisible to the sandbox. An agent editing the CRM while the operator
believes it is rehearsing is exactly the failure this mode exists to prevent.
`Risk.APP_WRITE` now names that class, requires `assist`, and the sandbox
blocks it; two tests pin the promise so it cannot quietly come untrue again.

### What always needs you

Regardless of level: anything estimated at or over `ATLAS_APPROVAL_THRESHOLD_USD`,
and turning on the app's own unattended calling. Held actions land in the
approvals queue with everything needed to decide; approving one actually runs
it, rather than just marking a row.

---

## Atlas and the Automaton agent

They are different things and they do not collide.

**Automaton** reaches this business through the app's scoped agent-key API:
one `X-Agent-Key`, six whitelisted actions, its own per-key hourly and daily
caps. It can push leads, build sites and dial leads it created. It cannot read
revenue totals, transcripts or recordings, and it cannot bill anyone. TWS is
one of the things it touches, not its purpose.

**Atlas** is the operator. It authenticates as a superadmin and its only job
is this company.

Nothing here changes Automaton — that repo is untouched. Their limits are
per-key and entirely separate, so neither consumes the other's allowance.
Atlas can see what Automaton has been doing (`other_agents_activity`) for one
reason: two agents working the same market is worse than either working it
alone, and an empty-looking pipeline may just be one the other agent is
already on.

---

## Setup

Atlas must be a **superadmin**, and the app has no endpoint that can create
one — the account is seeded from the app's own `SUPERADMIN_EMAIL` /
`SUPERADMIN_PASSWORD` at startup. Either give Atlas those credentials, or seed
a second superadmin on the app side. A non-superadmin Atlas can read and chat
but cannot run the business, and `/status` will say so in those words.

```bash
cp .env.example .env      # fill in the four required blocks
pip install -r requirements.txt
uvicorn atlas.main:app --port 8090
```

Then check it is honest about itself:

```bash
curl localhost:8090/health
curl -H "X-Atlas-Key: $ATLAS_CONSOLE_API_KEY" localhost:8090/status
```

`/status` reports each capability and, for anything missing, what it means and
which variable turns it on. Nothing here fakes a capability: an unset key
makes a feature report as OFF rather than quietly doing nothing.

### Seeing it in the app

The dashboard has an **Atlas** tab (first under Today). It is served through
the TWS backend, which holds the console key — the key never reaches a
browser, because it can raise the agent's authority and release a cold-call
batch. Set on the *app's* service:

```
ATLAS_URL=https://atlas-tws-agent.onrender.com
ATLAS_CONSOLE_API_KEY=<the same key Atlas was deployed with>
```

Until both are set the tab says so plainly instead of showing an empty
dashboard that looks broken.

---

## Turning it up

Do this over days, not in one sitting.

1. **Ship it.** `recommend`, sandbox on. Read a few morning and evening briefs.
   Are its priorities the ones you would have picked?
2. **Let it talk.** `assist`. It coaches the team, asks the AI employees what
   they are seeing, and queues work for you to release. Watch #general.
3. **Sandbox off, still `assist`.** Now its queued work is real, but a human
   still releases anything outbound.
4. **`operate`.** It runs the pipeline itself. Watch the first cold-call batch
   it stages — read what it *refused* as carefully as what it staged.
5. **`autopilot`,** if you ever want it. Money actions still need approval
   above the threshold, and turning on unattended dialling always does.

Go back down a rung at any time from the console. Nothing needs a redeploy.

---

## Things worth knowing

Found while building this, and handled in the code rather than left as
folklore:

- **`GET /admin/clients` returns every tenant's plaintext receptionist API key.**
  Results are redacted before they reach Atlas's audit log, or it would have
  quietly assembled a file of live customer credentials.
- **The app locks an account for 15 minutes after 5 failed logins — even for
  the correct password.** A wrong `TWS_PASSWORD` would have had Atlas lock
  itself out within five tool calls, so it stops trying after two.
- **An AI employee only answers an `@mention` posted by a team user**, always
  replies in `#general` no matter where you asked, and the reply carries no
  reference to the question. A DM to one is stored and silently never
  answered. `ask_ai_employee` posts and then matches the reply on sender and
  timestamp, because nothing better exists.
- **The app drafts cold outreach; it does not send it.** Atlas will never
  claim to have emailed anyone. It builds the site, drafts the message, and a
  human sends it.
- **The app has no task queue at all.** Background work is bare
  `asyncio.create_task` that dies with the process, plus an in-process
  scheduler that is off by default and, on a free instance, asleep overnight —
  exactly when the nightly digest and the audit window need it. The five cron
  jobs run from GitHub Actions against a hardcoded hostname that breaks the day
  the service is renamed.

  Atlas can drive those jobs on demand (`run_scheduled_job`), which matters
  most right after it creates work for one — draining speed-to-lead the moment
  a lead arrives turns "called tomorrow" into "called within minutes". It can
  also be the standing clock: set `ATLAS_DRIVE_APP_JOBS=true` and it drives
  speed-to-lead and workflow follow-ups on a timer, with no model in the loop,
  because plumbing should run on time rather than be re-reasoned every five
  minutes. Off by default, and still gated by the kill switch, sandbox and
  autonomy — those jobs call and email real people.

---

## Tests

```bash
python -m pytest tests/            # 29 tests, no network, no database
python scripts/verify_endpoints.py # every path Atlas calls exists in the app
```

`verify_endpoints.py` matters more than it looks. An agent that calls a path
which 404s does not crash — it records a failed tool call, reasons around it,
and quietly stops being able to do that thing. Nobody notices for weeks. Run
it in CI and after any TWS deploy.

The integration tests run against a fake TWS server that reproduces the app's
*awkward* behaviours on purpose: the fire-and-forget employee reply, the 401
that needs a re-login and replay, the leaked API key, the tenant-scoped 400.
Those are what Atlas actually has to survive.

---

## Layout

```
atlas/
  config.py          every knob, and what each one being off actually means
  db.py              its own atlas_* collections; never writes app collections
  guardrails/        the four gates — the whole safety model in one file
  llm/engine.py      the agentic loop: cached prompt, gated tools, real budget
  memory/            episodic and semantic memory, with supersession
  tools/             82 tools over verified endpoints
    observe.py         seeing the business
    comms.py           humans, AI employees, the AI Operator
    growth.py          the acquisition pipeline
    money.py           revenue, costs, and the fastest paths to more
    clientcare.py      keeping and growing the clients already paying
    reflect.py         its own plan, memory and briefings
  brain/
    prompts.py       who it is and how it decides
    loop.py          the cycle, the chat, and the scheduler
  main.py            the control API
```

Atlas never writes to the app's database directly. Every change goes through
the HTTP API so the app's own validation, logging and webhooks still fire. An
agent that reached into Mongo would silently bypass every rule the app
enforces, and the rules are most of what makes the app safe.
