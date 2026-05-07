# Agents in this project

This project uses a multi-agent setup coordinated by markdown notes in `.swarm/notes/`.

## Roster

| Agent | Channel | Cadence | Owns |
|---|---|---|---|
| **Cortex** | `#cortex` | every 30 min | strategy, prioritisation, decision logs |
| **Builder** | `#builder` | on demand | implementation, refactors, code review |
| **Scholar** | `#scholar` | hourly | research, library evaluation, citations |
| **Sentinel** | `#sentinel` | continuous | security, dependency audits, secrets hygiene |

Each agent's persona is in `.claude/agents/<name>/SOUL.md`.

## Coordination protocol

1. Before starting any task, read the latest notes under `.swarm/notes/<your-name>/` and the most recent 5 across all agents.
2. After each meaningful action, write a note: `.swarm/notes/<your-name>/<iso-timestamp>.md`.
3. Commit notes with `git add .swarm && git commit -m "<your-name>: <one-line summary>"`.
4. To request another agent's input, mention them in your note: `@scholar please verify…`.
5. Synthesis: `/synthesize` aggregates the last N notes into a single decision document at `.swarm/synthesis/<date>.md`.

## Heartbeat

If an agent is configured with a heartbeat (cron), it should:

- Read the last 10 notes across the swarm.
- If any contain `@<my-name>`, respond with a note.
- If owner-area work is pending (per the table above), action it.
- Always leave a note, even if just `nothing to do`.

## Note format

```markdown
---
agent: cortex
ts: 2026-05-05T22:30:00Z
tags: [decision, architecture]
---

**Decision**: switch storage to DynamoDB for swarm notes.
**Why**: per-scope query patterns; S3 list latency too high.
**Next**: @builder implement the table schema in PR.
```

## Branch convention (multi-agent async)

When multiple agents work on the same repo concurrently — possibly on different
machines — they MUST work on per-agent branches and merge through PRs. Never
push directly to `main` while a swarm is active.

| Convention | Value |
|---|---|
| Branch name | `agents/<your-name>/main` (one per agent) |
| Working dir | clone or worktree, named `<repo>-<your-name>` |
| Push cadence | after each meaningful commit |
| Pull cadence | every ~5 min via `boothub-watch` (or /loop) |
| Merge to main | explicit, via PR — humans or designated synthesis-agent only |

### When you join a session

If you don't have the repo locally:
```bash
git clone <repo-url> <repo-name>-<your-name>
cd <repo-name>-<your-name>
git checkout -b agents/<your-name>/main
```

If you DO have the repo (other agents on same machine), use a worktree to
share the object DB:
```bash
cd <repo-name>
git fetch origin
git worktree add ../<repo-name>-<your-name> -b agents/<your-name>/main origin/main
cd ../<repo-name>-<your-name>
```

### Pulling peer work

```bash
git fetch origin 'refs/heads/agents/*:refs/heads/agents/*'
git log --oneline --all --branches='agents/*' --since='1 hour ago'
```

To inspect a peer's working tree without disturbing yours:
```bash
git worktree add /tmp/peek-bob agents/bob/main
cd /tmp/peek-bob
# look around, then:
cd -; git worktree remove /tmp/peek-bob
```

### Pushing your work

After each commit:
```bash
git push -u origin agents/<your-name>/main
```

Then announce via swarm note (so peers know to pull). Use the `/wt-push`
slash command — it does both in one step.

### When you're done

Open a PR back to `main` from `agents/<your-name>/main`. Tag your final
swarm note with `["push","done"]`.

## Synthesis-agent role

For long-running swarms, designate one agent (or human) as the **synthesizer**:
- Pulls all `agents/*` branches
- Reviews each agent's recent commits + swarm notes
- Decides what merges to `main` (rebase, squash, cherry-pick as appropriate)
- Posts a synthesis note when state has been consolidated
