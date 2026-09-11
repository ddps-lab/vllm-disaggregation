---
name: skill-aws
description: >-
  Universal rules for doing work on the cloud / remote instances (AWS, GPU
  boxes, any SSH host). Claude Code runs on the LOCAL machine — that's where
  files are edited and git runs — but any RUNTIME work (running code, imports,
  version checks, benchmarks, server launches, package installs, GPU/hardware
  checks) must run on the REMOTE instance over SSH, never locally. Use this
  skill whenever a task involves running, verifying, testing, importing,
  installing, launching, or measuring anything that needs the remote
  environment — even if the user phrases it as "run X", "check Y on this
  machine", or "does Z import?". Also use it whenever you record results, so
  provenance (where each command ran, where each change was written) is always
  captured.
---

# Working on the cloud — rules to follow

These are the standing rules for any cloud/remote work, regardless of project.
Local edits happen where Claude Code runs; runtime happens on the remote box.

## 1. Know where you are before running anything

Claude Code runs on the **local machine**. It is for **editing files and driving
git** only — it usually does NOT have the project's runtime (the right GPU,
`.venv`/conda env, installed packages, services, or data).

When the user says "this machine", "여기서", "이 인스턴스", they often *believe*
Claude is already on the remote box. It is not. **Never silently run a runtime
command locally and report the result as if it came from the remote** — the
output is meaningless. Surface the mismatch and SSH to the real host.

Confirm location before any runtime command:

```bash
uname -srm            # local OS — is this the box you think it is?
hostname              # which machine am I on?
nvidia-smi -L 2>/dev/null || echo "no GPU here"   # if GPU work is involved
```

- **Local-only work:** editing files in the working tree, `git` on the local repo.
- **Remote-only work:** anything that *runs* — imports, version checks, tests,
  benchmarks, servers, installs, hardware/GPU checks.

## 2. Get hosts from `~/.ssh/config`, not from memory

SSH host aliases and their IPs live in `~/.ssh/config`. Cloud IPs **change when
instances restart**, so never hardcode or trust a remembered IP. Re-read the
config to find the right host:

```bash
grep -iE '^Host |HostName' ~/.ssh/config
```

Pick the host by its role/purpose. If the user named a host, use it. If it's
ambiguous which host the task belongs on, **ask** rather than guessing.

## 3. How to run remote commands reliably

The remote env (virtualenv/conda, tools like `uv`, `cargo`, etc.) is often **not
on a non-login shell's PATH**. Two reliable patterns:

**Single command** — use a login shell so the env activates and PATH is set:
```bash
ssh <host> 'bash -lc "cd <project-dir> && source <env>/bin/activate && <command>"'
```

**Multi-line script** — pipe a heredoc to `bash -s` (avoids quoting hell):
```bash
ssh <host> 'bash -s' <<'EOF'
cd <project-dir>
source <env>/bin/activate
export PATH="$HOME/.local/bin:$PATH"   # add tool dirs the login shell may miss
<commands...>
EOF
```

General gotchas:
- A project env may have **no `pip`/no module form of a tool** — use the tool's
  *binary* (e.g. `~/.local/bin/uv`), visible only in a login shell or via PATH.
- Do **not** run a foreground `sleep` to wait on remote output; it can be blocked.
  For long jobs, start them in the background / a log file and poll, don't block.
- Don't launch heavy/long-running services (servers, training, big jobs) unless
  the user explicitly asks — read-only checks (imports, versions) don't need one.

## 4. Package-install safety (don't break pinned environments)

Installing with a blanket upgrade flag (`pip install -U`, `uv pip install -U`)
upgrades the **whole dependency tree**, not just the named package. On
carefully-pinned cloud environments this can pull a generic build over a pinned
one (e.g. a CUDA-pinned `torch`, plus its transitive deps) and break the stack.

Before any install:
1. **Check the installed version first** — the requirement may already be
   satisfied, in which case **no install is needed at all**.
2. Run a **`--dry-run`** and read the full add/remove/upgrade list.
3. If it would change packages **beyond the one you intend**, **stop and ask the
   user** before proceeding. Prefer installing *without* `-U` so an
   already-satisfied constraint is a no-op.

## 5. Always record provenance

Every time you run or change something, make the **where** explicit — both in
your reply and in any doc you update. The user always wants to know: *where was
the work done, and where was the change written.*

In replies, label it plainly, e.g.:
> Ran on **remote** `<host>` (env `<env>`, branch `<branch>` @ `<sha>`).
> Edited file in **local** working tree (uncommitted).

When writing **measured results** into a doc, include a provenance header:
- environment (host alias + instance type + hardware/GPU + IP),
- repo path, branch, and commit SHA the measurement was taken at,
- date,
- the exact commands and their verbatim output (errors in full).

A runtime claim with no "ran on host X @ commit Y" is **not verified** — say so
rather than implying it was measured.

## 6. Don't commit/push unless asked

Edit files in the local working tree and **show the diff**. Commit or push only
on explicit instruction. Default: make the change, report where it landed, let
the user review before anything leaves the machine.
