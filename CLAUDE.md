@AGENTS.md

## vLLM 0.26.0 Exploration Rules

Analysis happens **locally**; execution happens **on the cloud**.

1. **Never run vLLM locally.** No local installs, servers, or benchmarks — code reading and analysis only.
2. **Run only on cloud instances** provisioned via Terraform in [IaC/](IaC/).
3. **Never commit without explicit instruction.** Sync code to the instance one-way (local → remote) via `rsync` — do not use git push/pull for instance sync.

<!-- research-wiki-link -->
## Research wiki
Central research wiki: `/Users/swjeong/research-wiki`.
Use the **research-wiki** skill. Keep project-specific notes in this repo, but
promote general / reusable findings to the central wiki, then commit + push it.
