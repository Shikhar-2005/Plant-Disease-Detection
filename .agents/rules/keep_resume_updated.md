# Rule: Always Keep RESUME.md Up To Date

## When This Applies
This rule applies to **every task** performed in the `Plant Disease Detection` project.

## Mandatory Behaviour

After completing **any** of the following actions, you MUST update `d:\Projects\Plant Disease Detection\RESUME.md`:

- Creating, modifying, or deleting a file in the project
- Completing or partially completing a task from the MASTER TASK LIST
- Installing dependencies or setting up the environment
- Downloading or processing the dataset
- Training or evaluating the model
- Deploying the app or pushing to GitHub
- Discovering a new issue or gotcha
- Receiving new instructions or requirements from the user/supervisor

## What To Update In RESUME.md

1. **`## 📍 Project Status:`** — Change to reflect current phase (e.g., `PHASE 1 IN PROGRESS`, `PHASE 2 COMPLETE`, etc.)
2. **`## 🕐 Last Updated:`** — Change to current date and time (IST)
3. **`## ✅ MASTER TASK LIST`** — Mark completed items with `[x]`, in-progress with `[/]`, pending with `[ ]`
4. **`## Files Created So Far`** — Add any new files created (if section exists)
5. **`## Model Performance`** — Fill in metrics once training is done
6. **`## Known Issues / Gotchas`** — Add any new issues discovered
7. **`## Change Log`** — Add a new row with the current date and a brief description of what changed

## Format For Change Log Entry

```
| YYYY-MM-DD | Brief description of what was done |
```

## Why This Matters

This file is the **single source of truth** for any agent resuming work in a new session.
If it is not kept up to date, the next agent will waste tokens re-reading old files and may redo completed work.

Keeping this file current is **not optional** — treat it as a mandatory final step of every task.
