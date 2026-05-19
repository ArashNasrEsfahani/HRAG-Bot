---
name: update-repo
description: Use this skill when the user wants to push the current state of the HRAG-Bot project to its GitHub repo — phrases like "update my repo", "push to github", "save my changes", "sync the repo", "commit and push", or "/update-repo". The skill inspects the working tree, drafts a commit message from the actual diff, commits, and pushes to `origin`. It is project-scoped (HRAG-Bot) and assumes git is already initialised with a remote.
---

# update-repo skill (HRAG-Bot)

## Why this exists

The user wants a one-command path to sync local changes to their GitHub repo without dictating commit messages or remembering git incantations. The skill enforces a small set of safety rules so this never destroys work or leaks secrets.

## What this skill does

1. **Inspect** — show `git status` and a `git diff --stat` so the user sees what's about to be committed.
2. **Draft** — write a commit message from the actual diff. Group related files; explain *why* in 1–2 sentences max. Follow the repo's existing commit-message style (run `git log --oneline -10` to check).
3. **Confirm** — print the draft message AND the file list. Ask the user to approve, edit, or split. If the diff is trivially small and clearly safe (single-file typo, one-line fix), the skill MAY commit without a second prompt — but never push without showing what's going up.
4. **Stage + commit** — `git add` only the files in the message (avoid `git add -A` so stray scratch files don't sneak in). Use a HEREDOC commit body so newlines are preserved.
5. **Push** — `git push` to the current branch's upstream. If the branch has no upstream, use `git push -u origin <current-branch>`.
6. **Report** — print the final commit hash and the GitHub URL of the new commit (derive from `git remote get-url origin`).

## Hard rules

- **Never `git add -A` or `git add .`.** Stage files by name. Untracked files that aren't part of the user's intent stay untracked.
- **Never `--force` or `--force-with-lease`** unless the user explicitly asks. The default push must be a plain fast-forward.
- **Never `--amend`** an already-pushed commit. Amending main is destructive once the remote has the old hash.
- **Never `--no-verify`** to skip hooks. If a hook fails, fix the underlying issue or report it to the user.
- **Refuse to stage files that look like secrets** — `.env`, `*.pem`, `*.key`, anything matching `*credentials*`, `*secret*`, `*token*`. If the user explicitly insists, warn loudly and proceed only after confirmation.
- **Don't commit large binaries the .gitignore should have caught.** If you see a file >5MB about to be committed (e.g. `*.sqlite`, `*.bin`, `*.pkl`, model weights), pause and ask. Add it to .gitignore instead of committing.
- **Refuse to push to `main` if the working tree has unrelated unstaged changes.** Either commit them in a separate commit (with the user's approval) or stash them.

## Commit-message style

The repo's existing style (see `git log --oneline -20`) is:

- First line: imperative, ≤72 chars, no trailing period. Examples:
  - `Fix taxonomy doc-count stale after delete`
  - `Add Nougat PDF loader scaffold`
  - `Phase 7-A: math-aware retrieval (filter + extraction)`
- Blank line, then a 1–3-line body explaining the *why* if the change isn't self-evident.
- Bullet body when bundling related but distinct edits.
- Always end with the standard co-author trailer:

  ```
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  ```

## Workflow (the exact commands)

```bash
# 1. Inspect — never skip this
git -C "D:/Hierachical RAG based ChatBot" status
git -C "D:/Hierachical RAG based ChatBot" diff --stat
git -C "D:/Hierachical RAG based ChatBot" log --oneline -5   # style reference

# 2. Stage chosen files (by name, not -A)
git -C "D:/Hierachical RAG based ChatBot" add path/to/file1 path/to/file2

# 3. Commit with HEREDOC (preserves newlines / quotes)
git -C "D:/Hierachical RAG based ChatBot" commit -m "$(cat <<'EOF'
<imperative subject ≤72 chars>

<one-paragraph body explaining why, when not obvious>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"

# 4. Push (uses the upstream the branch is already tracking)
git -C "D:/Hierachical RAG based ChatBot" push

# 4-bis. If no upstream yet (first push of a new branch):
# git -C "D:/Hierachical RAG based ChatBot" push -u origin <branch>

# 5. Report
git -C "D:/Hierachical RAG based ChatBot" log -1 --format='%h %s'
git -C "D:/Hierachical RAG based ChatBot" remote get-url origin
# Print: https://github.com/<user>/<repo>/commit/<hash>
```

## When the user gives an argument

`/update-repo <message>` — treat the argument as the commit subject. Still inspect the diff first; if the diff doesn't match the message, push back and ask. Don't silently override the user's intent.

`/update-repo --branch <name>` — create or check out the branch first, commit there, push with `-u`. Useful for opening a PR later.

## When something goes wrong

- **Pre-commit hook fails** — Do NOT pass `--no-verify`. Read the hook's output, fix the underlying issue, re-stage the fix, create a NEW commit (do not `--amend` if the broken commit was already created).
- **Push rejected (non-fast-forward)** — Someone pushed to the remote since the last fetch. Run `git fetch && git status` and report to the user. Offer to `git pull --rebase` if the local changes are clearly newer; never `--force`.
- **Authentication fails** — Git Credential Manager normally pops a browser window. If running headless, tell the user to run the same `git push` from their own terminal once and the credential will be cached for future automated pushes.
- **`git remote get-url origin` is empty** — The repo has no remote yet. Tell the user to create the GitHub repo (https://github.com/new) and paste the URL, then run `git -C "<repo>" remote add origin <url>`.

## What this skill does NOT do

- Open pull requests. (The user can do this manually on github.com, or install `gh` CLI later if they want it automated.)
- Tag releases.
- Bump version numbers in `pyproject.toml`.
- Modify branches other than the current one.

If the user wants any of the above, do it as a normal task — not via this skill.
