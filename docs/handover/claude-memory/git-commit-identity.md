---
name: git-commit-identity
description: Always commit as unimatrix099 <unimatrix099@github.com>; never use the session/profile email
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c33715ae-46cf-49ba-a859-668d5fe50c93
  modified: 2026-09-04T11:20:42.694Z
---

All git commits must be authored as `unimatrix099 <unimatrix099@github.com>`. Never use the email address that appears in the session context / user profile, and never let it reach a commit, reflog, or any file.

**Why:** The profile email is a work address the user does not want attached to this repo's history; they had to ask for a commit to be rewritten and the reflog purged after it leaked in.

**How to apply:** If a repo has no git identity configured, set it repo-locally (`git config user.name unimatrix099` / `git config user.email unimatrix099@github.com`) before committing rather than guessing from context. If the wrong identity ever lands, `git commit --amend --reset-author` with the right `GIT_AUTHOR_*`/`GIT_COMMITTER_*`, then `git reflog expire --expire=now --all && git gc --prune=now` to drop the stale entry and object.
