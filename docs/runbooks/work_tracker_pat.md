# Work Tracker PAT Setup

The `.github/workflows/add-to-project.yml` workflow auto-adds issues
labelled `tracked` to the user-level Work Tracker project
(`https://github.com/users/brandon-behring/projects/1`). GitHub's
default `GITHUB_TOKEN` cannot write to user-owned projects, so the
action uses a Personal Access Token (PAT) stored as the repo secret
`ADD_TO_PROJECT_PAT`.

This runbook documents how to create and rotate that PAT with the
**minimum** scopes needed, following the principle of least privilege.

## Create the PAT (one-time setup)

1. Navigate to **GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens**
   (direct link: <https://github.com/settings/personal-access-tokens>).
2. Click **Generate new token**.
3. Fill in the form:

   | Field | Value |
   |---|---|
   | **Token name** | `research-kb add-to-project` |
   | **Expiration** | 1 year (set calendar reminder to rotate) |
   | **Resource owner** | `brandon-behring` (your user account) |
   | **Repository access** | **Only select repositories** → choose `brandon-behring/research-kb` |
   | **Repository permissions → Issues** | **Read-only** |
   | **Account permissions → Projects** | **Read and write** |

   No other scopes are needed. Leave everything else on the default
   "No access".

4. Click **Generate token** and copy the token. You will not be able
   to view it again — paste it immediately into step 5.

## Add the token as a repo secret

1. Go to the repo: **Settings → Secrets and variables → Actions**.
2. Under **Repository secrets**, click **New repository secret**.
3. Name: `ADD_TO_PROJECT_PAT`. Value: the PAT from step 4 above.
4. Click **Add secret**.

## Verify the workflow

After the secret is in place, trigger the workflow by labelling any
existing issue with `tracked`:

```bash
gh issue edit <N> --repo brandon-behring/research-kb --add-label tracked
```

Then check the Actions tab for the `Add tracked issues to Work Tracker`
workflow run, and verify the issue appears in the project:

```bash
gh project item-list 1 --owner brandon-behring --format json \
  | jq '.items[] | select(.content.number == <N>)'
```

## Rotation

When the token is near expiry (GitHub emails ~7 days before), repeat
the "Create the PAT" steps and update the existing
`ADD_TO_PROJECT_PAT` secret with the new value. No code changes are
needed — the workflow just re-reads the secret.

## Why fine-grained over classic PAT

- Classic PATs use broad scopes (`repo`, `project`) that grant access
  to **all** your repositories and projects. A leaked classic PAT is
  a whole-account compromise.
- Fine-grained PATs let us scope the token to **just this repo** for
  issue reads and to user-level project write. A leaked token's blast
  radius is limited to listing issues and adding items to projects —
  no code read/write, no PR creation, no other-repo access.

For a small personal account the blast radius difference is modest,
but the habit is worth establishing and rotations are lower-friction
when scopes are clear.
