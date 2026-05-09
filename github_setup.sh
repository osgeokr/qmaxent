#!/usr/bin/env bash
# QMaxent — GitHub repo bootstrap
# Run from inside the qmaxent/ folder you received from Claude.
# Requires: gh CLI (https://cli.github.com/) and git already installed.

set -euo pipefail

REPO_OWNER="osgeokr"
REPO_NAME="qmaxent"
REMOTE_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}.git"

# 0. Sanity check: are we in the qmaxent folder?
if [ ! -f "metadata.txt" ] || [ ! -d "docs" ]; then
  echo "✗ Run this script from the qmaxent/ folder (the one containing metadata.txt and docs/)."
  exit 1
fi

# 1. Authenticate gh CLI (browser-based; one-time)
if ! gh auth status >/dev/null 2>&1; then
  echo "→ Logging in to GitHub..."
  gh auth login
fi

# 2. Initialize git
if [ ! -d ".git" ]; then
  git init -b main
fi
git add .
git commit -m "Initial commit: QMaxent v0.1.0 + landing page" || echo "(nothing to commit)"

# 3. Create the remote repo (public). Skip if it already exists.
if ! gh repo view "${REPO_OWNER}/${REPO_NAME}" >/dev/null 2>&1; then
  gh repo create "${REPO_OWNER}/${REPO_NAME}" \
    --public \
    --description "QGIS plugin for Maxent species distribution modeling (SDM)" \
    --homepage "https://${REPO_OWNER}.github.io/${REPO_NAME}/" \
    --source=. --remote=origin --push
else
  echo "→ Repo already exists; pushing to existing remote."
  git remote add origin "${REMOTE_URL}" 2>/dev/null || git remote set-url origin "${REMOTE_URL}"
  git push -u origin main
fi

# 4. Enable GitHub Pages from /docs on main branch
echo "→ Enabling GitHub Pages (main branch, /docs folder)..."
gh api -X POST "repos/${REPO_OWNER}/${REPO_NAME}/pages" \
  -f "source[branch]=main" \
  -f "source[path]=/docs" \
  >/dev/null 2>&1 \
  || gh api -X PUT "repos/${REPO_OWNER}/${REPO_NAME}/pages" \
       -f "source[branch]=main" \
       -f "source[path]=/docs" >/dev/null

# 5. Add useful repo topics
gh api -X PUT "repos/${REPO_OWNER}/${REPO_NAME}/topics" \
  -F 'names[]=qgis' -F 'names[]=qgis-plugin' \
  -F 'names[]=maxent' -F 'names[]=sdm' \
  -F 'names[]=species-distribution-modeling' \
  -F 'names[]=ecology' -F 'names[]=elapid' \
  -F 'names[]=python' >/dev/null

cat <<EOM

✓ Done.
  • Repo:  https://github.com/${REPO_OWNER}/${REPO_NAME}
  • Pages: https://${REPO_OWNER}.github.io/${REPO_NAME}/
    (the first build takes ~1 minute — refresh after a moment)
EOM
