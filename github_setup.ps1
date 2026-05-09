# QMaxent — GitHub repo bootstrap (Windows PowerShell)
# Run from inside the qmaxent\ folder.
# Requires: gh CLI (winget install GitHub.cli) and git already installed.

# Note: we intentionally do NOT set $ErrorActionPreference = "Stop", because
# native CLI tools (gh, git) often write to stderr on harmless conditions
# (e.g. "not logged in", "nothing to commit"), which would abort the script.
# Instead we check $LASTEXITCODE explicitly after each call.

$RepoOwner = "osgeokr"
$RepoName  = "qmaxent"
$RemoteUrl = "https://github.com/$RepoOwner/$RepoName.git"

function Stop-OnFail($msg) {
    Write-Host "[X] $msg" -ForegroundColor Red
    exit 1
}

# 0. Sanity check
if (-not (Test-Path "metadata.txt") -or -not (Test-Path "docs")) {
    Stop-OnFail "Run this script from the qmaxent\ folder (must contain metadata.txt and docs\)."
}

# 1. Authenticate gh CLI (run gh auth login if not already authenticated)
& gh auth status *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "-> Logging in to GitHub..." -ForegroundColor Cyan
    & gh auth login
    if ($LASTEXITCODE -ne 0) { Stop-OnFail "gh auth login failed. Re-run the script after logging in." }
} else {
    Write-Host "-> Already authenticated with gh." -ForegroundColor DarkGray
}

# 2. Init + configure + commit
if (-not (Test-Path ".git")) { & git init -b main | Out-Null }

# Quiet the harmless CRLF warnings on Windows
& git config core.autocrlf true | Out-Null

# Make sure git knows who's committing — required for the very first commit.
$gitName  = (& git config user.name)  2>$null
$gitEmail = (& git config user.email) 2>$null
if ([string]::IsNullOrWhiteSpace($gitName))  {
    & git config user.name  "Byeong-Hyeok Yu" | Out-Null
    Write-Host "-> git user.name set to 'Byeong-Hyeok Yu' (local repo)." -ForegroundColor DarkGray
}
if ([string]::IsNullOrWhiteSpace($gitEmail)) {
    & git config user.email "bhyu@knps.or.kr" | Out-Null
    Write-Host "-> git user.email set to 'bhyu@knps.or.kr' (local repo)." -ForegroundColor DarkGray
}

& git add . | Out-Null

# Commit only if there is something to commit. Don't silence real errors.
$pending = & git status --porcelain
if (-not [string]::IsNullOrWhiteSpace($pending)) {
    & git commit -m "Initial commit: QMaxent v0.1.0 + landing page"
    if ($LASTEXITCODE -ne 0) { Stop-OnFail "git commit failed." }
} else {
    Write-Host "-> Nothing to commit (working tree clean)." -ForegroundColor DarkGray
}

# Verify at least one commit exists before pushing
& git rev-parse --verify HEAD *> $null
if ($LASTEXITCODE -ne 0) { Stop-OnFail "No commits in this repo. Cannot push." }

# 3. Create remote (or push to existing)
& gh repo view "$RepoOwner/$RepoName" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "-> Creating GitHub repo $RepoOwner/$RepoName..." -ForegroundColor Cyan
    & gh repo create "$RepoOwner/$RepoName" `
        --public `
        --description "QGIS plugin for Maxent species distribution modeling (SDM)" `
        --homepage "https://$RepoOwner.github.io/$RepoName/" `
        --source=. --remote=origin --push
    if ($LASTEXITCODE -ne 0) { Stop-OnFail "gh repo create failed." }
} else {
    Write-Host "-> Repo already exists; pushing to existing remote." -ForegroundColor Yellow
    & git remote add origin $RemoteUrl *> $null
    if ($LASTEXITCODE -ne 0) { & git remote set-url origin $RemoteUrl }
    & git push -u origin main
    if ($LASTEXITCODE -ne 0) { Stop-OnFail "git push failed." }
}

# 4. Enable Pages from /docs (POST creates, PUT updates)
Write-Host "-> Enabling GitHub Pages (main branch, /docs folder)..." -ForegroundColor Cyan
& gh api -X POST "repos/$RepoOwner/$RepoName/pages" `
    -f "source[branch]=main" -f "source[path]=/docs" *> $null
if ($LASTEXITCODE -ne 0) {
    & gh api -X PUT "repos/$RepoOwner/$RepoName/pages" `
        -f "source[branch]=main" -f "source[path]=/docs" *> $null
}

# 5. Topics
& gh api -X PUT "repos/$RepoOwner/$RepoName/topics" `
    -F 'names[]=qgis' -F 'names[]=qgis-plugin' `
    -F 'names[]=maxent' -F 'names[]=sdm' `
    -F 'names[]=species-distribution-modeling' `
    -F 'names[]=ecology' -F 'names[]=elapid' -F 'names[]=python' *> $null

Write-Host ""
Write-Host "[OK] Done." -ForegroundColor Green
Write-Host "  Repo:  https://github.com/$RepoOwner/$RepoName"
Write-Host "  Pages: https://$RepoOwner.github.io/$RepoName/  (first build ~1 minute)"
