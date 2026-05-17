# Zenodo release workflow

QMaxent is archived on Zenodo via GitHub's native integration. This
document records the exact sequence used for every release, so a
release-day operator (or future maintainer) does not have to
reconstruct it from memory.

## One-time setup (already done for v0.1.7)

These steps need to happen only once for the repository. Recorded
here for posterity and in case the integration ever has to be
re-established.

1. Sign in to https://zenodo.org with the **same identity** that owns
   the GitHub repository (or has admin access to it).
2. Go to **Settings → GitHub** (left sidebar) and toggle the
   `osgeokr/qmaxent` repository **ON**. From this moment, every
   published GitHub Release on the repo will be archived as a
   Zenodo deposition automatically.
3. Confirm `.zenodo.json` exists at the repository root (this repo).
   Zenodo reads it on every release and uses it to fill in title,
   keywords, contributors, etc. so the deposition does not start
   blank.

## Pre-flight: getting a DOI ready *before* SoftwareX acceptance

The SoftwareX manuscript needs to cite the released software (per
reference [31] of the manuscript). Acceptance-day workflow:

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│ A. Pre-release      │ →  │ B. Tag + Publish    │ →  │ C. Patch metadata    │
│    sanity check     │    │    GitHub Release   │    │    with issued DOI   │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
   (this checklist)         (mints the DOI)            (closes the loop)
```

### A. Pre-release sanity check

Run from the repository root with a clean working tree:

```bash
# 1. Version triple-check. Must match the tag you're about to push.
python -m pytest tests/unit/test_version_consistency.py -q

# 2. Full unit + regression suite.
python -m pytest -q

# 3. CFF validation. Catches YAML / DOI-placeholder errors locally,
#    before GitHub Actions does.
pipx run cffconvert -i CITATION.cff --validate

# 4. Manuscript citation cross-check.
#    Open the manuscript, find reference [31], confirm version + URL
#    match what's about to be tagged.
```

### B. Tag and publish the GitHub Release

```bash
git tag -a v0.1.7 -m "QMaxent v0.1.7 (SoftwareX submission)"
git push origin v0.1.7
```

Then on GitHub:

1. **Releases → Draft a new release.**
2. Select the `v0.1.7` tag.
3. Title: `QMaxent v0.1.7 (SoftwareX submission)`.
4. Body: short release notes (highlights, breaking changes if any,
   acknowledgements). Keep this prose-only; Zenodo will combine it
   with the rich description in `.zenodo.json`.
5. Click **Publish release**.
6. Within ~5 minutes Zenodo's webhook fires and the deposition
   appears at https://zenodo.org/account/settings/github/repository/osgeokr/qmaxent.
   Two DOIs are minted:
     - **Concept DOI** (all versions) — pin this in `CITATION.cff`'s
       first `identifiers` entry.
     - **Version DOI** (this specific release) — pin in the second
       `identifiers` entry and in the BibTeX block of `README.md`.

### C. Patch metadata with the issued DOI

The DOIs were placeholders until step B finished. Now:

1. Edit `CITATION.cff`: replace the two `10.5281/zenodo.XXXXXXX`
   placeholders with the issued values.
2. Edit `README.md`: replace the same placeholder in the Zenodo
   badge, the APA-style citation, and the BibTeX block.
3. Commit on `main` with message:
   `Pin Zenodo DOI 10.5281/zenodo.NNNNNNN for v0.1.7`.
4. **Do not** re-tag `v0.1.7`. The published Zenodo archive is
   immutable; the post-release commit is the authoritative pointer
   for future readers and is what the manuscript will reference at
   galley-proof time.

### D. Cite the issued DOI in the SoftwareX manuscript

At galley-proof stage, replace reference [31]'s placeholder DOI with
the issued Zenodo version DOI. The manuscript's "Code metadata" table
(SoftwareX template) gets the same DOI as the *Software Code
Identifier*.

## Sanity nets that catch the most common mistakes

- `.github/workflows/release-zenodo.yml` runs on every `v*.*.*` tag
  push and **fails the workflow** if the tag does not match the
  versions inside `__init__.py`, `metadata.txt`, and `CITATION.cff`.
  This prevents the most common release-day mistake: bumping the
  tag without bumping the file metadata.
- `tests/unit/test_version_consistency.py` enforces the same check
  on every push, so the mismatch is caught long before tag time.
- `tests/regression/test_maxent_jar_compatibility.py` asserts that
  `pitta_golden_values.json` was regenerated for the current
  `__version__` — making a release with stale golden values is
  caught by CI.

## When to mint a release candidate (rc) instead

If you want to dry-run the Zenodo path without committing to a real
release (e.g. you want to verify that the deposition's title and
description render correctly), use a tag of the form `v0.1.7-rc1`.
Zenodo will treat it as a normal release and mint a DOI — that DOI
is a real, citable record forever, so:

- Only do this when you actually want to test the integration.
- Update `.zenodo.json` with a clear "RC" note so anyone who
  stumbles on the rc DOI sees it is not the canonical record.

For the SoftwareX submission, the rc round is optional; the existing
release-zenodo.yml sanity-check workflow is sufficient assurance.
