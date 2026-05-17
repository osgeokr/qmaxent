<!-- Thanks for contributing to QMaxent! Please skim CONTRIBUTING.md
     before opening the PR; the checklist below covers most of it. -->

## Summary

<!-- One or two sentences. What does this PR change for the user? -->

## Motivation

<!-- Link the issue this PR closes (Fixes #NN) or describe the bug /
     feature it addresses. -->

## Changes

<!-- Bullet list of the substantive code changes. -->

## Manuscript impact

<!-- QMaxent ships with a companion SoftwareX manuscript whose
     numerical claims (§ 3.3) are pinned in tests/fixtures/. If this
     PR could shift any of those numbers, say so here and regenerate
     the fixture per docs/golden-values.md. Otherwise: "No impact". -->

## Checklist

- [ ] `pytest tests/unit -q` passes locally.
- [ ] If the change can affect numerical output (model fit, threshold
      computation, projection): `pytest tests/regression -q` passes
      *and* the golden-values fixture has been regenerated and committed.
- [ ] If the change touches `core/venv_manager.py:REQUIRED_PACKAGES`,
      `NOTICE` has been updated in the same commit.
- [ ] If the change introduces a new user-visible feature, the user
      manual at https://osgeokr.github.io/qmaxent/manual/ has a
      corresponding update prepared.
- [ ] Version bump done in **all three** of `__init__.py`,
      `metadata.txt`, `CITATION.cff` (only required for release PRs).
