# Building the QGIS plugin release ZIP

The ZIP uploaded to the QGIS Plugin Repository should contain **only**
the files the plugin needs at runtime. Repository metadata
(`CITATION.cff`, `NOTICE`, `CONTRIBUTING.md`, `.zenodo.json`, the test
suite, GitHub configuration, etc.) is useful for collaborators on
GitHub but adds no value once the plugin is installed inside QGIS.

From **v0.1.8 onward**, the per-file exclusions are declared in
`.gitattributes` via `export-ignore` directives, which lets
`git archive` produce the correct release ZIP from a clean tag with
one command.

## One-command release build

From the repository root with a clean working tree:

```bash
git archive --format=zip --prefix=qmaxent/ HEAD > qmaxent-0.1.8.zip
```

The output ZIP unpacks to a single top-level `qmaxent/` directory
containing exactly the runtime tree the QGIS plugin manager expects.
Upload that ZIP via https://plugins.qgis.org/plugins/qmaxent/version/add/.

## What ends up inside qmaxent-X.Y.Z.zip

After applying the `export-ignore` rules in `.gitattributes`, the ZIP
contains the following and nothing else:

```
qmaxent/
├── LICENSE
├── README.md
├── __init__.py
├── metadata.txt
├── qmaxent_plugin.py
├── bridge/
├── core/
├── dialogs/
├── i18n/
├── icons/
└── workers/
```

The QGIS Plugin Repository reads `metadata.txt` and resolves
`__init__.py` → `qmaxent_plugin.py` → the submodules above. Nothing
else is needed for the plugin to load and run.

## What is intentionally NOT included

| Excluded | Why it's safe to exclude |
|---|---|
| `CITATION.cff`, `.zenodo.json` | Read by GitHub and Zenodo, never by the plugin runtime. |
| `NOTICE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` | Contributor-facing documents; not referenced by any plugin code. |
| `.github/`, `.gitignore`, `.gitattributes` | Repository plumbing. |
| `tests/`, `pyproject.toml`, `requirements-test.txt` | Test runner config; QGIS users do not run pytest inside their profile. |
| `docs/`, `manual/` | Source of the published website (https://osgeokr.github.io/qmaxent/). The rendered manual is online; users do not need the raw mkdocs source. |
| `github_setup.*` | One-off repository configuration scripts run by the maintainer. |

## Verifying the build before upload

The plugin should round-trip cleanly:

```bash
# Build the ZIP.
git archive --format=zip --prefix=qmaxent/ HEAD > /tmp/qmaxent-test.zip

# Inspect contents.
unzip -l /tmp/qmaxent-test.zip
# → expect ~30-40 entries, all rooted at qmaxent/, no .github/ no tests/.

# Simulate installation by extracting into a QGIS profile and
# launching QGIS. The plugin should load and the dock should open.
```

## Why `.gitattributes`-based exclusion, not a bespoke build script

- One source of truth — when a new top-level file is added (e.g. a
  future `SECURITY.md`), the maintainer toggles it in `.gitattributes`
  instead of editing a parallel build script.
- No extra tooling — `git archive` is built into git and works on
  every developer machine.
- The same ZIP can be reproduced exactly from any tagged commit, which
  is a property that auditors (Zenodo and academic reviewers) value.

## Releasing as part of the GitHub Release flow

When publishing a GitHub Release (per `docs/zenodo-release.md`),
attach the `git archive`-produced ZIP as a release asset. Both the
QGIS Plugin Repository and Zenodo can then point at the same artefact.
