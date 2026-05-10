# QMaxent Manual

Welcome to the QMaxent user manual. **QMaxent** integrates the
[elapid](https://github.com/earth-chris/elapid) Python library into QGIS to
deliver the full Maxent species distribution modeling (SDM) workflow —
training, spatial cross-validation, jackknife variable importance,
projection, and survey planning — through a familiar bilingual interface.

## Where to start

If this is your first visit:

1. Read [Introduction](introduction.md) for the 2-minute orientation
2. Install the plugin → [Installation](installation.md)
3. Set up Python dependencies → [Dependencies](dependencies.md)
4. Download the bundled example data → [Example datasets](example-datasets.md)
5. Walk through your first model in 5 minutes → [Quick start](quick-start.md)

If you already use QMaxent and want to dig into a specific feature, the
[User guide](analysis-dock.md) chapters mirror the plugin's five tabs
one-for-one. Each can be read on its own.

If you want to see the tool put through its paces on real datasets, the
[Worked examples](examples/index.md) cover three case studies of increasing
complexity, including a reproduction of the Lee et al. (2025) fairy pitta
study published in *Global Ecology and Conservation*.

## Conventions used in this manual

- **Bold** marks UI labels (buttons, tabs, menu items) you click.
- `monospace` marks file paths, code, and parameter values you type.
- Numbered tabs (**①**, **②**, …) refer to the QMaxent Analysis dock tabs.
- Inline citations like *(Phillips et al. 2017)* point to entries in the
    [References](references.md) chapter.
- Sidebar boxes flag warnings (⚠), tips (💡), and notes worth a second
    glance — but most of the manual is plain prose; we do not use boxes
    decoratively.

## Citation

If you use QMaxent in your research, please cite the software using the
metadata in the repository's
[`CITATION.cff`](https://github.com/osgeokr/qmaxent/blob/main/CITATION.cff).
The same metadata is rendered live on the project home page at
[osgeokr.github.io/qmaxent](https://osgeokr.github.io/qmaxent/) and in the
[References](references.md) chapter of this manual.
