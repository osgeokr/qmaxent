# Installation

QMaxent installs as a regular QGIS plugin. After the plugin itself is in
place, a one-time dependency setup downloads the Python libraries Maxent
needs into a separate QMaxent folder that does not affect your system
Python or QGIS itself. On Linux and macOS this folder is a Python virtual
environment; on Windows it is a plain package folder filled by the QGIS
Python's own pip (see [Dependencies](dependencies.md)).

## System requirements

- **QGIS** 3.44 LTR or later
- **Operating system**: Windows 10/11, macOS 12+, or any modern Linux distribution
- **Disk space**: about 600 MB free for the QMaxent package folder
- **Internet access** is required only on first run to download the Python
    dependencies; routine model fitting and projection run fully offline.

## Install from the QGIS Plugin Repository

QMaxent is published on the
[official QGIS Plugin Repository](https://plugins.qgis.org/plugins/qmaxent/).

1. Open QGIS and choose **Plugins → Manage and Install Plugins…**
2. In the **All** tab, type `qmaxent` in the search box. The plugin
    appears in the result list.
3. Click **Install Plugin**.

Once installed, three entry points appear under **Plugins → QMaxent**:

![Plugins → QMaxent menu showing the three entry points: Analysis, Dependencies, Download Example Dataset](images/ui/menu-plugins-qmaxent.png)

| Menu item | What it does |
|---|---|
| **QMaxent Analysis** | Opens the main analysis dock — the heart of the plugin |
| **QMaxent Dependencies** | Opens the dependency-management dialog (next chapter) |
| **Download Example Dataset** | Bundled SDM datasets for tutorials (see *Example datasets*) |

## Verifying the installation

Choose **Plugins → QMaxent → QMaxent Dependencies**. If the dialog opens,
the plugin is correctly installed. The **Environment Status** banner will
most likely read *Dependencies not installed* the first time — that is
expected and is dealt with in the next chapter.

## Updating to a newer version

When a new release ships, the QGIS Plugin Manager will mark QMaxent as
upgradable. Click **Upgrade Plugin** in the **Installed** tab. Dependencies
generally do not need to be reinstalled across patch releases (0.1.x →
0.1.y). When the package list or the installation method changes, the
**Dependencies** dialog reports that the dependencies need updating; click
**Install / Update Dependencies** once more. This is the case when you
upgrade to 0.2.0 on Windows, because 0.2.0 changed how packages are
installed there.

## Troubleshooting

??? warning "QMaxent does not appear in the search results"
    Make sure your QGIS version is 3.44 or later (`Help → About QGIS`).
    On older versions QMaxent is filtered out as incompatible. Also confirm
    the **Show experimental plugins** option in **Settings** is set to
    your preference — QMaxent's stable releases do not require it.

??? warning "Plugin installs but the menu entry does not appear"
    Restart QGIS. Some QGIS versions require a restart for top-level menu
    entries from new plugins to register.

??? warning "Windows: installation stops with 'ensurepip ... returned non-zero exit status 1'"
    QMaxent 0.1.x created a Python virtual environment, and this step
    fails with QGIS 3.44.15 and later on Windows. QMaxent 0.2.0 no longer
    needs a virtual environment on Windows: it installs the packages with
    the QGIS Python's own pip into the QMaxent package folder. Upgrade the
    plugin to 0.2.0 or later and click **Install / Update Dependencies**
    again. Details are in [Dependencies](dependencies.md).

??? warning "Corporate proxy or restricted network"
    The plugin install itself uses QGIS's network stack and respects
    QGIS's proxy settings. The dependency installer (next chapter)
    downloads from [PyPI](https://pypi.org); if your organisation blocks
    PyPI, ask your network administrator to allow `pypi.org` and
    `files.pythonhosted.org`.

Once installed, continue to [Dependencies](dependencies.md).
