# QMaxent

**QMaxent** is a QGIS plugin for performing **species distribution modeling (SDM)** using the Maxent algorithm. It provides an intuitive, streamlined workflow to prepare data, train models, and generate prediction maps – all within QGIS.

## ✨ Features

- Import and manage presence/background data
- Extract environmental covariates from raster layers
- Train Maxent models directly in QGIS
- Generate prediction rasters
- Save and load trained models for reuse
- User-friendly graphical interface

## 🚀 Why QMaxent?

Instead of switching between external tools, QMaxent integrates the Maxent modeling process into your QGIS workflow. It’s designed for ecologists, conservationists, and GIS analysts who want to create distribution models without leaving their familiar GIS environment.

## 📦 Installation

Download the plugin and install it through **Plugins > Manage and Install Plugins > Install from ZIP**.

> **Important:**  
> QMaxent requires the `scikit-learn` package to be installed in the **QGIS Python environment**.  
>  
> If you encounter an error such as `ModuleNotFoundError: No module named 'sklearn'`, please install it using the following steps:
>
> ### 🛠 Install `scikit-learn` via OSGeo4W Shell (Recommended)
>
> 1. Close QGIS if it's open.
> 2. Open the **OSGeo4W Shell** (installed with QGIS; search for it in the Start Menu).
> 3. Run the following command (replace the path with your actual Python version if different):
>
> ```bash
> python3 -m pip install scikit-learn
> ```
>
> You should see a message like `Successfully installed scikit-learn ...` when complete.
>
> ✅ Now, reopen QGIS and the plugin should work without import errors.
>
> If `python3` is not recognized, try:
>
> ```bash
> py3_env
> python -m pip install scikit-learn
> ```
>
> ℹ️ These commands install the package directly into the Python environment used by QGIS.

## 📝 License

This project is licensed under the **MIT License**.

## 🙏 Acknowledgments

QMaxent was inspired by the excellent work of [elapid](https://github.com/earth-chris/elapid) by **Christopher Anderson**. Much of the code and functionality in this plugin builds upon the elapid project.

Special thanks to C. Anderson for creating such a powerful foundation for ecological modeling in Python.
