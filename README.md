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
> QMaxent requires the `scikit-learn` package to be available in the **QGIS Python environment**.  
>  
> If you see an error like `ModuleNotFoundError: No module named 'sklearn'`, please install it as follows:
>
> 1. Open **QGIS**
> 2. Go to **Plugins > Python Console**
> 3. Paste and run the following code:
>
> ```python
> import sys
> import subprocess
> subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-learn'])
> ```
>
> This will install `scikit-learn` into the QGIS environment.  
> If installation fails, ensure you have an active internet connection and sufficient permissions.

## 📝 License

This project is licensed under the **MIT License**.

## 🙏 Acknowledgments

QMaxent was inspired by the excellent work of [elapid](https://github.com/earth-chris/elapid) by **Christopher Anderson**. Much of the code and functionality in this plugin builds upon the elapid project.

Special thanks to C. Anderson for creating such a powerful foundation for ecological modeling in Python.
