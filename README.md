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
> QMaxent requires the following Python packages to be installed in the **QGIS Python environment**:  
> - `scikit-learn`  
> - `rasterio`  
> - `tqdm`
>  
> If you encounter errors such as `ModuleNotFoundError: No module named 'sklearn'`, `rasterio`, or `tqdm`, please install them using the steps below:

### 🛠 Install Required Packages via OSGeo4W Shell (Recommended)

1. Close QGIS if it's open.  
2. Open the **OSGeo4W Shell** (installed with QGIS; search for it in the Start Menu).  
3. Run the following command (replace the Python version path if necessary):

   ```bash
   python3 -m pip install scikit-learn rasterio tqdm

## 📝 License

This project is licensed under the **MIT License**.

## 🙏 Acknowledgments

QMaxent was inspired by the excellent work of [elapid](https://github.com/earth-chris/elapid) by **Christopher Anderson**. Much of the code and functionality in this plugin builds upon the elapid project.

Special thanks to C. Anderson for creating such a powerful foundation for ecological modeling in Python.
