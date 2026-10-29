# Worked examples

Three end-to-end case studies, each highlighting a different aspect of
QMaxent and a different real-world workflow.

## Overview of the examples

| Example | Species | Highlights | What you will learn |
|---|---|---|---|
| [Bradypus variegatus](bradypus.md) | Three-toed sloth | Bundled dataset; 8 bioclimatic variables + biome, 116 presences | The complete QMaxent feature tour — same workflow as the original Maxent paper |
| [Ariolimax](ariolimax.md) | Pacific banana slug | Mismatched rasters | The **Check Raster Consistency** + **Harmonize to Folder…** preflight |
| [Pitta nympha](pitta-nympha.md) | Fairy pitta | Reproduction of [Lee et al. 2025](../references.md) | How to reproduce a published study in QMaxent and compare results with maxent.jar on identical inputs |

The three examples are designed to be **completed in order**: Bradypus
introduces the workflow, Ariolimax adds raster harmonisation, Pitta
nympha adds publication-quality reproduction with academic comparison.

## How to follow along

Every example assumes you have completed
[Installation](../installation.md) and that the **QMaxent environment
ready** banner is green.

Two of the three datasets are bundled with the plugin — load them via
**Plugins → QMaxent → Download Example Dataset → \<species name\>**:

- **Bradypus variegatus**: 9 variables (8 bioclimatic variables and a
  categorical biome raster), 116 presences, ships with the plugin.
  Bit-identical to the Phillips et al. 2006 dataset as distributed with
  the R package dismo.
- **Ariolimax** — 6 variables, 3,732 presences, the elapid library's
  default Pacific-coast slug dataset. **The official download is
  pre-harmonised** — to follow the Ariolimax worked example exactly,
  you will need to use the deliberately desynchronised version
  described in the example's header.
- **Pitta nympha**: 10 variables (2 categorical), 47 nests in
  Geoje-si, South Korea.
  Sourced from the supplementary archive of Lee et al. 2025; not
  bundled with the plugin (licence considerations).

## Reproducibility

All three examples are run with **fixed random seeds**. Since v0.2.0
the seed also fixes the background sample, so with the same inputs,
the same raster order and the same settings, a run reproduces the same
numbers. Some screenshots and numbers in the Bradypus
and Ariolimax chapters were recorded with v0.1.x (LQPHT under Auto,
unseeded background) and differ slightly from what v0.2.0 gives; those
places are marked.
