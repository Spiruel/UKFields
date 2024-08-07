# UKFields code

This document provides a brief description of the purposes of the scripts in the current directory.

## 1_export_harmonics.py

Produces harmonic composites over the UK, masked to arable areas. Exports to gdrive as overlapping patches.

## 2_tfrecord_to_tiff.py

Converts the tfrecords to more accessible geotiffs for processing.

## 3_sam_inference.py

Uses segment anything to produce prediction rasters

## 4_vectorise_preds.py

Vectorises and merges the overlapping matches using radial weighting.