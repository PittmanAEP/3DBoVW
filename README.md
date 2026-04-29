# 3D BoVW for Volumetric Microscopy Analysis

## Overview

This repository contains a two-step, interpretable analysis pipeline for 3D microscopy datasets. The workflow is designed to segment volumetric cell crops, extract local 3D image features, convert those features into a bag-of-visual-words representation, and compare biological conditions using both unsupervised structure and supervised, interpretable classification. Step 1 handles segmentation, crop filtering, metadata generation, and QC app creation, while Step 2 performs multiscale keypoint detection, custom 3D HOG feature extraction, sparse dictionary learning, image-level feature aggregation, logistic-regression-based classification, attention-map generation, Haralick texture analysis, and PCA/UMAP visualization.

## What this project does

- Segments 3D microscopy volumes into individual cell crops and filters them based on signal and size criteria.
- Detects multiscale 3D keypoints and describes local structure using a custom 3D histogram-of-oriented-gradients implementation.
- Learns a sparse visual dictionary and converts each image into a normalized visual-word frequency vector for downstream analysis.
- Uses logistic regression to classify conditions in an interpretable way, including word weights and voxel-level attention maps.
- Provides additional downstream analyses such as Haralick texture quantification, cluster purity summaries, and PCA/UMAP embeddings.
- Generates Streamlit apps for reviewing segmentation results, embeddings, clusters, image patches, and model outputs.

## Pipeline structure

### Step 1: Segmentation and crop preparation
`Step1SegmentBoVW3D.py` runs the segmentation workflow, refines cell crops, generates metadata, and prepares a Streamlit-based QC interface for reviewing results. In the Streamlit app the user can manually remove cell crops from the dataset [due to poor segmentation, poor signal, etc].

### Step 2: Feature extraction and classification
`Step2ClassifyCrops.py` runs the BoVW classification workflow, including codebook generation, feature normalization, interpretable classification, and downstream visualization/analysis.

## Repository contents

- `Step1SegmentBoVW3D.py` – entry point for segmentation and crop generation
- `step1GeneralHelperFcts.py` – segmentation settings, metadata, filtering, and run setup
- `step1SegmentationHelperFcts.py` – Cellpose-based 3D segmentation and crop extraction
- `step1AnalysisHelperFcts.py` – crop-level filtering and metadata summaries
- `step1streamlitHelperFcts.py` – segmentation review app generation
- `Step2ClassifyCrops.py` – entry point for BoVW classification
- `HOG3D_Keypoints.py` – custom 3D HOG descriptor extraction around detected keypoints
- `step2GeneralHelperFcts.py` – classification settings, orchestration, codebook generation, and run management
- `step2ClassificationHelperFcts.py` – keypoint detection, sparse coding, frequency-vector generation
- `step2AnalysisHelperFcts.py` – classification, attention maps, Haralick analysis, PCA/UMAP, and clustering summaries
- `step2streamlitHelperFcts.py` – interactive classification review app generation

## Notes

This README is intentionally high level. Detailed setup, parameters, and usage instructions are documented separately.


## Quick Start

After setting up the environment (and putting your tif files into a folder) run:
<br>python Step1_Segmentation --imagepath my/image/path/here
<br>For classification, similarily run:
<br>python Step2_Classification --imagepath my/segmentation/results
