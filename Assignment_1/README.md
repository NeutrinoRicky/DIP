# Implementation of Image Geometric Transformation

This repository contains my implementation of Assignment 01 for Digital Image Processing (DIP).

<img src="assets/teaser.png" alt="teaser" width="800">

## Requirements

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Running

Run basic global transformation (scale / rotation / translation / flip):

```bash
python run_global_transform.py
```

Run point-guided image deformation:

```bash
python run_point_transform.py
```

## Results

### Basic Transformation

<img src="assets/global_demp.gif" alt="global demo" width="800">

### Point Guided Deformation

<img src="assets/point_demo.gif" alt="point demo" width="800">

## Acknowledgement

Thanks for the algorithm inspiration from:
[Image Deformation Using Moving Least Squares](https://people.engr.tamu.edu/schaefer/research/mls.pdf).
