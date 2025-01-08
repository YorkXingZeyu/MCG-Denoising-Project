# Project Documentation

## Overview

This project involves the use of multiple datasets and configurable parameters for tasks such as denoising ECG/MCG signals and generating confidence maps. Below are detailed instructions on how to set up and use the system.

---

## Datasets

The `data_loader.py` file located in the `data` folder defines four datasets:

- **MCG_019_result**: Represents averaged data over 19 repetitions.
- **MCG_101_result**: Represents averaged data over 101 repetitions.
- **ECG_Data_result**: Represents ECG signals with simulated noise.
- **MCG_self_result**: Represents MCG signals with simulated noise.

To use these datasets, update the dataset paths in `data_preparation.py`, located in the `Data_Preparation` folder.

---

## Configuration Parameters

All parameters are defined in the `config/base.yaml` file:

### Dataset and Output Configuration

- **output_folder**: Determines the dataset to use and the output directory.  
  Example: If `output_folder: MCG_101_result`, the system will:
  1. Use the `MCG_101_result` dataset.
  2. Save results in a folder named `MCG_101_result`.
  3. Generate a compressed folder for easier downloads.

### Confidence Map Parameters

- **sigma**: Controls the range of Gaussian smoothing.  
- **distance**: Determines the interval between peaks.  
- **confidence_weight**: Adjusts the weight of the embedded confidence maps.

### Conditional Network Flags

- **cnet_condition**: Controls whether conditions are added to `cnet` (always `False` for generating confidence maps).  
- **unet_condition**: Controls whether conditions are added to `unet` (maps noisy signals to clean signals).  

### Training and Evaluation Parameters

- **train_cnet**: Whether to train the `cnet` model.  
- **train_unet**: Whether to train the `unet` model.  
- **evaluate_model**: Whether to evaluate the model.  

Example Usage:
- To train from `cnet` to evaluation, set all three parameters to `True`.
- To debug `unet` after training `cnet`, set `train_cnet` to `False`.

---

## Code Structure

### Key Modules

- **`utils/confidence_maps.py`**: Contains the methods for calculating confidence maps.
- **`models`** folder:
  - `cnet`: Learns to generate confidence maps.
  - `unet`: Maps noisy signals to clean signals, optionally using the confidence maps as conditions.

---

## Getting Started

1. Configure the dataset path in `Data_Preparation/data_preparation.py`.
2. Set the desired parameters in `config/base.yaml`.
3. Run the project pipeline:
   - Train `cnet` and/or `unet` as required.
   - Evaluate the model if needed.
4. Review results in the `output_folder`, including the generated compressed files for download.

---

## Notes

- `cnet_condition` is always `False` since it is not required for confidence map generation.
- The choice to add conditions to `unet` depends on your experimental goals.

---

## Example Configuration (`base.yaml`)

```yaml
output_folder: MCG_101_result
sigma: 12
distance: 55
confidence_weight: 1
cnet_condition: False
unet_condition: False
train_cnet: False
train_unet: True
evaluate_model: True



