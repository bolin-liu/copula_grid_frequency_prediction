# copula_based_grid_frequency_prediction

Code for the paper: **Copula-based Probabilistic Prediction of Grid Frequency Dynamics** (submitted).

## Overview

This repository contains code to perform copula-based probabilistic modeling of grid frequency dynamics. It introduces a copula-based error correction mechanism to improve predictive performance.

### Main Files

- **'copula_correction.py'** includes key components used in the experiments from the paper:
  - Feature-based clustering and visualization  
  - Copula-based error correction (copula estimation)  
  - Computation of confidence intervals for the hourly average frequency deviation  
  - Copula-based prediction and evaluation on the test data using energy scores and average CRPS  

- **'evaluation.py'** provides code to evaluate the predictive performance of different models using energy scores and average CRPS.

- copula_correction.py needs the script utilities.py from [2]. 

## Data

The experimental setup requires training and test data as used in [1] and [2]. To run the code:

1. Follow the instructions in [1] to obtain the preprocessed training and test datasets. Save them with the following filenames:
   - 'frequency_train.pkl' 
   - 'frequency_test.pkl' 
   - 'day_ahead_features_train.pkl'
   - 'day_ahead_features_test.pkl'

2. Use the code in [1] to generate predictions from PIML models and save them in the 'data' folder.

3. Use the code and data from [2] to generate predictions from the purely data-driven Gaussian-based models and k-NN models, and save them in the 'data' folder.

4. Use the scaler and trained models from [2], and save them in the 'trained_models' folder.

## References

[1]: https://github.com/johkruse/PIML-for-grid-frequency-modelling  
[2]: https://github.com/bolin-liu/sequence-model-and-gaussian-process-for-frequency-prediction  
