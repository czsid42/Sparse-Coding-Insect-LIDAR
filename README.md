

# Insect Lidar Supervised Classification with Sparse Coding as a preprocessing technique
Code for detecting insects in lidar data with the aid of Sparse Coding.

This repository contains the original code used to implement sparse coding as a preprocessing technique in lidar data. 
This code was used to create the results in our paper [*USING SPARSE CODING AS A PREPROCESSING TECHNIQUE FOR INSECT DETECTION IN PULSED LIDAR DATA*](dont have doi yet), which was submitted to the 2023 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP). This was also used for the work done on my thesis.
[This repository](https://github.com/BMW-lab-MSU/insect-lidar-supervised-classification) contains the original code used to create the results in our paper [*Detection of Insects in Class-imbalanced Lidar Field Measurements*](https://doi.org/10.1109/MLSP52302.2021.9596143), which was published in and presented at the 2021 IEEE Machine Learning for Signal Processing (MLSP) conference. 
The dataset used in this paper is archived at [Zenodo](https://zenodo.org/record/5504411) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5504411.svg)](https://doi.org/10.5281/zenodo.5504411)

## How to run the experiments

### Download and install KSVD and OMP
1. The toolboxes used by this project are here: https://www.cs.technion.ac.il/~ronrubin/software.html
2. Download OMP-Box v10  and follow the instructions for install
3. Download KSVD-Box v13 and follow the instructions for install

### Create training and testing data
1. Combine the individual data and label files into a more usable format: `combineScans.m`
2. Split the data into training and test sets: `trainTestSplit.m`

### Train KSVD, generate the reconstruction error/difference images, and extract features
1. Split the data into 3 categories: `formattingData.m`
2. Train KSVD with a user defined dictionary size and generate the reconstruction error/difference images and their features: `preprocessData.m`
3. Generate the reconstruction error/difference images for testing: `generateTestingRecon.m`


### Train and test the classifiers using the modified data, labels, and features. 
This work expands upon the previous work found in [this repository.](https://github.com/BMW-lab-MSU/insect-lidar-supervised-classification)
You will need to replace the trainingData, trainingFeatures, and trainingLabels with the modified data in each file.
1. Tune the under- and oversampling ratios: `tuneSampling{AdaBoost, RUSBoost, Net}.m`
2. The the model hyperparameters: `tuneHyperparams{AdaBoost, RUSBoost, Net}.m`
3. Train the final models: `train{AdaBoost, RUSBoost, Net}.m`
4. Test the classifiers: `testClassifiers.m`

### Results
The testing results are saved in `<data directory>/testing/results.mat`. 

To analyze the cross validation results, collect them by running `collectCrossValResults.m`. The results will be in `<data directory>/training/cvResults.m`. 
