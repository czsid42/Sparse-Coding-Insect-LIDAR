clc;
clear all;
clear figures;
rng(0, 'twister');

% This file will train the user defined dictionary by using K-Singular Value Decomposition. This
% is done using only 10% of non-insect images.After training the dictionary the reconstructions 
% of the data is produced by using Orthogonal Matching Pursuit. 
% The difference between the original image and the reconstructed image is taken to generate the reconstruction
% error/difference image. This new image is fed to the feature extraction
% algorithm to finish off the preprocessing of data. Due to K-SVD and OMP
% only being able to use type double data the newly generated images and
% their features must be stored back into the original data structure. The
% formattingData.m took the data from the original cell arrays and stacked it to form a single
% matrix of images, the last portion of code reverses this process. 

datadir = '\..\data';
%
addpath '\..\ompbox10'
addpath '\..\ksvdbox13'
addpath '\..\insect-lidar-supervised-classification-main\insect-lidar-supervised-classification-main'
%
if isempty(gcp('nocreate'))
    parpool();
end
%% Load in 10% of data to train ksvd
load("nonInsect10Percent.mat")
%% TRAIN KSVD and choose Dictionary Size

%trainKSVD(data, numSparse, numAtoms, numIter,memUsage)

%data is the 10% of noninsect data

%numSparse is the number of sparse coefficients allowed

%numAtoms determine the size of the dictionary, D = 1024xnumAtoms

%numIter is the number of iterations used to train KSVD, in this work past
%200 iterations caused minimal gains

%memUsage determines how much memory KSVD uses (if interested, the KSVD
%function used by this work has more info)

[D, err] = trainKSVD(double(nonInsectTenPercent), 4, 2048, 200, 'high'); %data needs to be double type

%Save trained dictionary and error convergence
% save("D2048.mat","D")
% save("Err2048.mat","err")
%% Plots
figure; plot(err); title('K-SVD error convergence');
xlabel('Iteration'); ylabel('RMSE');

clear nonInsectTenPercent %Remove the 10% of noninsect data used to train the dictionary

%% Reconstruction of NonInsect Data
load("nonInsect90Percent.mat")

%generateDifferenceImages(data, numSparse, Dtrain)
%This function generates the reconstruction error/difference images

%data should be of type double, in this portion of work data is the
%remaining 90% of non-insect images

%numSparse should be the same as the number of sparse coeffs used to train
%K-SVD, limits the number of dictionary atoms that OMP can use to
%reconstruct data

%Dtrain is the trained dictionary that K-SVD generated

numSparse = 4;

reconNonInsect = generateDifferenceImages(double(nonInsectNinetyPercent),numSparse,D)

errorNonInsect = norm(reconNonInsect,'fro')
% 
% save("nonInsectReconstructed_D2048.mat","reconNonInsect","-v7.3")
% save("nonInsectReconstructed_froError2048.mat","errorNonInsect")

clear nonInsectNinetyPercentfirsthalf %clear for mem
clear tildeNonInsect

%% Reconstruction of Insect Portion of Data
load("insectImages.mat")

reconInsect = generateDifferenceImages(double(insectImages),numSparse,D)
errorInsect = norm(reconInsect,'fro')

% save("insectReconstructed_D2048.mat","reconInsect","-v7.3")
% save("insectReconstructed_froError2048.mat","errorInsect")

%% Run feature extraction on recon data

%load("nonInsectReconstructed_D2048.mat") 

%I ran the previous code for
%each dictionary and saved results. I then needed to load the specific
%reconstruction error images for each dictionary to then extract features
%related to the specific dictionary

nonInsectNinetyPercent_Features = extractFeatures(reconNonInsect');

%save("reconNonInsectFeaturesD2048.mat","nonInsectNinetyPercent_Features","-v7.3")

%load("insectReconstructed_D2048.mat")

reconInsect_Features = extractFeatures(reconInsect');
%save("reconInsect_Features2048.mat","reconInsect_Features")

%% SETUP CELLS FOR TRAINING, Put Reconstructed data back into appropriate cells 
load("nonInsectImages.mat")
load("nonInsectLabels.mat")

count = 1;

funcount = 0;

for i = 1:80                                        
    for j = 1:length(NonInsectImages{i,1})
        
        if(isempty(NonInsectImages{i,1}{j,1}))
            funcount = funcount +1;
            continue
        end
           reconCell{i,1}{j,1} = reconNonInsect(:,1+178*(count-1):178*count)'; %make signals back into 178x1024 image, rather than 1024x178
           reconLabels{i,1}{j,1} = NonInsectLabels{i,1}{j,1}'; %Transpose labels so that it fits OG data
           reconFeatures{i,1}{j,1} = nonInsectNinetyPercent_Features(1+178*(count-1):178*count,:); %Features are stacked differently, ends up as 178x30 like training example data
        count = count+1;
    end
end

%% Test Add in insect images in the original spots

%The main reason I am doing this is due to the partitioning and the
%scanlabels variable. I want to ensure every variable is in the original
%place

 load([datadir filesep 'training' filesep 'trainingData.mat']);
 load('insectLabels.mat')

[insectImages, numrowsContainInsect] = prepinsectimages(trainingLabels); %(image num,cell num)

[imagenum,cellnum, ~] = find(insectImages);

[C,ia,ic] = unique(cellnum); %find unique cells and returns index vectors ia and ic 
AA = accumarray(ic,1); %Count the number of times each image in C appears in AA
count = 1;
for i = 1:length(C) %We have 56 unique cells                                       
    for j = 1:AA(i) %How many images are in any given cell
        
            reconCell{C(i),1}{imagenum(count),1} = reconInsect(:,1+178*(count-1):178*count)';
            reconLabels{C(i),1}{imagenum(count),1} = logical(insectLabels(:,1+178*(count-1):178*count))';
            reconFeatures{C(i),1}{imagenum(count),1} = reconInsect_Features(1+178*(count-1):178*count,:);
        count = count+1;
    end
end

%% Remove the empty cells left behind from the 10% training, RUN MANUALLY

for i = 1:80
     reconCell{i,1} = reconCell{i,1}(~cellfun('isempty',reconCell{i,1})); %remove empty cell arrays
     reconLabels{i,1} = reconLabels{i,1}(~cellfun('isempty',reconLabels{i,1})); %remove empty cell arrays
     reconFeatures{i,1} = reconFeatures{i,1}(~cellfun('isempty',reconFeatures{i,1})); %remove empty cell arrays
end

%% Check to see if labels and data are the same
labelcount = 0;
datacount = 0;
featureCount = 0;
 for p = 1:80
    datacount = datacount + size(reconCell{p,1},1);
    labelcount = labelcount + size(reconLabels{p,1},1);
    featureCount = featureCount + size(reconFeatures{p,1},1);
 end

  datacount 
  labelcount 
  featureCount
%% SAVE DATA

% save("reconTrainingDataD2048.mat","reconCell","-v7.3")
% save("reconTrainingLabelsD2048.mat","reconLabels")
% save("reconTrainingFeaturesD2048.mat","reconFeatures","-v7.3")

