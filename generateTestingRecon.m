%% Generate the Reconstruction Error Images for the Testing Data

%This file takes the pretrained dictionary from preprocessData.m and runs
%OMP with it on the testing portion of data. The reconstruction
%error/difference images are produced and features are extracted from them.
%The data is then stored back into the original format. 
%% Load in Paths and Data
clc;
clear all;
clear figures;
rng(0, 'twister');


datadir = 'D:\Users\Conno Z From The 303\Downloads\__THESIS&RESEARCHHDD\__insect-lidar-supervised-classification-main\data\insect-lidar\MLSP-2021';

addpath 'C:\Users\Conno Z From The 303\Downloads\Thesis&Research\ompbox10'
addpath 'C:\Users\Conno Z From The 303\Downloads\Thesis&Research\ksvdbox13'
addpath 'C:\Users\Conno Z From The 303\Downloads\Thesis&Research\PreviousWork\insect-lidar-supervised-classification-main\insect-lidar-supervised-classification-main'

if isempty(gcp('nocreate'))
    parpool();
end

 % Load data
 load([datadir filesep 'testing' filesep 'testingData.mat']);

%% Format data 

trainingMatrix = [];
trainingLabels = []; 
count = 1;

for i = 1:20                                        
    for j = 1:length(testingData{i,1})
        
        if(isempty(testingData{i,1}{j,1}))
            continue
        end
            trainingMatrix(:,1+178*(count-1):178*count) = testingData{i,1}{j,1}'; %Turn cell data into matrix
            trainingLabels(:,1+178*(count-1):178*count) = testingLabels{i,1}{j,1}';

        count = count+1;
    end
end


%% Generate Reconstruction Difference Images, D = 512
load('D512.mat')

%generateDifferenceImages(data, numSparse, Dtrain)
%data should be of type double
%numSparse is the number of sparse coefficients to be used
%Dtrain is the previously trained dictionary
%In this portion of code Dtrain = D512

differenceImgTestingD512 = generateDifferenceImages(trainingMatrix,4,D); %data should be of type double


figure; showdict(D,[32 32],10,10); title("Learned Dictionary 1024x512, 10 entries"); %Look at atoms in dictionary, not required

figure; imshow(differenceImgTestingD512(:,1+178*9:10*178)'); title("Reconstruction Error Image #9 D512")

%% Generate Reconstruction Difference Images, D = 1024
load('D1024.mat')

differenceImgTestingD1024 = generateDifferenceImages(trainingMatrix,4,D);


figure; showdict(D,[32 32],10,10); title("Learned Dictionary 1024x1024, 10 entries");

figure; imshow(differenceImgTestingD1024(:,1+178*9:10*178)'); title("Reconstruction Error Image #9 D1024")

%% Generate Reconstruction Difference Images, D = 2048
load('D2048.mat')

differenceImgTestingD2048 = generateDifferenceImages(trainingMatrix,4,D);


figure; showdict(D,[32 32],10,10); title("Learned Dictionary 1024x2048, 10 entries");

figure; imshow(differenceImgTestingD2048(:,1+178*9:10*178)'); title("Reconstruction Error Image #9 D2048")

%% Save reconstruction error/difference images and the matrix

%save("trainingNonCell","trainingMatrix","-v7.3")
save("testingDataD512","differenceImgTestingD512","-v7.3")
save("testingDataD1024","differenceImgTestingD1024","-v7.3")
save("testingDataD2048","differenceImgTestingD2048","-v7.3")

%% Feature Extraction

% Feature Extract D512 Data
testingFeaturesD512 = extractFeatures(differenceImgTestingD512');

% Feature Extract D512 Data
testingFeaturesD1024 = extractFeatures(differenceImgTestingD1024');

% Feature Extract D512 Data
testingFeaturesD2048 = extractFeatures(differenceImgTestingD2048');

%% Save Feauture Extraction

save("testingFeaturesD512","testingFeaturesD512","-v7.3")
save("testingFeaturesD1024","testingFeaturesD1024","-v7.3")
save("testingFeaturesD2048","testingFeaturesD2048","-v7.3")

%% Slap those boys back into Cells, D512

%The cell versions of the testing data,labels, and features will be required for testing the classifiers

count = 1;

funcount = 0;

for i = 1:20                                        
    for j = 1:length(testingData{i,1})
        
        if(isempty(testingData{i,1}{j,1}))
            funcount = funcount +1;
            continue
        end
           testingDataD512Cell{i,1}{j,1} = differenceImgTestingD512(:,1+178*(count-1):178*count)'; %make signals back into 178x1024 image, rather than 1024x178
           testingLabelsD512Cell{i,1}{j,1} = testingLabels{i,1}{j,1}; %Take og labels
           testingFeaturesD512Cell{i,1}{j,1} = testingFeaturesD512(1+178*(count-1):178*count,:); %Features are stacked differently, ends up as 178x30 like training example data
        count = count+1;
    end
end

save("testingDataD512Cell","testingDataD512Cell","-v7.3")
save("testingLabelsD512Cell","testingLabelsD512Cell","-v7.3")
save("testingFeaturesD512Cell","testingFeaturesD512Cell","-v7.3")

%% Slap those boys back into Cells, D1024

count = 1;

funcount = 0;

for i = 1:20                                        
    for j = 1:length(testingData{i,1})
        
        if(isempty(testingData{i,1}{j,1}))
            funcount = funcount +1;
            continue
        end
           testingDataD1024Cell{i,1}{j,1} = differenceImgTestingD1024(:,1+178*(count-1):178*count)'; %make signals back into 178x1024 image, rather than 1024x178
           testingLabelsD1024Cell{i,1}{j,1} = testingLabels{i,1}{j,1}; %Take og labels
           testingFeaturesD1024Cell{i,1}{j,1} = testingFeaturesD1024(1+178*(count-1):178*count,:); %Features are stacked differently, ends up as 178x30 like training example data
        count = count+1;
    end
end

save("testingDataD1024Cell","testingDataD1024Cell","-v7.3")
save("testingLabelsD1024Cell","testingLabelsD1024Cell","-v7.3")
save("testingFeaturesD1024Cell","testingFeaturesD1024Cell","-v7.3")

%% Slap those boys back into Cells, D2048

count = 1;

funcount = 0;

for i = 1:20                                        
    for j = 1:length(testingData{i,1})
        
        if(isempty(testingData{i,1}{j,1}))
            funcount = funcount +1;
            continue
        end
           testingDataD2048Cell{i,1}{j,1} = differenceImgTestingD2048(:,1+178*(count-1):178*count)'; %make signals back into 178x1024 image, rather than 1024x178
           testingLabelsD2048Cell{i,1}{j,1} = testingLabels{i,1}{j,1}; %Take og labels
           testingFeaturesD2048Cell{i,1}{j,1} = testingFeaturesD2048(1+178*(count-1):178*count,:); %Features are stacked differently, ends up as 178x30 like training example data
        count = count+1;
    end
end

save("testingDataD2048Cell","testingDataD2048Cell","-v7.3")
save("testingLabelsD2048Cell","testingLabelsD2048Cell","-v7.3")
save("testingFeaturesD2048Cell","testingFeaturesD2048Cell","-v7.3")