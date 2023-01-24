clear;
clc;
clear figures; 
rng(0, 'twister');


 datadir = 'D:\Users\Conno Z From The 303\Downloads\__THESIS&RESEARCHHDD\__insect-lidar-supervised-classification-main\data\insect-lidar\MLSP-2021';

if isempty(gcp('nocreate'))
    parpool();
end
% 
 % Load data
 load([datadir filesep 'training' filesep 'trainingData.mat']);

%This program seperates the original dataset into three different portions.
%These three portions are the insect images, 10% of the noninsect images,
%and 90% of noninsect images. This is required for the training of KSVD and
%to facilitate the removal of the 10% of noninsect that were used. The
%remaining 90% of noninsect data and the insect images will be used later. 

%% First Prep the insect and non insect images

%% INSECTS
%% Insect Dataset

% Get Bugs 
[insectImages, numrowsContainInsect] = prepinsectimages(trainingLabels); %(image num,cell num)

%InsectImages is a sparse matrix containing the coordinates for each insect
%containing image

[imagenum,cellnum, ~] = find(insectImages); %Find retruns the row, col, and val of the sparse matrix
%The row and val corresponds to the image number
%The col corresponds to the cell in the data matrix that the image can be
%found


%length of imagenum or cellnum corresponds to number of insects in data
%cellnum is the cell number in which an insect can be found
%imagenum is the image number stored within that cell

%An example: cellnum = [1, 1, 1, 2...] 
%            imagenum =[23,25,83,19,...]
%There are 3 images in the first cell of the trainingData matrix
%These images are in rows 23, 25, and 83

% Set up Data Matrix
insectImages = [];
insectLabels = [];

%save insect images as a matrix rather than cell array for future use
for jj = 1:length(imagenum)
    insectImages = [insectImages, double(trainingData{cellnum(jj),1}{imagenum(jj),1}')]; %store as double for later
    insectLabels = [insectLabels, trainingLabels{cellnum(jj),1}{imagenum(jj),1}']; %store matching labels
end

%save("insectImages.mat","insectImages")
%save("insectLabels.mat","insectLabels")

%% Non Insect Dataset

%extract all noninsect containing images
for i = 1:80
    for j = 1:length(trainingLabels{i,1})

        count = 0;
        for c = 1:178 
             if(trainingLabels{i,1}{j,1}(c) == 0) %check for insect in labels
                 count = count + 1; 
             end
        end

        if (count == 178) %if count reaches 178, then no rows contained an insect 
            NonInsectImages{i,1}{j,1} = trainingData{i,1}{j,1}'; %transpose for later training
            NonInsectLabels{i,1}{j,1} = trainingLabels{i,1}{j,1}';%save matching labels
        end
    end
end

%% See how many non insect images exist
nonInsectImageCount = 0; %Final Number = 8144

for p = 1:80
    nonInsectImageCount =   nonInsectImageCount+ size(NonInsectImages{p,1},1);
end
%% Alrighty we need 801 noninsect images to sacrifice to training, rng(123)
rng(123);

nonemptyimages = setdiff(1:96,imagenum); %if the image number is every used, indicates an insect in image,
% remove from possible rand choices
imagerand = randsample(nonemptyimages,10); %Grab 10 ~random images from each cell

howlong = 178*801; %Find out how many columns should be in final 10% of data, 142578

nonInsectTenPercent = [];
nonInsectTenPercentLabels = [];
moreimg = 0;

%save noninsect images as a matrix rather than cell array for future use
for i = 1:80
    if (i == 58) %row 58 in data matrix contains only 10 images, remove from process
        continue
    end
    for j = 1:10  
        if(isempty(NonInsectImages{i,1}{imagerand(j),1})) %statement should not activate, failsafe
            j = j+1; %grab the next image and skip 
            moreimg = moreimg + 1; %count how many extra images that are required
            nonInsectTenPercent = [nonInsectTenPercent, NonInsectImages{i,1}{imagerand(j),1}];
            nonInsectTenPercentLabels = [nonInsectTenPercentLabels, NonInsectLabels{i,1}{imagerand(j),1}];
            NonInsectImages{i,1}{imagerand(j),1} = [];
            NonInsectLabels{i,1}{imagerand(j),1} = [];
        else
       
            nonInsectTenPercent = [nonInsectTenPercent, NonInsectImages{i,1}{imagerand(j),1}];
            nonInsectTenPercentLabels = [nonInsectTenPercentLabels, NonInsectLabels{i,1}{imagerand(j),1}]; 
            NonInsectImages{i,1}{imagerand(j),1} = []; %remove image  from noninsect dataset
            NonInsectLabels{i,1}{imagerand(j),1} = []; %remove labels from noninsect dataset

        end
    end
end

for i = 1:11 %make up for the 10 lost images from row 58, grab the 10th image from every third data matrix
    nonInsectTenPercent = [nonInsectTenPercent, NonInsectImages{3*i,1}{10,1}];
    NonInsectImages{3*i,1}{10,1} = [];
end

%save("nonInsect10Percent.mat","nonInsectTenPercent")

%% 90% of Non Insect
clear trainingData %clear from memory
clear trainingLabels
clear trainingFeatures
clear nonInsectTenPercent



nonInsectNinetyPercent= single(ones(1024,1284448)); %Preallocate

count = 1;


for i = 1:80                                        
    for j = 1:length(NonInsectImages{i,1})
        
        if(isempty(NonInsectImages{i,1}{j,1})) %Empty cells correspond to images used for 10%
            continue
        end
            nonInsectNinetyPercent(:,1+178*(count-1):178*count) = NonInsectImages{i,1}{j,1};
            nonInsectNinetyPercentLabels(:,1+178*(count-1):178*count) = NonInsectLabels{i,1}{j,1};

        count = count+1;
    end
end

%save("nonInsect90Percent.mat","nonInsectNinetyPercent")
%save("nonInsect90PercentLabels.mat","nonInsectNinetyPercentLabels")

%You can save the noninsect image cell with empty arrays, I chose not to

%% Get rid of empty cells, RUN MANUALLY FOR IT TO ACTUALLY WORK
for i = 1:80
     NonInsectImages{i,1} = NonInsectImages{i,1}(~cellfun('isempty',NonInsectImages{i,1})); %remove empty cell arrays
     NonInsectLabels{i,1} = NonInsectLabels{i,1}(~cellfun('isempty',NonInsectLabels{i,1})); %remove empty cell arrays
end

%Run the noninsectimage counter again to verify the remaining number of
%images, in this case it is 7216. 8017-801 = 7216, so the program
%correctly removed the 801 images from the new dataset.