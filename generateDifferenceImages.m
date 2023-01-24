function [newData] = generateDifferenceImages(data, numSparse, Dtrain)
% Generates the reconstruction error (difference) images. These will be used to retrain
% the models previously used. This is the sparse coding
% stage of ksvd, since the dictionary is known (and trained), and not being updated.
    
    D = Dtrain; %Pretrained Dictionary
    G = D'*D; %Gram matrix 
    thresh = numSparse; %number of sparse coefficients to use
    
    Gamma = omp(D, double(data), G, thresh); %Run OMP, data must be of type double
    
    dataTilde = D*Gamma; %Make reconstruction of all data
    
    newData = data - dataTilde; %Reconstruction Error: original data - new data
    
end