function [output,insectrows]= prepinsectimages(trainingLabels)
%This function checks all labels and returns back how many rows contain
%insects and the images containing insects. The images that contains
%insects are stored as a sparse matrix

insectcelltemp = [];
insectimagetemp = [];
insectrows= 0;
for p = 1:80
    for j = 1:length(trainingLabels{p,1})
        for c = 1:178
            if(trainingLabels{p,1}{j,1}(c) == 1) 
                insectrows = insectrows+1;
                insectcelltemp = [insectcelltemp,p];
                insectimagetemp(j,p) = j; %Store image number 
                %in this data matrix the columns are equal to the current cell
            end

        end 
    end
end
%This eliminates the data matrix that I made
%The matrix contains mostly zeros, except for when an image contains an
%insect
insectcellnum = unique(insectcelltemp); %what cells contain insects
insectimage = sparse(insectimagetemp); 

output = insectimage;
end