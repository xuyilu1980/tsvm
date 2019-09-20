function totalEEGSignals = read_BCI_III_DSIVa(workpath)
%input:
% workpath: the root directory where you put the code;
% for example: read_BCI_III_DSIVa('C:\Program Files\MATLAB')

% For each subject on BCI competition III dataset IVa, we sequentially put the training set and the
% testing set into a total set. Then we randomly selected some samples as
% the training set and the other samples as the testing set from the total
% set over ten repetitions.

filenames = [strcat(workpath,'\tsvm\EEGData\OriginalData\BCI_III_DSIVa\data_set_IVa_aa.mat');
             strcat(workpath,'\tsvm\EEGData\OriginalData\BCI_III_DSIVa\data_set_IVa_al.mat');
             strcat(workpath,'\tsvm\EEGData\OriginalData\BCI_III_DSIVa\data_set_IVa_av.mat');
             strcat(workpath,'\tsvm\EEGData\OriginalData\BCI_III_DSIVa\data_set_IVa_aw.mat');
             strcat(workpath,'\tsvm\EEGData\OriginalData\BCI_III_DSIVa\data_set_IVa_ay.mat')];
         
trueLabelsFiles = [strcat(workpath,'\tsvm\EEGData\OriginalData\BCI_III_DSIVa\true_labels_aa.mat');
                  strcat(workpath,'\tsvm\EEGData\OriginalData\BCI_III_DSIVa\true_labels_al.mat');
                  strcat(workpath,'\tsvm\EEGData\OriginalData\BCI_III_DSIVa\true_labels_av.mat');
                  strcat(workpath,'\tsvm\EEGData\OriginalData\BCI_III_DSIVa\true_labels_aw.mat');
                  strcat(workpath,'\tsvm\EEGData\OriginalData\BCI_III_DSIVa\true_labels_ay.mat')];

%The struct "subject" in the file of "indexFile" contains the indexes of
%the samples with different classes. 
indexFile = strcat(workpath,'\tsvm\EEGdata\OriginalData\BCI_III_DSIVa\random_left_right_index.mat'); % random indexes 
load(indexFile);

nbSubjects = 5;
trainingEEGSignals = cell(nbSubjects,1);
testingEEGSignals = cell(nbSubjects,1);
totalEEGSignals = cell(nbSubjects,1);

%some constants
fs = 100; %sampling rate
startEpoch = 0.5; %an epoch starts 0.5s after the cue
endEpoch = 2.5;%an epoch ends 2.5s after the cue
nbSamplesPerTrial = ceil((endEpoch - startEpoch) * fs) + 1;

for s=1:nbSubjects    
    
    disp(['Reading data from subject ' num2str(s)]);
    
    %reading the data from this subject
    disp('reading files...');
    load(filenames(s,:));
    load(trueLabelsFiles(s,:));
    disp('...done!');
    
    %conversion to uV values
    cnt= 0.1*double(cnt);
    
    passBand.low = 8;
    passBand.high = 30;
    
    disp(['will band-pass filter all signals in ' num2str(passBand.low) '-' num2str(passBand.high) 'Hz']); 
    order = 5; 
    lowFreq = passBand.low * (2/fs);
    highFreq = passBand.high * (2/fs);
    [B A] = butter(order, [lowFreq highFreq]);
    cnt = filter(B,A,cnt);    
    
    nbChannels = size(cnt,2);
    
    %identifying the training and testing trials
    labels = mrk.y;
    cues = mrk.pos;
    trueLabels = true_y;
    trainingIndexes = find(~isnan(labels));
    testingIndexes = find(isnan(labels));
    totalIndexes = 1:length(true_y);
    
    nbTrainTrials = length(trainingIndexes);
    disp(['nbTrainTrials = ' num2str(nbTrainTrials)]);
    nbTestTrials = length(testingIndexes);
    disp(['nbTestTrials =  ' num2str(nbTestTrials)]);
    nbTotalTrials = length(true_y);
    disp(['nbTotalTrails = ' num2str(nbTotalTrials)]);
    

    %initializing structures
    disp('initializing structures...');

    
    trainingEEGSignals{s}.x = zeros(nbSamplesPerTrial, nbChannels, nbTrainTrials);
    trainingEEGSignals{s}.y = -2*labels(trainingIndexes)+3;
    trainingEEGSignals{s}.l = -2*labels(trainingIndexes)+3;
    trainingEEGSignals{s}.s = fs;
    trainingEEGSignals{s}.c = nfo.clab; 
    testingEEGSignals{s}.x = zeros(nbSamplesPerTrial, nbChannels, nbTestTrials);
    testingEEGSignals{s}.y = -2*trueLabels(testingIndexes)+3;
    testingEEGSignals{s}.l = zeros(1,nbTestTrials);
    testingEEGSignals{s}.s = fs;
    testingEEGSignals{s}.c = nfo.clab;
    
    totalEEGSignals{s}.y =  [trainingEEGSignals{s}.y,testingEEGSignals{s}.y];
    totalEEGSignals{s}.l = [trainingEEGSignals{s}.l,testingEEGSignals{s}.l];
    totalEEGSignals{s}.s = fs;
    totalEEGSignals{s}.c = nfo.clab;
    

    disp('...done!');
    
    %assigning data to the corresponding structure
    disp('assigning data to the corresponding structure...');
    
    %training set    
    for trial=1:nbTrainTrials    
        %getting the cue
        cueIndex = cues(trainingIndexes(trial));
        %getting the data
        epoch = cnt((cueIndex + round(startEpoch*fs)):(cueIndex + round(endEpoch*fs)),:);
        %disp(size(epoch));
        %disp(size(trainingEEGSignals{s}.x(:,:,trial)));
        trainingEEGSignals{s}.x(:,:,trial) = epoch;
        totalEEGSignals{s}.x(:,:,trial) = epoch;
    end
    
    %testing set
    for trial=1:nbTestTrials    
        %getting the cue
        cueIndex = cues(testingIndexes(trial));
        %getting the data
        testingEEGSignals{s}.x(:,:,trial) = cnt((cueIndex + round(startEpoch*fs)):(cueIndex + round(endEpoch*fs)),:);
        totalEEGSignals{s}.x(:,:,trial+nbTrainTrials) = testingEEGSignals{s}.x(:,:,trial);
    end
    
    %In order to randomly select the training set and 
    %the test set over ten repetitions, For subject s,the random index of
    %right hand and the random index of left hand are obtained in the
    %following way
    
    %left = find(totalEEGSignals{s}.y == 1);
    %right = find(totalEEGSignals{s}.y == -1);
    %for iter = 1:10 
%         index{s}.random_left_index(iter,:) = left(randperm(length(left)));
%         index{s}.random_right_index(iter,:) = right(randperm(length(right)));
%     end;

    totalEEGSignals{s}.left_index = index{s}.random_left_index;
    totalEEGSignals{s}.right_index = index{s}.random_right_index;
    disp('...done!');
end

save([workpath,'\tsvm\EEGData\BCI_III_DSIVa\totalEEGSignals_BCI3.mat'],'totalEEGSignals');    
    
    
    
    
    