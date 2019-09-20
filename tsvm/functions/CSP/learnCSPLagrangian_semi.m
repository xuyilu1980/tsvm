function CSPMatrix = learnCSPLagrangian_semi(EEGSignals,index)
%
% <The RCSP (Regularized Common Spatial Pattern) Toolbox.>
%     Copyright (C) 2010  Fabien LOTTE
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%this function learn the CSP (Common Spatial Patterns) filters to
%discriminate two mental states in EEG signals.
%The optimization problem is solved here using the lagrangian approach
%(which is equivalent to joint diagonalization approach)
%
%Input:
%EEGSignals: the training EEG signals, composed of 2 classes. These signals
%are a structure such that:
%   EEGSignals.x: the EEG signals as a [Ns * Nc * Nt] Matrix where
%       Ns: number of EEG samples per trial
%       Nc: number of channels (EEG electrodes)
%       nT: number of trials
%   EEGSignals.y: a [1 * Nt] vector containing the class labels for each trial
%   EEGSignals.s: the sampling frequency (in Hz)
%
%Output:
%CSPMatrix: the learnt CSP filters (a [Nc*Nc] matrix with the filters as rows)
%
%by Fabien LOTTE (fprlotte@i2r.a-star.edu.sg)
%created: 02/03/2010
%last revised: 02/03/2010
%
%See also: extractCSPFeatures

%check and initializations
nbChannels = size(EEGSignals.x,2);
nbTrials = length(index);
classLabels = unique(EEGSignals.y);
nbClasses = length(classLabels);
if nbClasses ~= 2
    disp('ERROR! CSP can only be used for two classes');
    return;
end
covMatrices = cell(nbClasses,1); %the covariance matrices for each class

%computing the normalized covariance matrices for each trial
trialCov = zeros(nbChannels,nbChannels,nbTrials);
for t = 1:length(index)
    E = EEGSignals.x(:,:,index(t))';
    EE = E * E';
    trialCov(:,:,t) = EE ./ trace(EE);
end
clear E;
clear EE;

%computing the covariance matrix for each class
for c=1:nbClasses      
    covMatrices{c} = mean(trialCov(:,:,EEGSignals.y(index) == classLabels(c)),3); 
    covMatrices{c} = covMatrices{c} ./ sum(EEGSignals.y(index) == classLabels(c));
end

%computing the matrix M to be decomposed
M = inv(covMatrices{2}) * covMatrices{1};

%eigen value decomposition of matrix M
[U D] = eig(M);
eigenvalues = diag(D);
[eigenvalues egIndex] = sort(eigenvalues, 'descend');
U = U(:,egIndex);
CSPMatrix = U';