% % addpath ../common/
addpath C:\PHD\Fault Honeywell\Data\AE andrew ng\sparseae_exercise\Anomay Detection SAE

%% STEP 0: Load data from the database
%  Instructions
%  ------------
% 
%  main Code for SAE Anommaly Detection Industrial Data
trainData = sampleTrainDATA;
trainData=trainData';
trainLabels=[2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1]';
%%======================================================================
%% STEP 1: Provide Parameters


inputSize =size(trainData,1);
numClasses = 2;
hiddenSizeL1 = 200;    % Layer 1 Hidden Size
hiddenSizeL2 = 200;    % Layer 2 Hidden Size
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-3;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term       


%%======================================================================
%% STEP 2: Train the first sparse autoencoder

%  Randomly initialize the parameters
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);

%% ---------------------- -- ---------------------------------

options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.MaxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.Display = 'iter';
options.GradObj = 'on';

[sae1OptTheta, cost] = fminlbfgs( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSizeL1, ...
                                   lambda, sparsityParam, ...
                                   beta, trainData), ...
                              sae1Theta, options);


 % save('layer1.mat',  'sae1OptTheta');

 % load('layer1.mat');

% -------------------------------------------------------------------------



%%======================================================================
%% STEP 2: Train the second sparse autoencoder

[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);

%  Randomly initialize the parameters
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);

%% ---------------------- ---  ---------------------------------

 [sae2OptTheta, cost] = fminlbfgs( @(p) sparseAutoencoderCost(p, ...
                                   hiddenSizeL1, hiddenSizeL2, ...
                                   lambda, sparsityParam, ...
                                   beta, sae1Features), ...
                              sae2Theta, options);

 % save('layer2.mat',  'sae2OptTheta');
 % load('layer2.mat')'

% -------------------------------------------------------------------------


%%======================================================================
%% STEP 3: Train the softmax classifier


[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);

%  Randomly initialize the parameters
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);


%% ---------------------- ----  ---------------------------------


options.MaxIter = 100;
softmaxModel = softmaxTrain(hiddenSizeL2, numClasses, lambda, ...
                            sae2Features, trainLabels, options);

saeSoftmaxOptTheta = softmaxModel.optTheta(:);

% save('layer3.mat', 'saeSoftmaxOptTheta');
% load('layer3.mat');

% -------------------------------------------------------------------------



%%======================================================================
%% STEP 5: Finetune softmax model

% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.

% Initialize the stack using the parameters learned
stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

%% ---------------------- ---------  ---------------------------------
%
%

 [stackedAEOptTheta, cost] = fminlbfgs( @(p) stackedAECost(p, ...
                                   inputSize, hiddenSizeL2, ...
                                   numClasses, netconfig, ...
                                   lambda, trainData, trainLabels), ...
                              stackedAETheta, options);

% save('layer4.mat', 'stackedAEOptTheta');


% -------------------------------------------------------------------------


%%======================================================================
%% STEP 6: Test 

testData = sampleTestDATA();
testData = testData';
testLabels = [2 2 2 2 2 1 1]';

[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);


