function data = sampleTestDATA()
% sampleIMAGES
% Returns 10000 patches for training

% % load IMAGES;    % load images from disk 
% % 
% % patchsize = 8;  % we'll use 8x8 patches 
% % numpatches = 10000;

% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns. 
% % data1 = zeros(44,237);

%% ------------------------------------------------
 

% % counter = 1;
% % ranimg = ceil(rand(1, numpatches) * 10);
% % ranpix = ceil(rand(2, numpatches) * (512 - patchsize));
% % ranpixm = ranpix + patchsize - 1;
% % while(counter <= numpatches)
% % whichimg = ranimg(1, counter);
% % whichpix = ranpix(:, counter);
% % whichpixm = ranpixm(:, counter);
% % patch = IMAGES(whichpix(1):whichpixm(1), whichpix(2):whichpixm(2), whichimg);
% % repatch = reshape(patch, patchsize * patchsize, 1);
% % patches(:, counter) = repatch;
% % counter = counter + 1;
% % end

IMPT(:,:,2) = xlsread('Series2');
IMPT(:,:,3) = xlsread('Series3');
IMPT(:,:,4) = xlsread('Series4');
IMPT(:,:,5) = xlsread('Series5');
IMPT(:,:,6)= xlsread('Series36');
IMPT(:,:,7)= xlsread('Series37');
IMPT(:,:,1)  = xlsread('Series6');
IMP_Norm = zeros(size(IMPT,1),size(IMPT,2),size(IMPT,3));
for i_num1 = 1:7
    mean_data(:,:,i_num1) = mean(IMPT(:,:,i_num1),1);
    max_data(:,:,i_num1) = max(IMPT(:,:,i_num1));
    min_data(:,:,i_num1) = min(IMPT(:,:,i_num1));
    IMP_Norm(:,:,i_num1) = (IMPT(:,:,i_num1)-mean_data(:,:,i_num1))./(max_data(:,:,i_num1)-min_data(:,:,i_num1));
end
IMP_Norm_Scale = (IMP_Norm + 1) * 0.4 + 0.1;
for i_num2=1:7
    data(i_num2,:) = reshape(IMP_Norm(:,:,i_num2),1,[]);
end
% Nwindow = 10; %Number of Window
% Noverlap = 3; % number of overlap time steps
% alpha = .01; % Learning Rate
% T = [ones(1,200);zeros(1,200)];
% [NC TT] = size(IMP5);
% count =1;
%% Normalize the data
% % meanVal = mean((IMP5),2);
% % stdVal = std(IMP5')';
% % maxVal = max(IMP5,[],2);
% % minVal = min(IMP5,[],2);
% % data1 = (IMP5 - meanVal)./(maxVal-minVal);
% % data1 = (data1 + 1) * 0.4 + 0.1;
%% Time Window
% % for i = 1:Noverlap:(TT - Nwindow)
% %       seg = data1(:,i:(i+ Nwindow));
% %       data1(:,count)=reshape(seg,[NC*(Nwindow+1),1]); %New Input matrix after data preprocessed
% %       count = count+1;
% % end
%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure 
% the range of pixel values is also bounded between [0,1]
%data1 = normalizeData(data1);
 
end

%% ---------------------------------------------------------------
% % function data1 = normalizeData(data1)
% % 
% % % Squash data to [0.1, 0.9] since we use sigmoid as the activation
% % % function in the output layer
% % 
% % % Remove DC (mean of images). 
% % data1 = bsxfun(@minus, data1, mean(data1));
% % 
% % % Truncate to +/-3 standard deviations and scale to -1 to 1
% % pstd = 3 * std(data1(:));
% % data1 = max(min(data1, pstd), -pstd) / pstd;
% % 
% % % Rescale from [-1,1] to [0.1,0.9]
% % data1 = (data1 + 1) * 0.4 + 0.1;
% % 
% % end
