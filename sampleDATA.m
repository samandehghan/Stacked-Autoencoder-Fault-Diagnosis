function data = sampleDATA()
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

IMP(:,:,2) = xlsread('Series2');
IMP(:,:,3) = xlsread('Series3');
IMP(:,:,4) = xlsread('Series4');
IMP(:,:,5) = xlsread('Series5');
IMP(:,:,6) = xlsread('Series6');
IMP(:,:,7) = xlsread('Series7');
IMP(:,:,8) = xlsread('Series8');
IMP(:,:,9) = xlsread('Series9');
IMP(:,:,10) = xlsread('Series10');
IMP(:,:,11) = xlsread('Series11');
IMP(:,:,12) = xlsread('Series12');
IMP(:,:,13) = xlsread('Series13');
IMP(:,:,14)= xlsread('Series14');
IMP(:,:,15)= xlsread('Series15');
IMP(:,:,16)= xlsread('Series16');
IMP(:,:,17) = xlsread('Series17');
IMP(:,:,18) = xlsread('Series18');
IMP(:,:,19) = xlsread('Series19');
IMP(:,:,20) = xlsread('Series20');
IMP(:,:,21) = xlsread('Series21');
IMP(:,:,22) = xlsread('Series22');
IMP(:,:,23) = xlsread('Series23');
IMP(:,:,24)= xlsread('Series24');
IMP(:,:,25)= xlsread('Series25');
IMP(:,:,26)= xlsread('Series26');
IMP(:,:,27)= xlsread('Series27');
IMP(:,:,28)= xlsread('Series28');
IMP(:,:,29) = xlsread('Series29');
IMP(:,:,30) = xlsread('Series30');
IMP(:,:,31) = xlsread('Series31');
IMP(:,:,32) = xlsread('Series32');
IMP(:,:,33) = xlsread('Series33');
IMP(:,:,34)= xlsread('Series34');
IMP(:,:,35)= xlsread('Series35');
IMP(:,:,36)= xlsread('Series36');
IMP(:,:,37)= xlsread('Series37');
IMP(:,:,38)= xlsread('Series38');
IMP(:,:,1)= xlsread('Series39');
IMP_Norm = zeros(size(IMP,1),size(IMP,2),size(IMP,3));
for i_num1 = 1:38
    mean_data(:,:,i_num1) = mean(IMP(:,:,i_num1),1);
    max_data(:,:,i_num1) = max(IMP(:,:,i_num1));
    min_data(:,:,i_num1) = min(IMP(:,:,i_num1));
    IMP_Norm(:,:,i_num1) = (IMP(:,:,i_num1)-mean_data(:,:,i_num1))./(max_data(:,:,i_num1)-min_data(:,:,i_num1));
end
IMP_Norm_Scale = (IMP_Norm + 1) * 0.4 + 0.1;
for i_num2=1:38
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
