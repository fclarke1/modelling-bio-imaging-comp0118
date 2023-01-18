% code for Q1.1

% load the data
%   108 image volumes each with a different gradient
%   x 145x174 voxels in each image
%   x 145 image slices through brain
% resolution 1.25x1.25x1.25	mm3
load('data');
dwis=double(dwis);
dwis=permute(dwis,[4,1,2,3]);

%% 


% check the data has loaded correctly
% Middle slice of the 1st image volume, which has b=0
imshow(flipud(squeeze(dwis(1,:,:,72))'), []);

%%

% Middle slice of the 2nd image volume, which has b=1000
imshow(flipud(squeeze(dwis(1,:,:,72))'), []);

%%

% load the gradient directions
qhat = load('bvecs');
% calculate b values for each image using qhat
bvals = 1000*sum(qhat.*qhat);

%%
% Q1.1.1 START

% measurements for one voxel
Avox = dwis(:,92,65,72);

