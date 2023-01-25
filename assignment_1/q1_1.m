format short e

%% code for Q1.1

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
imshow(flipud(squeeze(dwis(2,:,:,72))'), []);

%%

% load the gradient directions
qhat = load('bvecs');
% calculate b values for each image using qhat
bvals = 1000*sum(qhat.*qhat);

%%
% Q1.1.1 START

% solve for x in log(A) = G x where x has all unknowns
x = zeros(145, 174, 7);
quadratic_matrix = -bvals'.*[qhat(1,:).^2; 2.*qhat(1,:).*qhat(2,:); 2.*qhat(1,:).*qhat(3,:); qhat(2,:).^2; 2.*qhat(2,:).*qhat(3,:); qhat(3,:).^2]';
G = [ones(108,1) quadratic_matrix];

% solve for x using weighted standard dviation. This is needed because we
% have taken the log of the measurements
for i = 1:145
    for j = 1:174
        A = dwis(:,i,j,72);
        if (min(A) > 0)
            W = diag(A.^2);
            invmap = inv(G'*W*G)*G'*W;
            x(i,j,:) = invmap*log(A);
        end
    end
end

clear Ginv quadratic_matrix invmap W

% Calculate Diffusion Tensor
D = zeros(145, 174, 3, 3);
for i = 1:145
    for j = 1:174
        Dxx = x(i,j,2);
        Dxy = x(i,j,3);
        Dxz = x(i,j,4);
        Dyy = x(i,j,5);
        Dyz = x(i,j,6);
        Dzz = x(i,j,7);
        D(i,j,1,:) = [Dxx Dxy Dxz];
        D(i,j,2,:) = [Dxy Dyy Dyz];
        D(i,j,3,:) = [Dxz Dyz Dzz];
    end
end
clear Dxx Dxy Dxz Dyy Dyz Dzz


%%
% calculate mean diffusivity
mean_D = zeros(145,174);

for i = 1:145
    for j = 1:174
        mean_D(i,j) = trace(D(i,j)) / 3;
    end
end

% plot mean Diffusivity
imshow(flipud(mean_D'), []);

%%

% calculate FA
FA = zeros(145,174);
eig_val_D = zeros(145,174,3);
eig_vec_D = zeros(145,174,3,3);

for i = 1:145
    for j = 1:174
        [eig_vec, eig_val] = eig(squeeze(D(i,j,:,:)));
        eig_val_D(i,j,:) = diag(eig_val);
        eig_vec_D(i,j,1,:) = squeeze(eig_vec(1,:));
        eig_vec_D(i,j,2,:) = squeeze(eig_vec(2,:));
        eig_vec_D(i,j,3,:) = squeeze(eig_vec(3,:));
        FA(i,j) = sqrt(1.5 * sum((diag(eig_val) - mean(diag(eig_val)) ).^2) / sum(diag(eig_val).^2));
    end
end

clear eig_val eig_vec

%%
% plot the model estimate against the measured signal at voxel 96,65 across
% all slices
Aest = exp(G*squeeze(x(96,65,:)));
A = squeeze(dwis(:,96,65,72));
plot(Aest, ' gx');
hold on; plot(A, ' rx');

%%
% plot FA

imshow(flipud(FA'), []);

clear Dxx Dyy Dzz

%%
% plot the FA weighted with eigenvalues on RGB spectrum

FA_RGB = zeros(145, 174, 3);

for i = 1:145
    for j = 1:174
        [eig_val, principal_eig_val_idx] = max(squeeze(eig_val_D(i,j,:)));
        if eig_val > 0
            eig_vec = squeeze(eig_vec_D(i,j, principal_eig_val_idx, :));
            FA_RGB(i,j,:) = FA(i,j)*abs([eig_vec(2), eig_vec(1), eig_vec(3)]);
        end
    end
end

% normalise RGB values
FA_RGB = FA_RGB ./ max(FA_RGB, [], 'all');

imshow(flipud(permute(FA_RGB, [2,1,3])), []);

clear eig_vec principal_eig_val_idx eig_val

%% Q 1.1.2


