clear all;
close all;
warning off;

addpath(genpath(fullfile('FOptMshare/')));
addpath(genpath(fullfile('data/')));
addpath(genpath(fullfile('data/cross-dataset')));
addpath(genpath(fullfile('data/VLSC')));
addpath(genpath(fullfile('data/Office31')));
addpath(genpath(fullfile('data/Office-Home(vgg)')));
addpath(genpath(fullfile('DA_tool/')));
addpath(genpath(fullfile('utils/')));

% load('../DAR2/random_seed.mat');
% rng(random_seed);
                               
%% ---------------------------------------------------------------

% alpha = 1e-4;    % Office-Home：Pr-Rw、Rw-Pr、Cl-Rw、Rw-Cl、Ar-Rw、Rw-Ar
% beta = 0.01;
% dimension = 512;
     
% alpha = 1e-4;      % Office31: A-W
% beta = 0.01;
% dimension = 512;

alpha = [1e-2];        % MNIST-USPS
beta = [0.05];
dimension = 32;

% alpha = 1e-4;        % COIL1-COIL2
% beta = 0.01;
% dimension = 128;
 

%% ---------------------------------------------------------------
param.gama = 0.01;
param.maxIter = 15;
plot_loss_acc = 0;      % 是否绘制曲线

bits = [16, 48, 64, 96, 128];
lambdas = [6, 6, 7, 7, 8, 8];
for b = 1:length(bits)
    mAP = [];
    for t = 1:10
        param.bit = bits(b);
        param.m = param.bit * 1;
        param.lambda = lambdas(b);
        
        dataset = construct_dataset('MNIST-USPS', 0.1);          % 构造数据集    
        [new_dataset] = DAL_step(dataset, dimension);           % domain adaptation learning step
        [result] = HL_step(new_dataset, param, plot_loss_acc, alpha, beta);   % hashing learning step
        
        fprintf('alpha=%f, beta=%.2f, cross_MAP_W=%.2f, cross_MAP_B=%.2f, single_MAP_W=%.2f, single_MAP_B=%.2f \n', ...
            alpha, beta, result.cross_MAP_W, result.cross_MAP_B, result.single_MAP_W, result.single_MAP_B);
        mAP(1, t) = result.cross_MAP_W;
        mAP(2, t) = result.cross_MAP_B;
        mAP(3, t) = result.single_MAP_W;
        mAP(4, t) = result.single_MAP_B;
        
    end
    fprintf('bit=%d \n', bits(b));
    fprintf('mean: %.2f \n', mean(mAP'));
    fprintf('std: %.2f \n', std(mAP'));
end
            
            
            
            
            
            