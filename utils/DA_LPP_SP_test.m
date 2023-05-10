% =====================================================================
% Code for conference paper:
% Qian Wang, Penghui Bu, Toby Breckon, Unifying Unsupervised Domain
% Adaptation and Zero-Shot Visual Recognition, IJCNN 2019
% By Qian Wang, qian.wang173@hotmail.com
% =====================================================================
function [domainS_proj, domainT_proj, predLabels, test_proj, test_predLabels] = DA_LPP_SP_test(domainS_features,domainS_labels,domainT_features,domainT_labels,test,test_labels,d,T)
domainS_features_ori = L2Norm(domainS_features);
domainS_labels = domainS_labels;
domainT_features_ori = L2Norm(domainT_features);
domainT_labels = domainT_labels;

opts.ReducedDim = d;
X = double([domainS_features_ori;domainT_features_ori]);
P_pca = PCA(X,opts);

domainS_features = domainS_features_ori*P_pca;      % PCA 降维
domainT_features = domainT_features_ori*P_pca;      

domainS_features = L2Norm(domainS_features);        % 行归一化
domainT_features = L2Norm(domainT_features);

num_iter = T;
options.NeighborMode='KNN';
options.WeightMode = 'HeatKernel';
options.k = 30;
options.t = 1;
options.ReducedDim = d;
options.alpha = 1;
num_class = length(unique(domainS_labels));
W_all = zeros(size(domainS_features,1)+size(domainT_features,1));
W_s = constructW1(domainS_labels);
W = W_all;      % 相似矩阵
W(1:size(W_s,1),1:size(W_s,2)) =  W_s;      % 只给源域赋值，目标域部分全为0

% looping
p = 1;
% fprintf('d=%d\n',options.ReducedDim);
for iter = 1:num_iter
    P = LPP([domainS_features;domainT_features],W,options);
    %P = LPP(domainS_features,W_s,options);
    domainS_proj = domainS_features*P;
    domainT_proj = domainT_features*P;
    proj_mean = mean([domainS_proj;domainT_proj]);      % 源域和目标域的均值
    domainS_proj = domainS_proj - repmat(proj_mean,[size(domainS_proj,1) 1 ]);      % zs --> zs - mean
    domainT_proj = domainT_proj - repmat(proj_mean,[size(domainT_proj,1) 1 ]);      % zt --> zt - mean
    domainS_proj = L2Norm(domainS_proj);        % ||zs||  l2正则
    domainT_proj = L2Norm(domainT_proj);        % ||zt||
    
    %% distance to class means
    classMeans = zeros(num_class,options.ReducedDim);
    for i = 1:num_class
        classMeans(i,:) = mean(domainS_proj(domainS_labels==i,:));
    end
    classMeans = L2Norm(classMeans);    % 源域类中心
    
    targetClusterMeans = vgg_kmeans(double(domainT_proj'), num_class, classMeans')';        % 目标域簇中心
    targetClusterMeans = L2Norm(targetClusterMeans);
    
    distClassMeans = EuDist2(domainT_proj,classMeans);      % 目标域样本到源域类中心距离
    distClusterMeans = EuDist2(domainT_proj,targetClusterMeans);        % 目标域样本到目标域簇中心距离
    
    expMatrix = exp(-distClassMeans);
    expMatrix2 = exp(-distClusterMeans);
    probMatrix = expMatrix./repmat(sum(expMatrix,2),[1 num_class]);     % NCP
    probMatrix2 = expMatrix2./repmat(sum(expMatrix2,2),[1 num_class]);  % SP
    probMatrix = max(probMatrix,probMatrix2);
    %probMatrix = probMatrix2;
    [prob,predLabels] = max(probMatrix');
    
    % 挑选样本
    p=1-iter/(num_iter-1);
    p = max(p,0);
    [sortedProb,index] = sort(prob);
    sortedPredLabels = predLabels(index);
    trustable = zeros(1,length(prob));
    for i = 1:num_class
        thisClassProb = sortedProb(sortedPredLabels==i);
        if length(thisClassProb)>0
            bbb = thisClassProb(floor(length(thisClassProb)*p)+1);
            trustable = trustable+ (prob>bbb).*(predLabels==i);
        end
    end
    pseudoLabels = predLabels;
    pseudoLabels(~trustable) = -1;
    W = constructW1([domainS_labels,pseudoLabels]);
    
    %% calculate ACC
    acc(iter) = sum(predLabels==domainT_labels)/length(domainT_labels);
    for i = 1:num_class
        acc_per_class(iter,i) = sum((predLabels == domainT_labels).*(domainT_labels==i))/sum(domainT_labels==i);
    end
%     fprintf('Iteration=%d/%d, Acc:%0.3f,Mean acc per class: %0.3f\n', iter,num_iter, acc(iter), mean(acc_per_class(iter,:)));
    if sum(trustable)>=length(prob)
        break;
    end
end
test = L2Norm(test);
test = test*P_pca;      % PCA 降维
test = L2Norm(test);

test_proj = test * P;
test_proj = test_proj - repmat(proj_mean,[size(test_proj,1) 1 ]);
test_proj = L2Norm(test_proj); 

distClassMeans = EuDist2(test_proj,classMeans);      % 目标域样本到源域类中心距离
distClusterMeans = EuDist2(test_proj,targetClusterMeans);        % 目标域样本到目标域簇中心距离
expMatrix = exp(-distClassMeans);
expMatrix2 = exp(-distClusterMeans);
probMatrix = expMatrix./repmat(sum(expMatrix,2),[1 num_class]);     % NCP
probMatrix2 = expMatrix2./repmat(sum(expMatrix2,2),[1 num_class]);  % SP
probMatrix = max(probMatrix,probMatrix2);
%probMatrix = probMatrix2;
[test_prob,test_predLabels] = max(probMatrix');
% accu = sum(test_predLabels==test_labels')/length(test_labels);
% fprintf('SLPP_acc_te:%0.4f\n', accu);
end


