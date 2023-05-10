function [dataset] = DAL_step(dataset, dimension)
fprintf('Domain adaptation learning step\n');
T = 10;
tic
[domainS_proj, domainT_proj, predLabels, test_proj, test_predLabels] = ...
    DA_LPP_SP_test(dataset.Xs, dataset.Ys', dataset.Xt, dataset.Yt', dataset.Xt_test, dataset.Yt_test, dimension, T);
toc
correct_num = length(find(predLabels == dataset.Yt'));
rate = correct_num / length(dataset.Yt);
fprintf('SLPP_acc_tr=%.4f \n', rate);

dataset.Xs = [dataset.Xs, domainS_proj];        % 特征融合
dataset.Xt = [dataset.Xt, domainT_proj];
dataset.Xt_test = [dataset.Xt_test, test_proj];

dataset.Yt_pseudo = predLabels';       % 源域伪标签
dataset.Y_test_pseudo = test_predLabels';       % 目标域伪标签
end