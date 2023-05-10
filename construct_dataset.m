function dataset = construct_dataset(dataname, test_num)
NumNeighbors = 1;       % knn-neighbors
switch dataname
    case 'MNIST-USPS'
        disp('Conducting MNIST-USPS task');
        
        load MNIST_vs_USPS X_src X_tar Y_src Y_tar;
        X_src = double(normalize1(X_src'));
        X_tar = double(normalize1(X_tar'));
        Y_src = double(Y_src);
        Y_tar = double(Y_tar);
                
    case 'USPS-MNIST'
        disp('Conducting USPS-MNIST task');
        
        load USPS_vs_MNIST X_src X_tar Y_src Y_tar;
        X_src = double(normalize1(X_src'));
        X_tar = double(normalize1(X_tar'));
        Y_src = double(Y_src);
        Y_tar = double(Y_tar);
        
    case 'VOC2007-Caltech101'
        disp('Conducting VOC2007-Caltech101 task');
        
        load VOC2007 data;
        X_src = double(normalize1(data(:, 1:4096)));       
        Y_src = double(data(:, 4097));
        clear data;
        
        load Caltech101 data;
        X_tar = double(normalize1(data(:, 1:4096)));
        Y_tar = double(data(:, 4097));
        clear data
                
    case 'Caltech256-ImageNet'
        disp('Conducting Caltech256-ImageNet task');
        
        load dense_caltech256_decaf7_subsampled fts labels;
        X_src = double(fts);
        Y_src = double(labels);
        clear fts labels;
        
        load dense_imagenet_decaf7_subsampled fts labels;
        X_tar = double(fts);
        Y_tar = double(labels);
        clear fts labels;
                
    case 'Pr-Rw'
        disp('Conducting Pr-Rw task');
        
        load Product_feature_mat deepfea label;
        X_src = double(normalize1(deepfea));
        Y_src = double(label' + 1);
        clear deepfea label;
        
        load Real_World_feature_mat deepfea label;
        X_tar = double(normalize1(deepfea));
        Y_tar = double(label' + 1);
        clear deepfea label;
        
    case 'Rw-Pr'
        disp('Conducting Rw-Pr task');
        
        load Real_World_feature_mat deepfea label;
        X_src = double(normalize1(deepfea));
        Y_src = double(label' + 1);
        clear deepfea label;
        
        load Product_feature_mat deepfea label;
        X_tar = double(normalize1(deepfea));
        Y_tar = double(label' + 1);
        clear deepfea label;
        
    case 'Cl-Rw'
        disp('Conducting Cl-Rw task');
        
        load Clipart_feature_mat deepfea label;
        X_src = double(normalize1(deepfea));
        Y_src = double(label' + 1);
        clear deepfea label;
        
        load Real_World_feature_mat deepfea label;
        X_tar = double(normalize1(deepfea));
        Y_tar = double(label' + 1);
        clear deepfea label;
        
    case 'Rw-Cl'
        disp('Conducting Rw-Cl task');
        
        load Real_World_feature_mat deepfea label;
        X_src = double(normalize1(deepfea));
        Y_src = double(label' + 1);
        clear deepfea label;
        
        load Clipart_feature_mat deepfea label;
        X_tar = double(normalize1(deepfea));
        Y_tar = double(label' + 1);
        clear deepfea label;
        
    case 'Ar-Rw'
        disp('Conducting Ar-Rw task');
        
        load Art_feature_mat deepfea label;
        X_src = double(normalize1(deepfea));
        Y_src = double(label' + 1);
        clear deepfea label;
        
        load Real_World_feature_mat deepfea label;
        X_tar = double(normalize1(deepfea));
        Y_tar = double(label' + 1);
        clear deepfea label;
        
    case 'Rw-Ar'
        disp('Conducting Rw-Ar task');
        
        load Real_World_feature_mat deepfea label;
        X_src = double(normalize1(deepfea));
        Y_src = double(label' + 1);
        clear deepfea label;
        
        load Art_feature_mat deepfea label;
        X_tar = double(normalize1(deepfea));
        Y_tar = double(label' + 1);
        clear deepfea label;
        
    case 'P27-P05'
        disp('Conducting PIE27-PIE05 task');
        
        load PIE27 fea gnd;
        X_src = double(normalize1(fea));
        Y_src = double(gnd);
        
        load PIE05 fea gnd;
        X_tar = double(normalize1(fea));
        Y_tar = double(gnd);
        
    case 'A-W'
        disp('Conducting A-W task');
        load amazon_fc6 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        
        load webcam_fc6 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
       
    case 'A-D'
        disp('Conducting A-D task');
        load amazon_fc6 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        
        load dslr_fc6 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
    
    case 'W-D'
        disp('Conducting W-D task');
        load webcam_fc6 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        
        load dslr_fc6 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
        
    case 'D-A'
        disp('Conducting D-A task');
        load dslr_fc6 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        
        load amazon_fc6 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
       
    case 'W-A'
        disp('Conducting W-A task');
        load webcam_fc6 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        
        load amazon_fc6 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
    
    case 'D-W'
        disp('Conducting D-W task');
        load dslr_fc6 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        
        load webcam_fc6 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
        
    case 'COIL1-COIL2'
        disp('Conducting COIL1-COIL2 task');
        
        load COIL_1 X_src X_tar Y_src Y_tar;
        X_src = double(normalize1(X_src'));
        X_tar = double(normalize1(X_tar'));
        Y_src = double(Y_src);
        Y_tar = double(Y_tar);
                
    case 'COIL2-COIL1'
        disp('Conducting COIL2-COIL1 task');
        
        load COIL_2 X_src X_tar Y_src Y_tar;
        X_src = double(normalize1(X_src'));
        X_tar = double(normalize1(X_tar'));
        Y_src = double(Y_src);
        Y_tar = double(Y_tar);    
  
end

c = length(unique(Y_tar));  % The number of classes;
dataset.c = c;

%% 挑选测试集
randIdx = randperm(length(Y_tar));
if test_num < 1
    sele_num = round(test_num * size(X_tar, 1));    % 百分之十作为测试集
else
    sele_num = test_num;
end
Xt_test = X_tar(randIdx(1: sele_num), :);       % 测试集
Yt_test = Y_tar(randIdx(1: sele_num));
Xt = X_tar(randIdx(sele_num + 1: length(Y_tar)), :);        % 目标域训练集
Yt = Y_tar(randIdx(sele_num + 1: length(Y_tar)));
nt = length(Y_tar) - sele_num;          % 目标域训练集数量

% 测试集
dataset.Xt_test = Xt_test;

% 剩下一部分目标域和全部源域样本作为训练集
dataset.Xs = X_src;
dataset.Xt = Xt;

dataset.nt = nt;
dataset.ns = size(X_src, 1);

dataset.Yt_test = Yt_test;

%% knn制作伪标签
model = fitcknn(X_src, Y_src, 'NumNeighbors', NumNeighbors);        

Yt_pseudo = predict(model, Xt);         % 训练集伪标签
correct_num = length(find(Yt_pseudo == Yt));
rate = correct_num / nt;
fprintf('KNN_acc_tr=%.4f \n', rate);

Y_test_pseudo = predict(model, Xt_test);        % 测试集伪标签
correct_num = length(find(Y_test_pseudo == Yt_test));
rate = correct_num / length(Yt_test);
fprintf('KNN_acc_te=%.4f \n', rate);

% dataset.Yt = Yt_pseudo;                     % 训练集伪标签
% dataset.Y_test_pseudo = Y_test_pseudo;      % 测试集伪标签
dataset.Ys = Y_src;
dataset.Yt = Yt;



%% 用于PR曲线
YS = repmat(Y_src, 1, length(Yt_test));     
YT = repmat(Yt_test, 1, length(Y_src));
WTT = (YT==YS');
dataset.WTT = WTT;

YT = repmat(Yt, 1, length(Yt_test));
YTest = repmat(Yt_test,1,length(Yt));
WTT_single = (YTest==YT');
dataset.WTT_single = WTT_single;


end

