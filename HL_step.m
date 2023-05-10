function [result] = HL_step(dataset, param, plot_loss_acc, alpha, beta)
warning off;
fprintf('Hashing learning step\n');
Xs = dataset.Xs;        % 源域样本
ns = dataset.ns;        % 源域样本数量
Xt = dataset.Xt;        % 目标域样本
nt = dataset.nt;        % 目标域样本数量（训练集）
Ys = dataset.Ys;
n = ns + nt;
X_test = dataset.Xt_test;      % 测试集样本
c = dataset.c;

%% 标签编码
d = size(dataset.Xs, 2);
Ys_emb = label_embedding(Ys, d);        
Yt_emb = label_embedding(dataset.Yt_pseudo, d);  % 伪
Y_test_emb = label_embedding(dataset.Y_test_pseudo, d);  % 伪

data_s = Xs + beta * Ys_emb;
data_t = Xt + beta * Yt_emb;
data_test = X_test + beta * Y_test_emb;

%% 标签转为onehot编码
Ys_one = (full(ind2vec(Ys', c)))';      
Yt_one_pseudo = (full(ind2vec(dataset.Yt_pseudo', c)))';   % 伪
Y_train = [Ys_one; Yt_one_pseudo];                % 训练集标签
%% 相似矩阵
S = Y_train * Y_train' > 0;   % 相似矩阵


bit = param.bit;
maxIter = param.maxIter;
m = param.m;
lambda = param.lambda;
gama = param.gama;

%% 初始化
Bt = ones(nt, bit);             % 初始化Bt
Bt(randn(nt, bit) < 0) = -1;

Bs = ones(ns, bit);             % 初始化Bs
Bs(randn(ns, bit) < 0) = -1;
B = [Bs; Bt];
V = B;                        % 初始化V

Wt = data_t \ Bt;             % 初始化W
Ws = data_s \ Bs;             % 初始化W


%% plot the convergence curve
if plot_loss_acc ==1 
    figure('Color','w');
    h = animatedline;
    h.Color = 'r';
    h.LineWidth = 1.3;
    h.LineStyle = '-.';
    title('loss');

    figure('Color','w');
    h2 = animatedline;
    h2.Color = 'g';
    h2.LineWidth = 1.3;
    h2.LineStyle = '-.';
    title('mAP');
end
%% 
opts.record = 0;                % OptStiefelGBB 的参数
opts.mxitr  = 30;
opts.xtol = 1e-3;
opts.gtol = 1e-3;
opts.ftol = 1e-4;
tic
for t = 1:maxIter
    
    % updata Wt
    [Wt, ~] = OptStiefelGBB(Wt, @fun_Wt, opts, Bt, data_t, alpha);
    
    % updata Ws
    [Ws, ~] = OptStiefelGBB(Ws, @fun_Ws, opts, Bs, data_s, alpha);
        
    % update V
    m_ind = randperm(n, m);
    Sm = S(:, m_ind);       % 大小：n*m ，相似矩阵
    V = updateColumnV(V, B, Sm, m_ind, bit, lambda, m, gama);
    
    % update Bs
    Bs = updateColumnBs(V, B, Bs, Ws, data_s, Sm, m_ind, bit, lambda, m, gama, ns, alpha);

    % update Bt
    Bt = updateColumnBt(V, B, Bt, Wt, data_t, Sm, m_ind, bit, lambda, m, gama, ns, nt, alpha);
    B = [Bs; Bt];
   
    if plot_loss_acc == 1
            
    % objective function value
        [loss(t)] = objective_function(Wt, Ws, Bs, Bt, V, data_s, data_t, lambda, alpha, gama, S, bit);
        addpoints(h, t, loss(t));
        drawnow;
    
    % real-time evaluation
        B_test = (data_test * Wt >= 0);       % 生成测试集汉明码
        B_train = (data_s * Ws >= 0);                % 生成源域汉明码
        B_tr_comp = compactbit(B_train);
        B_te_comp = compactbit(B_test);       
        Dhamm = hammingDist(B_te_comp, B_tr_comp);
        [recall, precision, ~] = recall_precision(dataset.WTT, Dhamm);
        mAP(t) = area_RP(recall, precision);
        addpoints(h2, t, mAP(t));
        drawnow;
    end
end
toc
B_test = (data_test * Wt >= 0);       % 生成测试集汉明码
B_te_comp = compactbit(B_test);

B_train = (data_s * Ws >= 0);                % cross-domain  Ws
B_tr_comp = compactbit(B_train);
Dhamm = hammingDist(B_te_comp, B_tr_comp);
[recall, precision, ~] = recall_precision(dataset.WTT, Dhamm);
result.cross_MAP_W = area_RP(recall, precision) * 100;

B_train_B = (Bs >= 0);                      % cross-domain  Bs
B_tr_comp_B = compactbit(B_train_B);
Dhamm_B = hammingDist(B_te_comp, B_tr_comp_B);
[recall, precision, ~] = recall_precision(dataset.WTT, Dhamm_B);
result.cross_MAP_B = area_RP(recall, precision) * 100;

B_train_single = (data_t * Wt >= 0);        % single-domain  Wt
B_tr_comp_single = compactbit(B_train_single);
Dhamm = hammingDist(B_te_comp, B_tr_comp_single);
[recall, precision, ~] = recall_precision(dataset.WTT_single, Dhamm);
result.single_MAP_W = area_RP(recall, precision) * 100;

B_train_single_B = (Bt >= 0);               % single-domain  Bt
B_tr_comp_single_B = compactbit(B_train_single_B);
Dhamm_B = hammingDist(B_te_comp, B_tr_comp_single_B);
[recall, precision, ~] = recall_precision(dataset.WTT_single, Dhamm_B);
result.single_MAP_B = area_RP(recall, precision) * 100;

end


function [L, J] = fun_Wt(Wt, Bt, data_t, alpha)
L = alpha * trace((Bt - data_t * Wt)' * (Bt - data_t * Wt)) / 2;
J = alpha * (data_t' * data_t * Wt - data_t' * Bt);
end

function [L, J] = fun_Ws(Ws, Bs, data_s, alpha)
L = alpha * trace((Bs - data_s * Ws)' * (Bs - data_s * Ws)) / 2;
J = alpha * (data_s' * data_s * Ws - data_s' * Bs);
end

function V = updateColumnV(V, B, Sm, m_ind, bit, lambda, m, gama)
n = size(B, 1);
for i = 1:bit
    theta = lambda * B(m_ind, :) * V' / bit;
    A = (1 ./ (1 + exp(-theta)))';
    Bji = B(m_ind, i)';
    omega = Sm - A;
    omega(Sm == 0) = A(Sm == 0);
    omega = omega.^gama;
    p = lambda / bit * omega .* ((Sm - A) .* repmat(Bji, n, 1)) * ones(m, 1) + m *lambda^2 / (4 * bit^2) * V(:, i);
    V_sgn = ones(n, 1);
    V_sgn(p < 0) = -1;
    V(:, i) = V_sgn;
end
end

function Bs = updateColumnBs(V, B, Bs, Ws, data_s, Sm, m_ind, bit, lambda, m, gama, ns, lambda2)
n = size(B, 1);
for i = 1:bit
    theta = lambda * B * V(m_ind, :)' / bit;
    A = 1 ./ (1 + exp(-theta));
    Vji = V(m_ind, i)';
    omega = Sm - A;
    omega(Sm == 0) = A(Sm == 0);
    omega = omega.^gama;
    Km = lambda / bit * omega .* ((Sm - A) .* repmat(Vji, n, 1)) * ones(m, 1);
    p =  Km(1: ns) ...
        + (m * lambda^2 / (4 * bit^2)+ lambda2) * Bs(:, i) ...
        + lambda2 * (Bs(:, i) - data_s * Ws(:, i));
    Bs_sgn = ones(ns, 1);
    Bs_sgn(p < 0) = -1;
    Bs(:, i) = Bs_sgn;
end
end

function Bt = updateColumnBt(V, B, Bt, Wt, data_t, Sm, m_ind, bit, lambda, m, gama, ns, nt, alpha)
n = size(B, 1);
for i = 1:bit
    theta = lambda * B * V(m_ind, :)' / bit;
    A = 1 ./ (1 + exp(-theta));
    Vji = V(m_ind, i)';
    omega = Sm - A;
    omega(Sm == 0) = A(Sm == 0);
    omega = omega.^gama;
    Km = lambda / bit * omega .* ((Sm - A) .* repmat(Vji, n, 1)) * ones(m, 1);
    p =  Km(ns + 1: n) ...
        + (m * lambda^2 / (4 * bit^2)+ alpha) * Bt(:, i) ...
        + alpha * (Bt(:, i) - data_t * Wt(:, i));
    Bt_sgn = ones(nt, 1);
    Bt_sgn(p < 0) = -1;
    Bt(:, i) = Bt_sgn;
end
end

function [loss] = objective_function(Wt, Ws, Bs, Bt, V, data_s, data_t, lambda, alpha, gama, S, bit)
l1 = trace((Bt - data_t * Wt)' * (Bt - data_t * Wt)) + trace((Bs - data_s * Ws)' * (Bs - data_s * Ws));

B = [Bs; Bt];
theta = lambda * B * V' / bit;
A = 1 ./ (1 + exp(-theta));
omega = S - A;
omega(S == 0) = A(S == 0);
omega = omega.^gama;
p = omega .* (S .* theta - log(1 + exp(theta)));
l2 = sum(sum(p));

loss = alpha / 2 * l1 - l2;
end



