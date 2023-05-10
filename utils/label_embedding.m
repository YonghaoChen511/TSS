function label = label_embedding(Y, d)
% ±êÇ©Éú³É±àÂë
label = ones(length(Y), d);
j = 1;
for i = 1:2:d
    label(:, i) = sin(Y/10000^(2*j/d));
    label(:, i+1) = cos(Y/10000^(2*j/d));
    j = j+1;
end
if d < size(label, 2)
    label = label(:, 1:d);
end
end