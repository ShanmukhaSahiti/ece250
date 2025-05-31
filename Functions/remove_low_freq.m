function y = remove_low_freq(x,w)

filt = (1/w)*ones(1,w);
if size(x,1) ~= 1
lf = conv2(filt,1, x, 'same');
else
    lf = conv(x, filt, 'same');
end
y = x - lf;