
close all
clear all

n_r = 16;
n_t = 16;
n_sc = 64;
n_s = 1.9;

J = 100;

H = zeros(J,fix(n_s),n_sc,n_r,n_t,2);

for i=1:J
    h = permute(channel_generator(n_r,n_t,n_sc,n_s), [1,4,2,3]);
    H(i,:,:,:,:,1) = real(h);
    H(i,:,:,:,:,2) = imag(h);
end

max_value = max(abs(H), [], 'all');
power_ten = -1;
while max_value<1
    power_ten = power_ten+1;
    max_value = max_value*10;
end
% H = round(H*10.^power_ten,4);
disp(power_ten);
power_ten = 0;

file_path = sprintf('%s/../data/h_%d_%d_%d_%d.mat',fileparts(mfilename('fullpath')),n_r, n_t, n_sc, fix(n_s));
save(file_path, 'H', 'power_ten', '-v7.3');