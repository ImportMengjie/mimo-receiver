
close all
clear all

n_r_v = 4;
n_r_h = 4;

n_t_v = 2;
n_t_h = 4;

n_r = n_r_v * n_r_h * 2;
n_t = n_t_v * n_t_h * 2;
n_sc = 64;
timeslots = 5;
n_ts = 5;

J = timeslots*n_ts;

H = ones(J,n_sc,n_r,n_t,2);

for i=1:n_ts
     h = permute(channel_generator(n_r_v,n_r_h,n_t_v,n_t_h,n_sc,timeslots), [1,4,2,3]);
     left = (i-1)*timeslots+1;
     right = i*timeslots;
     H(left:right,:,:,:,1) = real(h);
     H(left:right,:,:,:,2) = imag(h);
end

% max_value = max(abs(H), [], 'all');
% power_ten = -1;
% while max_value<1
%     power_ten = power_ten+1;
%     max_value = max_value*10;
% end
% H = round(H*10.^power_ten,4);
% disp(power_ten);
power_ten = 0;

file_path = sprintf('%s/../../data/3gpp_%d_%d_%d_%d_%d.mat',fileparts(mfilename('fullpath')),n_r, n_t, n_sc, timeslots, n_ts);
% save(file_path, 'H', '-hdf5');
save(file_path, 'H', '-v7.3');
