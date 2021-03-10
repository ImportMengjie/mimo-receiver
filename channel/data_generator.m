n_r = 16;
n_t = 8;
n_sc = 64;
n_s = 64;

J = 10;

H = zeros(J,n_s,n_r,n_t,n_sc);

for i=1:J
    H(i,:,:,:,:) = channel_generator(n_r,n_t,n_sc,n_s);
end

max_value = max(max(real(H), [], 'all'), max(imag(H), [], 'all'));
power_ten = -1;
while max_value<1
    power_ten = power_ten+1;
    max_value = max_value*10;
end
H = round(H*10.^power_ten,4);

file_path = sprintf('%s/../data/h_%d_%d_%d_%d.mat',fileparts(mfilename('fullpath')),n_r, n_t, n_sc, n_s);
save(file_path, 'H', 'power_ten', '-v7.3');