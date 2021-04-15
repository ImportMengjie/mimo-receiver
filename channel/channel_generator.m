function h = channel_generator(n_r, n_t, n_sc, n_s)
    s = qd_simulation_parameters;
    s.center_frequency = 3.0e8;
    s.sample_density = n_s;

    sc_bw = 2e4;

    area_len = 400;         % area of MT initial positions [meters]
    area_half = area_len/2;
    track_distance = physconst('LightSpeed')/s.center_frequency*0.5;  % [meters] - for 300MHz
    track_speed = 0.9;  % [meters/second] - for 300MHz

    x_i = area_len*rand(1) - area_half;
    y_i = area_len*rand(1) - area_half;
    theta = pi*(2*rand(1) - 1);
    t = qd_track('linear', track_distance, theta);
    t.initial_position = [x_i; y_i; 1.5];       
    t.interpolate_positions( s.samples_per_meter ); 
    t.set_speed( track_speed );
    t.scenario            = {'3GPP_3D_UMa_LOS'};

    l = qd_layout( s );
    l.rx_track = t;
        
    l.tx_position = [0 0 20]';

    l.tx_array = qd_arrayant.generate( '3gpp-3d',  1, n_t, s.center_frequency(1), 1);                        

    l.rx_array = qd_arrayant.generate( '3gpp-3d',  1, n_r, s.center_frequency(1), 1);                       % Set omni-rx antenna
    c = l.get_channels;
    h = zeros(fix(n_s), n_r, n_t, n_sc);
    for t_i = 1:n_s
        h(t_i,:,:,:) = reshape(c.fr(n_sc*sc_bw, n_sc, t_i), 1, n_r, n_t, n_sc);
    end
end