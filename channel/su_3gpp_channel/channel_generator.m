function h = channel_generator(n_r_v, n_r_h, n_t_v, n_t_h, n_sc, timeslots)
    s = qd_simulation_parameters;
    s.center_frequency = 2.4e9;
    s.sample_density = 2.1;
    s.show_progress_bars = 0;
    s.use_absolute_delays = 1;
    LightSpeed = 299792458;
    half_wave_distance = (LightSpeed/s.center_frequency)*0.5;
    n_r = n_r_v * n_r_h * 2;
    n_t = n_t_v * n_t_h * 2;

    sc_bw = 2e4;

    area_len = 50;         % area of MT initial positions [meters]
    area_half = area_len/2;
    track_distance = half_wave_distance*0.5*timeslots;  % [meters] - for 300MHz
    track_speed = 0.9;  % [meters/second] - for 300MHz

    x_i = area_len*rand(1) - area_half;
    y_i = 0;
    % theta = pi*(2*rand(1) - 1);
    t = qd_track('circular', track_distance);
    t.initial_position = [x_i; y_i; 1.6];
    t.interpolate_positions( s.samples_per_meter ); 
    t.set_speed( track_speed );
    t.scenario            = {'3GPP_3D_UMa_LOS'};
    t.no_segments = 1;

    l = qd_layout( s );
    l.rx_track = t;
        
    l.tx_position = [0 0 25]';

    l.tx_array = qd_arrayant.generate( '3gpp-3d',  n_t_v, n_t_h, s.center_frequency(1), 3, 0);
    l.rx_array = qd_arrayant.generate( '3gpp-3d',  n_r_v, n_r_h, s.center_frequency(1), 3, 0);

    l.tx_array.normalize_gain;
    l.rx_array.normalize_gain;

    % l.visualize();

    c = l.get_channels;
    h = zeros(fix(timeslots), n_r, n_t, n_sc);
    for t_i = 1:timeslots
        h(t_i,:,:,:) = reshape(c.fr(n_sc*sc_bw, n_sc, t_i, 1), 1, n_r, n_t, n_sc);
    end
end
