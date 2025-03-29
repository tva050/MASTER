function [grid_lat,grid_lon,P_grid,P_unc_grid,N_grid,N_tracks_grid] = grid_parameter(P_t,P_lat,P_lon,P_parameter,P_unc,grid_folder,grid,method,smoothing_scale,prec_redu,sic_folder)

%% Grid scattered samples

% Grid points treated as centres of grid cells

% Parallel processing available

% Data cannot contain NaN values

%%% Available Grids %%%
% v1north_polarstereo_80km = used in PREMELT for V1 of year-round SIT product
% v2north_polarstereo_80km = same as above but extended to lower latitude
% north_polarstereo_12km = 12.5 km grid from Bremen melt pond datasets
% EASE_25km = standard NSIDC 25-km grid for e.g. SnowModel-LG
% SINXS_EASE2_N25_356 = standardized 25-km EASE2 grid used in SIN'XS

%%% Method 0 %%%
% Drop in bucket averaging
% No smoothing option

%%% Method 1 %%%
% Inverse distance weighting
% smoothing_scale includes samples at a distance of
% smoothing_scale*resolution*0.5 away from the grid cell centre (requires
% P_t and SIC data)

%%% Uncertainty Weighting %%%
% Weights grid solution by the inverse of the parameter uncertainty after
% scaling uncertainties over the 5th-95th percentile range

%%% Precision Reduction %%%
% Assumed reduction in uncertainty by averaging N samples in a grid cell:
% 0 = no reduction, samples assumed to be totally correlated
% 1 = reduced by 1/sqrt(number of tracks crossing grid cell), which assumes
% that uncertainties between tracks are uncorrelated
% 2 = reduced by 1/sqrt(number of samples in grid cell), which assumes that
% uncertainties between samples are uncorrelated

% INPUT:
% folder = 'R:\IFT\EarthObservation\SatelliteAltimetry'; % EarthObsWin
% P_t = vector of dates (Matlab datetime format)
% P_lat = vector of latitudes
% P_lon = vector of longitudes
% P_parameter = vector of parameter to be gridded
% P_unc = vector of parameter uncertainties
% grid_folder = 'Grid Files';
% grid = 'v2north_polarstereo_80km';
% method = 0;
% smoothing_scale = 1;
% prec_redu = 0;
% sic_folder = fullfile(folder,'OSISAF Sea Ice Concentration');

% OUTPUT:
% P_grid = gridded parameter
% P_unc_grid = gridded parameter uncertainty
% N_grid = number of samples used in grid cell


%% Usage
% [grid_lat,grid_lon,P_grid,P_unc_grid,N_grid,N_tracks_grid] = grid_parameter([],P_lat,P_lon,P_parameter,P_unc,'Grid Files','v2_north_polarstereo_80km',1,2,[]);
% [grid_lat,grid_lon,P_grid,P_unc_grid,N_grid,N_tracks_grid] = grid_parameter(P_t,P_lat,P_lon,P_parameter,P_unc,'Grid Files','EASE_25km',2,1,sic_folder);

%% Code dependency
% polarstereo_fwd

%% Main Code

%% Open grid

if strcmp(grid,"v1north_polarstereo_80km")
    grid_lat = load(fullfile(grid_folder,"v1north_polarstereo_80km.mat"),'lat_cs2');
    grid_lat = grid_lat.lat_cs2;
    grid_lon = load(fullfile(grid_folder,"v1north_polarstereo_80km.mat"),'lon_cs2');
    grid_lon = grid_lon.lon_cs2;
    search_radius = 80e3/2;
elseif strcmp(grid,"v2north_polarstereo_80km")
    grid_lat = load(fullfile(grid_folder,"v2north_polarstereo_80km.mat"),'lat_cs2');
    grid_lat = grid_lat.lat_cs2;
    grid_lon = load(fullfile(grid_folder,"v2north_polarstereo_80km.mat"),'lon_cs2');
    grid_lon = grid_lon.lon_cs2;
    search_radius = 80e3/2;
elseif strcmp(grid,"north_polarstereo_12km")
    grid_lat = double(hdfread(fullfile(grid_folder,"north_polarstereo_lat_12km.hdf"),'Latitude [degrees]')');
    grid_lon = double(hdfread(fullfile(grid_folder,"north_polarstereo_lon_12km.hdf"),'Longitude [degrees]')');
    search_radius = 12.5e3/2;
elseif strcmp(grid,"EASE_25km")
    fid = fopen(fullfile(grid_folder,'EASE_25km_lats.bin'));
    grid_lat = fread(fid,[361 361],'float32'); fclose(fid);
    fid = fopen(fullfile(grid_folder,'EASE_25km_lons.bin'));
    grid_lon = fread(fid,[361 361],'float32'); fclose(fid);
    search_radius = 25e3/2;
elseif strcmp(grid,"SINXS_EASE2_N25_356")
    grid_lat = ncread(fullfile(grid_folder,"SINXS_EASE2_N25_356.nc"),'latitude');
    grid_lon = ncread(fullfile(grid_folder,"SINXS_EASE2_N25_356.nc"),'longitude');
    search_radius = 25e3/2;
end

% Reference polarstereo north projection
[grid_x,grid_y] = polarstereo_fwd(grid_lat,grid_lon,6378137,0.08181919,70,0);
[P_x,P_y] = polarstereo_fwd(P_lat,P_lon,6378137,0.08181919,70,0);

%% Create weights

if isempty(P_unc)
    P_unc = ones(size(P_parameter));
else
end
w_low = prctile(P_unc,5);
w_high = prctile(P_unc,95);
w = 1 - (P_unc - w_low)/(w_high - w_low);
w(w<0) = 0; w(w>1) = 1;

%% SIC Data

if isempty(sic_folder)
else
    t_vec = datevec(P_t);

    sic_files = dir(fullfile(sic_folder, num2str(unique(t_vec(:,1))), '*.nc'));
    sic_lat = double(ncread(fullfile(sic_files(1).folder,sic_files(1).name),'lat'));
    sic_lon = double(ncread(fullfile(sic_files(1).folder,sic_files(1).name),'lon'));
    
    [sic_x,sic_y] = polarstereo_fwd(sic_lat,sic_lon,6378137,0.08181919,70,0);
    
    % daily file indices
    C_sic = NaN(length(sic_files),3);
    for j = 1:length(sic_files)
        if unique(t_vec(:,1))<=2015
            C_sic(j,:) = [str2double(sic_files(j).name(32:35)) str2double(sic_files(j).name(36:37)) str2double(sic_files(j).name(38:39))];
        else
            C_sic(j,:) = [str2double(sic_files(j).name(33:36)) str2double(sic_files(j).name(37:38)) str2double(sic_files(j).name(39:40))];        
        end
    end
end

%% Gridding

if method==0
    
    p = gcp('nocreate');
    if isempty(p)

        Idx = cell(size(grid_x));
        for j = 1:numel(grid_x)
            Idx{j} = find(inpolygon(P_x,P_y,[grid_x(j)-search_radius grid_x(j)+search_radius grid_x(j)+search_radius grid_x(j)-search_radius],[grid_y(j)+search_radius grid_y(j)+search_radius grid_y(j)-search_radius grid_y(j)-search_radius]))';
        end
    
    else

        Idx = cell(size(grid_x));
        parfor j = 1:numel(grid_x)
            Idx{j} = find(inpolygon(P_x,P_y,[grid_x(j)-search_radius grid_x(j)+search_radius grid_x(j)+search_radius grid_x(j)-search_radius],[grid_y(j)+search_radius grid_y(j)+search_radius grid_y(j)-search_radius grid_y(j)-search_radius]))';
        end

    end

    P_grid = cellfun(@(x) sum(P_parameter(x).*w(x))/sum(w(x)), Idx);
    P_unc_grid = cellfun(@(x) sum(P_unc(x).*w(x))/sum(w(x)), Idx);
    N_grid = cellfun(@numel, Idx);

elseif method==1

    r = sqrt(2*search_radius^2)*smoothing_scale;
    [Idx,D] = rangesearch([P_x P_y],[grid_x(:) grid_y(:)],r);

    w_total = cellfun(@(x,y) (r - y)/r + w(x)', Idx, D, 'UniformOutput', 0);
    
    P_grid = reshape(cellfun(@(x,y) sum(P_parameter(x)'.*y)/sum(y), Idx, w_total), size(grid_lat));
    P_unc_grid = reshape(cellfun(@(x,y) sum(P_unc(x)'.*y)/sum(y), Idx, w_total), size(grid_lat));
    N_grid = reshape(cellfun(@numel, Idx), size(grid_lat));

    % Filter by SIC
    id_sic = find(C_sic(:,2)==t_vec(1,2) & C_sic(:,3)==t_vec(1,3)):find(C_sic(:,2)==t_vec(end,2) & C_sic(:,3)==t_vec(end,3));
    SIC = NaN(432,432,length(id_sic));
    for j = 1:length(id_sic)
        SIC(:,:,j) = ncread(fullfile(sic_files(id_sic(j)).folder,sic_files(id_sic(j)).name),'ice_conc');
    end
    sic = nanmean(SIC,3);        
    
    idn = find(~isnan(sic));
    [Idx2,D2] = knnsearch([sic_x(idn) sic_y(idn)],[grid_x(:) grid_y(:)]);
    P_grid(sic(idn(Idx2))==0 | D2>sqrt(2*search_radius^2)) = NaN;
    P_unc_grid(sic(idn(Idx2))==0 | D2>sqrt(2*search_radius^2)) = NaN;
    
end

%% Reduction to uncertainty based on precision assumption

if prec_redu==0

    N_tracks_grid = NaN(size(N_grid));

elseif prec_redu==1

    % Calculate discrete orbits
    d_P_t = diff(P_t);
    d_P_xy = sqrt(diff(P_x).^2 + diff(P_y).^2);
    
    % Assume new track with >10s and >10km gap
    idT = (d_P_t > seconds(10)) & (d_P_xy > 10e3);
    track_id = [0; cumsum(idT)];

    N_tracks_grid = reshape(cellfun(@(x) length(unique(track_id(x))), Idx), size(grid_lat));
    
    P_unc_grid = P_unc_grid.*(1./sqrt(N_tracks_grid));

elseif prec_redu==2

    N_tracks_grid = NaN(size(N_grid));

    P_unc_grid = P_unc_grid.*(1./sqrt(N_grid));

end