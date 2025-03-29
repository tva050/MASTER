function [P_parameter,land_mask_id] = land_mask(P_lat,P_lon,P_parameter,land_mask_folder)

%% Filter out scattered samples on land

% INPUT:
% folder = 'R:\IFT\EarthObservation\SatelliteAltimetry'; % EarthObsWin
% P_lat = vector of latitudes
% P_lon = vector of longitudes
% P_parameter = vector of parameter
% land_mask_folder = folder;

% OUTPUT:
% P_parameter = vector of parameter with land samples NaN
% land_mask_id = 0=ocean, 3=coast, 216=land


%% Current compatibility
% OSM global land mask cropped to the Arctic, arctic_landmask.mat

%% Code dependency
% polarstereo_fwd

%% Main Code

% Open land mask
load(fullfile(land_mask_folder,"arctic_landmask.mat"))

% Reproject to polarstereo north
[P_x,P_y] = polarstereo_fwd(P_lat,P_lon,6378137,0.08181919,70,0);
[mask_x,mask_y] = polarstereo_fwd(mask_lat,mask_lon,6378137,0.08181919,70,0);

% Find nearest neighbour for each scattered sample
[Idx,D] = knnsearch([mask_x(:) mask_y(:)],[P_x P_y]);

% NaN land samples
land_mask_id = double(landmask(Idx));
P_parameter(land_mask_id > 3) = NaN;

end