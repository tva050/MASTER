function stats_bgep = validate_mooring_bgep(sit_folder,ref_folder,search_radius,land_mask_folder,outfolder,plot_opt,version)

%% Generates validation statistics and plots with BGEP Mooring data as reference

% Converts satellite observations to sea ice draft for comparison

% Uses all satellite observations within search_radius of each mooring over
% the time interval specified in the satellite product

% folder = 'R:\IFT\EarthObservation\SatelliteAltimetry'; % EarthObsWin
% sit_folder = fullfile('test','uit_sit_biweekly_v2north_polarstereo_80km_v2.0');
% ref_folder = fullfile(folder,'Mooring Data BGEP');
% search_radius = 200e3; % in meters
% land_mask_folder = folder;
% outfolder = 'test';
% plot_opt = 1; % 0 = no plotting, 1 = plotting
% version = '2.0';


%% Current compatibility
% OSI-SAF Global Sea Ice Concentration (OSI-401-b)


%% Code dependency
% polarstereo_fwd
% linortfit2
% linortfitn
% land_mask
% rfb_to_sit
% othercolor

%%  Reference mooring locations

% Assumed to be fixed across time series, despite small interannual
% variations https://www2.whoi.edu/site/beaufortgyre/data/mooring-data/

% https://www.mdpi.com/remotesensing/remotesensing-12-03094/article_deploy/html/images/remotesensing-12-03094-g001.png

mooring_A_lat = 75;
mooring_A_lon = -150;
mooring_B_lat = 78;
mooring_B_lon = -150;
mooring_D_lat = 74;
mooring_D_lon = -140;

% In polarstereo
[mooring_A_x,mooring_A_y] = polarstereo_fwd(mooring_A_lat,mooring_A_lon,6378137,0.08181919,70,0);
[mooring_B_x,mooring_B_y] = polarstereo_fwd(mooring_B_lat,mooring_B_lon,6378137,0.08181919,70,0);
[mooring_D_x,mooring_D_y] = polarstereo_fwd(mooring_D_lat,mooring_D_lon,6378137,0.08181919,70,0);

%% Prepare mooring data time series

% IDS: daily ice draft statistics: number, mean, std, minimum, maximum, median

% Mooring files
bgep_files_A = dir(fullfile(ref_folder,'*a_*.mat'));
bgep_files_B = dir(fullfile(ref_folder,'*b_*.mat'));
bgep_files_D = dir(fullfile(ref_folder,'*d_*.mat'));

% Create single dataset for each mooring
IDS_full_A = [];
t_full_A = [];
for i = 1:length(bgep_files_A)
    
    dates_sub = load(fullfile(ref_folder,bgep_files_A(i).name),'dates');
    t_full_A = [t_full_A; datetime(dates_sub.dates)];
    
    IDS_sub = load(fullfile(ref_folder,bgep_files_A(i).name),'IDS');
    IDS_full_A = [IDS_full_A; IDS_sub.IDS(:,1:6)];
end

IDS_full_B = [];
t_full_B = [];
for i = 1:length(bgep_files_B)
    
    dates_sub = load(fullfile(ref_folder,bgep_files_B(i).name),'dates');
    t_full_B = [t_full_B; datetime(dates_sub.dates)];
    
    IDS_sub = load(fullfile(ref_folder,bgep_files_B(i).name),'IDS');
    IDS_full_B = [IDS_full_B; IDS_sub.IDS(:,1:6)];
end

IDS_full_D = [];
t_full_D = [];
for i = 1:length(bgep_files_D)
    
    dates_sub = load(fullfile(ref_folder,bgep_files_D(i).name),'dates');
    t_full_D = [t_full_D; datetime(dates_sub.dates)];
    
    IDS_sub = load(fullfile(ref_folder,bgep_files_D(i).name),'IDS');
    IDS_full_D = [IDS_full_D; IDS_sub.IDS(:,1:6)];
end


%% Satellite data

sit_files = dir(fullfile(sit_folder,['uit_sit_*_v' version '.nc']));
sit_names = cellfun(@(grid_x) extract(grid_x,digitsPattern(8)), {sit_files.name}');
tvec_sit = cell2mat(cellfun(@(grid_x) [str2double(grid_x(1:4)) str2double(grid_x(5:6)) str2double(grid_x(7:8))], sit_names, 'uni',0));

% Grid
sit_filename = fullfile(sit_files(1).folder,sit_files(1).name);
grid_def = ncreadatt(sit_filename,'/','grid');
grid_lat = ncread(sit_filename,'latitude');
grid_lon = ncread(sit_filename,'longitude');
[grid_x,grid_y] = polarstereo_fwd(grid_lat,grid_lon,6378137,0.08181919,70,0);

% Land mask
[~,land_mask_id] = land_mask(grid_lat(:),grid_lon(:),NaN(numel(grid_lat),1),land_mask_folder);
land_mask_id = reshape(land_mask_id,size(grid_lat));

%% Export setup

outdir = ['validation_v' version];
if isfolder(fullfile(outfolder,outdir))
else
    mkdir(fullfile(outfolder,outdir))
end

%% Identify grid cells within search radius of moorings

IDX_A = rangesearch([grid_x(:) grid_y(:)],[mooring_A_x mooring_A_y],search_radius);
IDX_B = rangesearch([grid_x(:) grid_y(:)],[mooring_B_x mooring_B_y],search_radius);
IDX_D = rangesearch([grid_x(:) grid_y(:)],[mooring_D_x mooring_D_y],search_radius);


%% Prepare satellite data time series

% Convert derived sea ice freeboard to draft using:
% sid = sit - ifb
% sid = (ifb*si_dens + sd*s_dens)/(ow_dens - si_dens)

sat_A = NaN(length(sit_files),3);
sat_B = NaN(length(sit_files),3);
sat_D = NaN(length(sit_files),3);
sat_unc_A = NaN(length(sit_files),3);
sat_unc_B = NaN(length(sit_files),3);
sat_unc_D = NaN(length(sit_files),3);
ref_A = NaN(length(sit_files),3);
ref_B = NaN(length(sit_files),3);
ref_D = NaN(length(sit_files),3);
for i = 1:length(sit_files)
    
    % fprintf(['Date = ' num2str(tvec_sit(i,:)) ' \n\n'])

    %% Load sit grids
    
    sit_filename = fullfile(sit_files(i).folder,sit_files(i).name);

    sit_t = ncread(sit_filename,'time');
    sit_t_interval = ncread(sit_filename,'time_interval');
    
    sit = ncread(sit_filename,'sea_ice_thickness');
    sit_unc = ncread(sit_filename,'sea_ice_thickness_unc');
    
    ifb = ncread(sit_filename,'sea_ice_freeboard');
    ifb_unc = ncread(sit_filename,'sea_ice_freeboard_unc');
    
    sic = ncread(sit_filename,'sea_ice_concentration');
    type = ncread(sit_filename,'sea_ice_type');

    % Load relevant attributes
    sit_source_name = ncreadatt(sit_filename,'/','file_name');
    t_interval = ncreadatt(sit_filename,'/','t_interval');
    
    % Time interval of radar freeboard grid
    t_lim = datetime(datevec([sit_t-sit_t_interval/2; sit_t+sit_t_interval/2]));

    %% Convert to sea ice draft

    sid = sit - ifb;
    sid((isnan(sic) | sic<15) & isnan(sid) & land_mask_id==0) = 0;
    sid_unc = sit_unc - ifb_unc;

    %% Satellite ice draft statistics at mooring locations

    sat_A(i,1) = nanmean(sid(IDX_A{1}));
    sat_A(i,2) = nanmedian(sid(IDX_A{1}));
    sat_A(i,3) = nanstd(sid(IDX_A{1}));

    sat_unc_A(i,1) = nanmean(sid_unc(IDX_A{1}));
    sat_unc_A(i,2) = nanmedian(sid_unc(IDX_A{1}));
    sat_unc_A(i,3) = nanstd(sid_unc(IDX_A{1}));

    sat_B(i,1) = nanmean(sid(IDX_B{1}));
    sat_B(i,2) = nanmedian(sid(IDX_B{1}));
    sat_B(i,3) = nanstd(sid(IDX_B{1}));

    sat_unc_B(i,1) = nanmean(sid_unc(IDX_B{1}));
    sat_unc_B(i,2) = nanmedian(sid_unc(IDX_B{1}));
    sat_unc_B(i,3) = nanstd(sid_unc(IDX_B{1}));

    sat_D(i,1) = nanmean(sid(IDX_D{1}));
    sat_D(i,2) = nanmedian(sid(IDX_D{1}));
    sat_D(i,3) = nanstd(sid(IDX_D{1}));

    sat_unc_D(i,1) = nanmean(sid_unc(IDX_D{1}));
    sat_unc_D(i,2) = nanmedian(sid_unc(IDX_D{1}));
    sat_unc_D(i,3) = nanstd(sid_unc(IDX_D{1}));

    %% Reference ice draft statistics at mooring locations

    % Valid mooring dates within time interval
    idt_A = isbetween(t_full_A,t_lim(1),t_lim(2));
    idt_B = isbetween(t_full_B,t_lim(1),t_lim(2));
    idt_D = isbetween(t_full_D,t_lim(1),t_lim(2));
    
    % Calculate stats from mean daily ice draft at mooring
    ref_A(i,1) = nanmean(IDS_full_A(idt_A,2));
    ref_A(i,2) = nanmedian(IDS_full_A(idt_A,2));
    ref_A(i,3) = nanstd(IDS_full_A(idt_A,2));
    
    ref_B(i,1) = nanmean(IDS_full_B(idt_B,2));
    ref_B(i,2) = nanmedian(IDS_full_B(idt_B,2));
    ref_B(i,3) = nanstd(IDS_full_B(idt_B,2));
    
    ref_D(i,1) = nanmean(IDS_full_D(idt_D,2));
    ref_D(i,2) = nanmedian(IDS_full_D(idt_D,2));
    ref_D(i,3) = nanstd(IDS_full_D(idt_D,2));


end


%% Climatologies

tvec_clim = uniquetol(tvec_sit(:,2:3),1/max(tvec_sit(:,3)),'highest','Byrows',true);
[~,LocB] = ismembertol(tvec_sit(:,2:3),tvec_clim,1/max(tvec_sit(:,3)),'Byrows',true);

sat_clim_A = accumarray(LocB,sat_A(:,1),size(tvec_clim(:,1)),@nanmean);
sat_clim_B = accumarray(LocB,sat_B(:,1),size(tvec_clim(:,1)),@nanmean);
sat_clim_D = accumarray(LocB,sat_D(:,1),size(tvec_clim(:,1)),@nanmean);

ref_clim_A = accumarray(LocB,ref_A(:,1),size(tvec_clim(:,1)),@nanmean);
ref_clim_B = accumarray(LocB,ref_B(:,1),size(tvec_clim(:,1)),@nanmean);
ref_clim_D = accumarray(LocB,ref_D(:,1),size(tvec_clim(:,1)),@nanmean);


%% Calculate anomalies

sat_anom_A = sat_A(:,1) - sat_clim_A(LocB);
sat_anom_B = sat_B(:,1) - sat_clim_B(LocB);
sat_anom_D = sat_D(:,1) - sat_clim_D(LocB);

ref_anom_A = ref_A(:,1) - ref_clim_A(LocB);
ref_anom_B = ref_B(:,1) - ref_clim_B(LocB);
ref_anom_D = ref_D(:,1) - ref_clim_D(LocB);


%% Calculate lag time in days of highest cross-correlation

% Positive lag = satellite preceding ref time series
% Negative lag = satellite lagging behind ref time series

warning('off','all')

% Upsample sat data to daily with spline interpolation (to retain SID
% oscillations from coarser time series)
sat_A_up = interp1(datenum(tv   ec_sit),sat_A(:,1),datenum(t_full_A),'spline',NaN);
sat_B_up = interp1(datenum(tvec_sit),sat_B(:,1),datenum(t_full_B),'spline',NaN);
sat_D_up = interp1(datenum(tvec_sit),sat_D(:,1),datenum(t_full_D),'spline',NaN);

% Cross-correlation coefficients
idn = ~isnan(IDS_full_A(:,2)) & ~isnan(sat_A_up);
[xc,lags] = xcorr(IDS_full_A(idn,2),sat_A_up(idn),'normalized');
lag_max_xc_A = lags(xc == max(xc));

idn = ~isnan(IDS_full_B(:,2)) & ~isnan(sat_B_up);
[xc,lags] = xcorr(IDS_full_B(idn,2),sat_B_up(idn),'normalized');
lag_max_xc_B = lags(xc == max(xc));

idn = ~isnan(IDS_full_D(:,2)) & ~isnan(sat_D_up);
[xc,lags] = xcorr(IDS_full_D(idn,2),sat_D_up(idn),'normalized');
lag_max_xc_D = lags(xc == max(xc));

warning('on','all')

%% Stats

stats_bgep = [];

% On raw time series
idn = ~isnan(sat_A(:,1)) & ~isnan(ref_A(:,1));
stats_bgep.N_A = sum(idn);
r_A = corrcoef(sat_A(idn,1),ref_A(idn,1));
stats_bgep.r_A = r_A(1,2);
stats_bgep.bias_A = nanmean(sat_A(idn,1) - ref_A(idn,1));
stats_bgep.std_A = std(sat_A(idn,1) - ref_A(idn,1));
stats_bgep.rmse_A = rmse(sat_A(idn,1),ref_A(idn,1));
stats_bgep.se_A = std(sat_A(idn,1) - ref_A(idn,1))/sqrt(length(sat_A(idn,1)));       
p_A = polyfit(ref_A(idn,1),sat_A(idn,1),1);
stats_bgep.slope_A = p_A(1);
stats_bgep.lag_max_xc_A = lag_max_xc_A;

idn = ~isnan(sat_B(:,1)) & ~isnan(ref_B(:,1));
stats_bgep.N_B = sum(idn);
r_B = corrcoef(sat_B(idn,1),ref_B(idn,1));
stats_bgep.r_B = r_B(1,2);
stats_bgep.bias_B = nanmean(sat_B(idn,1) - ref_B(idn,1));
stats_bgep.std_B = std(sat_B(idn,1) - ref_B(idn,1));
stats_bgep.rmse_B = rmse(sat_B(idn,1),ref_B(idn,1));
stats_bgep.se_B = std(sat_B(idn,1) - ref_B(idn,1))/sqrt(length(sat_B(idn,1)));       
p_B = polyfit(ref_B(idn,1),sat_B(idn,1),1);
stats_bgep.slope_B = p_B(1);
stats_bgep.lag_max_xc_B = lag_max_xc_B;

idn = ~isnan(sat_D(:,1)) & ~isnan(ref_D(:,1));
stats_bgep.N_D = sum(idn);
r_D = corrcoef(sat_D(idn,1),ref_D(idn,1));
stats_bgep.r_D = r_D(1,2);
stats_bgep.bias_D = nanmean(sat_D(idn,1) - ref_D(idn,1));
stats_bgep.std_D = std(sat_D(idn,1) - ref_D(idn,1));
stats_bgep.rmse_D = rmse(sat_D(idn,1),ref_D(idn,1));
stats_bgep.se_D = std(sat_D(idn,1) - ref_D(idn,1))/sqrt(length(sat_D(idn,1)));       
p_D = polyfit(ref_D(idn,1),sat_D(idn,1),1);
stats_bgep.slope_D = p_D(1);
stats_bgep.lag_max_xc_D = lag_max_xc_D;

% On anomalies
idn = ~isnan(sat_anom_A(:,1)) & ~isnan(ref_anom_A(:,1));
stats_bgep.N_anom_A = sum(idn);
r_anom_A = corrcoef(sat_anom_A(idn,1),ref_anom_A(idn,1));
stats_bgep.r_anom_A = r_anom_A(1,2);
stats_bgep.bias_anom_A = nanmean(sat_anom_A(idn,1) - ref_anom_A(idn,1));
stats_bgep.std_anom_A = std(sat_anom_A(idn,1) - ref_anom_A(idn,1));
stats_bgep.rmse_anom_A = rmse(sat_anom_A(idn,1),ref_anom_A(idn,1));
stats_bgep.se_anom_A = std(sat_anom_A(idn,1) - ref_anom_A(idn,1))/sqrt(length(sat_anom_A(idn,1)));       
p_anom_A = polyfit(ref_anom_A(idn,1),sat_anom_A(idn,1),1);
stats_bgep.slope_anom_A = p_anom_A(1);

idn = ~isnan(sat_anom_B(:,1)) & ~isnan(ref_anom_B(:,1));
stats_bgep.N_anom_B = sum(idn);
r_anom_B = corrcoef(sat_anom_B(idn,1),ref_anom_B(idn,1));
stats_bgep.r_anom_B = r_anom_B(1,2);
stats_bgep.bias_anom_B = nanmean(sat_anom_B(idn,1) - ref_anom_B(idn,1));
stats_bgep.std_anom_B = std(sat_anom_B(idn,1) - ref_anom_B(idn,1));
stats_bgep.rmse_anom_B = rmse(sat_anom_B(idn,1),ref_anom_B(idn,1));
stats_bgep.se_anom_B = std(sat_anom_B(idn,1) - ref_anom_B(idn,1))/sqrt(length(sat_anom_B(idn,1)));       
p_anom_B = polyfit(ref_anom_B(idn,1),sat_anom_B(idn,1),1);
stats_bgep.slope_anom_B = p_anom_B(1);

idn = ~isnan(sat_anom_D(:,1)) & ~isnan(ref_anom_D(:,1));
stats_bgep.N_anom_D = sum(idn);
r_anom_D = corrcoef(sat_anom_D(idn,1),ref_anom_D(idn,1));
stats_bgep.r_anom_D = r_anom_D(1,2);
stats_bgep.bias_anom_D = nanmean(sat_anom_D(idn,1) - ref_anom_D(idn,1));
stats_bgep.std_anom_D = std(sat_anom_D(idn,1) - ref_anom_D(idn,1));
stats_bgep.rmse_anom_D = rmse(sat_anom_D(idn,1),ref_anom_D(idn,1));
stats_bgep.se_anom_D = std(sat_anom_D(idn,1) - ref_anom_D(idn,1))/sqrt(length(sat_anom_D(idn,1)));       
p_anom_D = polyfit(ref_anom_D(idn,1),sat_anom_D(idn,1),1);
stats_bgep.slope_anom_D = p_anom_D(1);


%% Export stats

outname = ['bgep_stats_v' version '.txt'];
    
outfile = fullfile(outfolder,outdir,outname);

if isfile(outfile)
    delete(outfile)
else
end

writetable(rows2vars(struct2table(stats_bgep),'VariableNamingRule','preserve'),outfile)

%% Plotting

if plot_opt==0
else
    
    idt = zeros(size(tvec_sit,1),1); idt(tvec_sit(:,2)>4 & tvec_sit(:,2)<10) = 1;
    
    %% Time series

    cmap = othercolor('Paired4',10);
    figure(20); clf(20);
    set(gcf,'Position',[1	41	1280	1317.60000000000],'renderer','painters')
    p = panel();
    p.margin = [20 10 10 10];
    p.pack(3,1);
    % p.select('all');
    p.de.margin = 10;
    
    p(1, 1).select();
    cla
    plot(datenum(t_full_A),IDS_full_A(:,2),'color',[0.7 0.7 0.7])
    hold on
    plot(datenum(t_full_A),smooth(IDS_full_A(:,2),31),'k','linewidth',1.5)
    scatter(datenum(tvec_sit(idt<1,:)),sat_A(idt<1,1),20,cmap(10,:),'filled')
    scatter(datenum(tvec_sit(idt>0,:)),sat_A(idt>0,1),20,cmap(4,:),'filled')
    hold off
    datetick('x','yyyy','keeplimits')
    text(datenum(t_full_A(1)) + 15,2.8,'BGEP Mooring A','fontsize',14,'fontweight','bold')
    box on
    grid on
    xlim([datenum(t_full_A(1)) datenum(t_full_A(end))])
    ylim([0 3])
    xticklabels('')
    % xlabel('Date')
    ylabel('Sea Ice Draft [m]')
    % legend('Mooring Ice Draft','31-Day Smoothing','CS2 Winter Ice Draft','CS2 Summer Ice Draft')
    set(gca,'fontsize',12)
    
    p(2, 1).select();
    cla
    plot(datenum(t_full_B),IDS_full_B(:,2),'color',[0.7 0.7 0.7])
    hold on
    plot(datenum(t_full_B),smooth(IDS_full_B(:,2),31),'k','linewidth',1.5)
    scatter(datenum(tvec_sit(idt<1,:)),sat_B(idt<1,1),20,cmap(10,:),'filled')
    scatter(datenum(tvec_sit(idt>0,:)),sat_B(idt>0,1),20,cmap(4,:),'filled')
    hold off
    datetick('x','yyyy','keeplimits')
    text(datenum(t_full_B(1)) + 15,2.8,'BGEP Mooring B','fontsize',14,'fontweight','bold')
    box on
    grid on
    xlim([datenum(t_full_B(1)) datenum(t_full_B(end))])
    ylim([0 3])
    xticklabels('')
    % xlabel('Date')
    ylabel('Sea Ice Draft [m]')
    % legend('Mooring Ice Draft','31-Day Smoothing','CS2 Winter Ice Draft','CS2 Summer Ice Draft')
    set(gca,'fontsize',12)
    
    p(3, 1).select();
    cla
    plot(datenum(t_full_D),IDS_full_D(:,2),'color',[0.7 0.7 0.7])
    hold on
    plot(datenum(t_full_D),smooth(IDS_full_D(:,2),31),'k','linewidth',1.5)
    scatter(datenum(tvec_sit(idt<1,:)),sat_D(idt<1,1),20,cmap(10,:),'filled')
    scatter(datenum(tvec_sit(idt>0,:)),sat_D(idt>0,1),20,cmap(4,:),'filled')
    hold off
    datetick('x','yyyy','keeplimits')
    text(datenum(t_full_D(1)) + 15,2.8,'BGEP Mooring D','fontsize',14,'fontweight','bold')
    box on
    grid on
    xlim([datenum(t_full_D(1)) datenum(t_full_D(end))])
    ylim([0 3])
    % xticklabels('')
    xlabel('Date')
    ylabel('Sea Ice Draft [m]')
    legend('Mooring Ice Draft','31-Day Smoothing','CS2 Winter Ice Draft','CS2 Summer Ice Draft','fontsize',12)
    set(gca,'fontsize',12)
    
    % export
    set(gcf,'color','w')
    outfig1name = ['bgep_time_series_v' version '.png'];
    saveas(gcf,fullfile(outfolder,outdir,outfig1name),'png')

    
    %% Anomaly scatterplot

    figure(30); clf(30);
    set(gcf,'Position',[16.2000000000000	353	1485.60000000000	420],'renderer','painters')
    p = panel();
    p.margin = [20 20 20 10];
    p.de.margin = 5;
    p.pack(1,3);
    
    p(1, 1).select();
    hold on
    idn = ~isnan(sat_anom_A(:,1)) & ~isnan(ref_anom_A(:,1));
    fancy_scatter(ref_anom_A(idn,1),sat_anom_A(idn,1),0.05,-1,1,-1,1)
    xlabel('Mooring Ice Draft Anomalies [m]')
    ylabel('Satellite Ice Draft Anomalies [m]')
    text(-0.98,0.9,'BGEP Mooring A','fontsize',14,'fontweight','bold')
    
    p(1, 2).select();
    hold on
    idn = ~isnan(sat_anom_B(:,1)) & ~isnan(ref_anom_B(:,1));
    fancy_scatter(ref_anom_B(idn,1),sat_anom_B(idn,1),0.05,-1,1,-1,1)
    xlabel('Mooring Ice Draft Anomalies [m]')
    % ylabel('Satellite Ice Draft Anomalies [m]')
    % yticklabels('')
    text(-0.98,0.9,'BGEP Mooring B','fontsize',14,'fontweight','bold')
    
    p(1, 3).select();
    hold on
    idn = ~isnan(sat_anom_D(:,1)) & ~isnan(ref_anom_D(:,1));
    fancy_scatter(ref_anom_D(idn,1),sat_anom_D(idn,1),0.05,-1,1,-1,1)
    xlabel('Mooring Ice Draft Anomalies [m]')
    % ylabel('Satellite Ice Draft Anomalies [m]')
    % yticklabels('')
    text(-0.98,0.9,'BGEP Mooring D','fontsize',14,'fontweight','bold')
    
    % export
    set(gcf,'color','w')
    outfig2name = ['bgep_anomaly_scatter_v' version '.png'];
    saveas(gcf,fullfile(outfolder,outdir,outfig2name),'png')

end

end