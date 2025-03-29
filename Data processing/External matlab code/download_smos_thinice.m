
%% Download SMOS Data %%

% Local data

folder = 'R:\IFT\EarthObservation\SatelliteAltimetry'; % Main working directory, EarthObsWin

smos_files = dir(fullfile(folder,"SMOS Thin Ice Thickness",'*.nc'));


% Add data from server

years = 2023:2025; %% MODIFY


server_name = "https://icdc.cen.uni-hamburg.de/thredds/fileServer/ftpthredds/smos_sea_ice_thickness/v3.3/nh/";

fname_pattern = 'SMOS_Icethickness_v3.3_north_';

dnums = (datenum(years(1),1,1):datenum(years(end),12,31))';
tvec = datevec(dnums);
dstr = datestr(dnums,'yyyymmdd');

for i = 1:length(dnums)
    
    dirname = fullfile(server_name,num2str(tvec(i,1)),num2str(tvec(i,2),'%02u'));
    
    fname = [fname_pattern dstr(i,:) '.nc'];

    idf = cellfun(@(x) strcmp(fname,x), {smos_files.name}');

    if sum(idf) > 0
    else
        
        url_name = fullfile(dirname,fname);
        
        try
            filename_out = websave(fullfile(folder,"SMOS Thin Ice Thickness",fname),url_name);
            fprintf([dstr(i,:) ' Downloaded \n']);
        catch
            fprintf([dstr(i,:) ' No file \n']);
        end

    end



end


% Remove html files
html_files = dir(fullfile(folder,"SMOS Thin Ice Thickness",'*.html'));

for i = 1:length(html_files)
    delete(fullfile(html_files(i).folder,html_files(i).name))
end



