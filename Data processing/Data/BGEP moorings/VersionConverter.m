% Code for conversion of V7.3 mat files to V7 mat files

% 1) Make an function that reads the indepent .mat files from data folder
% 2) Extract the variables IDS and dates from the .mat file
% 3) save into a new .mat file with the same naming, but adds a big V in
%    the start of the file name.

function processMatFiles(dataFolder)
    % Get a list of all .mat files in the specified folder
    files = dir(fullfile(dataFolder, '*.mat'));

    % Loop through each file
    for i = 1:length(files)
        % Get full file path
        filePath = fullfile(dataFolder, files(i).name);
        
        % Load the .mat file
        data = load(filePath);

        % Check if the required variables exist
        if isfield(data, 'IDS') && isfield(data, 'dates')
            % Extract variables
            IDS = data.IDS;
            dates = data.dates;
            
            % Create new file name with 'V' prefix
            newFileName = ['V', files(i).name];
            newFilePath = fullfile(dataFolder, newFileName);

            % Save in MATLAB v7.2 format
            save(newFilePath, 'IDS', 'dates', '-v7');
        else
            fprintf('Skipping %s: Required variables not found.\n', files(i).name);
        end
    end

    fprintf('Processing complete.\n');
end


function processMatFiles_22(dataFolder)
    % Get a list of all .mat files in the specified folder
    files = dir(fullfile(dataFolder, '*.mat'));

    % Loop through each file
    for i = 1:length(files)
        % Get full file path
        filePath = fullfile(dataFolder, files(i).name);
        
        % Load the .mat file
        data = load(filePath);

        % Check if the required variables exist
        if isfield(data, 'IDS') && isfield(data, 'dates')
            % Extract variables
            IDS = data.IDS;
            dates = data.dates;
            
            % Convert dates to datetime format.
            if iscell(dates)
                dt = datetime(dates, 'InputFormat', 'dd-MMM-yyyy');
            elseif ischar(dates)
                dt = datetime(cellstr(dates), 'InputFormat', 'dd-MMM-yyyy');
            elseif isnumeric(dates)
                dt = datetime(dates, 'ConvertFrom', 'datenum');
            else
                warning('Unsupported date format in file %s. Skipping file.', files(i).name);
                continue;
            end
            
            % Filter out only October 2023 dates
            keepIdx = ~((month(dt) == 10) & (year(dt) == 2023));
            dt_keep = dt(keepIdx);
            IDS = IDS(keepIdx, :);
            
            % Convert the filtered datetime array back to the original format
            if ischar(dates)
                dates = char(datestr(dt_keep, 'dd-mmm-yyyy')); % Convert to character array
            else
                dates = cellstr(datestr(dt_keep, 'dd-mmm-yyyy')); % Convert to cell array of strings
            end
            
            % Create new file name with 'V' prefix
            newFileName = ['V', files(i).name];
            newFilePath = fullfile(dataFolder, newFileName);

            % Save in MATLAB v7.2 format
            save(newFilePath, 'IDS', 'dates', '-v7');
        else
            fprintf('Skipping %s: Required variables not found.\n', files(i).name);
        end
    end

    fprintf('Processing complete.\n');
end


% Comment out, if you not want to convert det datasets again
processMatFiles("/MATLAB Drive/testing/new_data")

% The other function is used 2022 as we dont have sat 
% data for oct 2023 from smos and cryos
processMatFiles_22("/MATLAB Drive/testing/new_data")

% Check of new mat file
data_conv = load("/MATLAB Drive/testing/new_data/Vuls22a_dailyn.mat");
disp({data_conv.name});
disp("converted data:");
disp(data_conv.dates);
disp(data.IDS);

data = load("/MATLAB Drive/testing/new_data/uls22a_dailyn.mat");
disp("org data:");
disp(data.dates);

% Display the shape of the variable
variableShape = size(data.IDS);
disp(['Shape of myVariable: ', num2str(variableShape)]);




