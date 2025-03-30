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

% Comment out, if you not want to convert det datasets again
processMatFiles("/MATLAB Drive/testing/data")


% Check of new mat file

vars = whos('-file', '/MATLAB Drive/testing/data/Vuls16a_dailyn.mat');
disp({vars.name}); 
data = load("Vuls12a_dailyn.mat");
disp(data.dates);
disp(data.IDS);


