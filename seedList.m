function seeds = seedList()
% SEEDLIST list of RNG seeds used for multi-seed reproduction.

thisDir = fileparts(mfilename('fullpath'));
txt = fullfile(thisDir,'seeds.txt');

if exist(txt,'file') == 2
    s = fileread(txt);
    nums = regexp(s,'-?\d+','match');
    seeds = str2double(nums(:))';
    seeds = seeds(~isnan(seeds));
    if isempty(seeds)
        warning('seedList:emptyText','seeds.txt found but empty; defaulting to 1:50');
        seeds = 1:50;
    end
else
    seeds = 1:50;
end
end