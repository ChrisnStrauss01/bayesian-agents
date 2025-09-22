function run_repro(mode)
% RUN_REPRO — minimal, loud entrypoint.
% Usage:
%   run_repro            % single-seed smoke test (default)
%   run_repro('single')  % same as above
%   run_repro('multi')   % iterate over seedList() if present

if nargin < 1 || isempty(mode), mode = 'single'; end
fprintf('>> run_repro(%s) starting...\n', string(mode));

% Make results dir
if ~exist('results','dir'), mkdir results; fprintf('  created ./results\n'); end

switch lower(mode)
    case 'single'
        seeds = 12345;    % quick smoke test
    case 'multi'
        if exist('seedList.m','file') == 2
            seeds = seedList();
        else
            error('run_repro:NoSeedList', ...
                'mode="multi" but seedList.m not found on path.');
        end
    otherwise
        error('run_repro:BadMode','Unknown mode: %s', mode);
end

for s = seeds(:).'
    fprintf('\n[seed %d]\n', s);
    rng(s);

    % 1) Config + schedule
    ex = config_bayesian_categorisation();
    ex.schedule_seed = s;
    ex = restoreOGSchedule(ex);

    % 2) Stimuli (+ contexts)
    [stim, ytrue, yperc, contexts] = generateStimuli(ex);
    ex.contexts = contexts;

    % 3) Init + simulate
    agents = initAgents(ex, stim);
    [results, agents] = simulateAgents(ex, agents, stim, ytrue, yperc, []);

    % 4) Metrics
    metrics = computeMetrics(results, ytrue, ex, stim);

    % 5) Save per-seed MAT
    outFile = fullfile('results', sprintf('metrics_seed_%d.mat', s));
    save(outFile, 'metrics', 'ex', 's');
    fprintf('  saved %s\n', outFile);

    % 6) Console summary (short)
    names = fieldnames(metrics);
    for i = 1:numel(names)
        nm = names{i};
        fprintf('  %s: Acc=%.3f  Fit=%.1f  RA=%.1f  CSI=%.3f\n', ...
            nm, metrics.(nm).accuracy, ...
            metrics.(nm).fitness_payoff, ...
            metrics.(nm).regret, ...
            metrics.(nm).csi);
    end
end
% 7) (Optional) write a tiny CSV summary for last run
try
    csvPath = fullfile('results','metrics_summary.csv');
    rows = {'seed','agent','accuracy','fitness','relative_adv','csi','auc'};
    for i = 1:numel(names)
        nm = names{i};
        rows(end+1,:) = {s, nm, ...
            metrics.(nm).accuracy, ...
            metrics.(nm).fitness_payoff, ...
            metrics.(nm).regret, ...
            metrics.(nm).csi, ...
            metrics.(nm).roc_auc}; %#ok<AGROW>
    end
    writecell(rows, csvPath);
    fprintf('  wrote summary CSV: %s\n', csvPath);
catch ME
    warning('run_repro:CSVwrite', '%s', ME.message);
end