function expt = restoreOGSchedule(expt)
% RESTOREOGSCHEDULE
% Build a hazard-driven schedule with optional deceptive blocks.
% nb* expt for dwnstream:
%   expt.p_threat_true, expt.contexts, expt.block_types, expt.block_starts,
%   expt.block_ends, expt.blockLabel.

% Defaults (pulled from expt if present) 
if ~isfield(expt,'n_trials'),        expt.n_trials     = 1000;   end
if ~isfield(expt,'hazard'),          expt.hazard       = 1/80;   end
if ~isfield(expt,'p_H'),             expt.p_H          = 0.9;    end
if ~isfield(expt,'p_L'),             expt.p_L          = 0.1;    end
if ~isfield(expt,'minVolLen'),       expt.minVolLen    = 10;     end
if ~isfield(expt,'nStable0'),        expt.nStable0     = 240;    end
if ~isfield(expt,'p_deceptive'),     expt.p_deceptive  = 0.12;   end
if ~isfield(expt,'len_decept'),      expt.len_decept   = 60;     end
if ~isfield(expt,'schedule_seed'),    expt.schedule_seed = [];    end

opts = struct( ...
    'minVolLen',   expt.minVolLen, ...
    'nStable0',    expt.nStable0, ...
    'p_deceptive', expt.p_deceptive, ...
    'len_decept',  expt.len_decept, ...
    'seed',        expt.schedule_seed);

[p_true, block_types, block_starts, block_ends] = buildHazardSchedule( ...
    expt.n_trials, expt.hazard, expt.p_H, expt.p_L, opts);

% Write back (downstream expects)
expt.p_threat_true = p_true(:);
expt.contexts      = expt.p_threat_true;
expt.block_types   = block_types(:);
expt.block_starts  = block_starts(:);
expt.block_ends    = block_ends(:);

% Optional numeric labels 
lbl = zeros(numel(block_types),1);
for i = 1:numel(block_types)
    switch lower(block_types{i})
        case 'stable',    lbl(i) = 1;
        case 'volatile',  lbl(i) = 2;
        case 'deceptive', lbl(i) = 3;
        otherwise,        lbl(i) = 0;
    end
end
BlockLabel = zeros(expt.n_trials,1);
for i = 1:numel(block_starts)
    BlockLabel(block_starts(i):block_ends(i)) = lbl(i);
end
expt.blockLabel = BlockLabel;
end

function [p_true, block_types, block_starts, block_ends] = buildHazardSchedule(n_trials, h, p_H, p_L, opts)
% BUILDHAZARDSCHEDULE
% Volatile segments ~ geometric (h), alternating between p_H and p_L, with an
% initial stable block at 0.5 and some deceptive blocks.

if nargin < 5 || isempty(opts), opts = struct; end
minVolLen   = getfieldwithdefault(opts,'minVolLen',10);
nStable0    = getfieldwithdefault(opts,'nStable0',240);
p_deceptive = getfieldwithdefault(opts,'p_deceptive',0.0);
len_decept  = getfieldwithdefault(opts,'len_decept',60);
rng_seed    = getfieldwithdefault(opts,'seed',[]);
if ~isempty(rng_seed), rng(rng_seed); end

p_true       = zeros(n_trials,1);
block_types  = {};
block_starts = [];
block_ends   = [];

% 1) Initial stable block at 0.5
n0 = min(nStable0, n_trials);
p_true(1:n0) = 0.5;
block_types{end+1}  = 'stable';
block_starts(end+1) = 1;
block_ends(end+1)   = n0;

% 2) Volatile/deceptive sequence
t = n0 + 1;
curr_p = p_H;  % start volatile at p_H, then alternate to p_L and back
while t <= n_trials

    % deceptive block ?
    if (p_deceptive > 0) && (rand < p_deceptive)
        L = min(len_decept, n_trials - t + 1);
        p_true(t:t+L-1) = 0.5;
        block_types{end+1}  = 'deceptive';
        block_starts(end+1) = t;
        block_ends(end+1)   = t+L-1;
        t = t + L;
        continue
    end

    % volatile segment length ~ geometric(h); clamp to [minVolLen, remaining]
    L = geo_len(h);
    L = max(L, minVolLen);
    L = min(L, n_trials - t + 1);

    p_true(t:t+L-1) = curr_p;
    block_types{end+1}  = 'volatile';
    block_starts(end+1) = t;
    block_ends(end+1)   = t+L-1;

    % alternate H <-> L (tolerant compare)
    if abs(curr_p - p_H) < 1e-12
        curr_p = p_L;
    else
        curr_p = p_H;
    end

    t = t + L;
end

% enforce column vectors (defensive already true)
p_true       = p_true(:);
block_starts = block_starts(:);
block_ends   = block_ends(:);
end

function L = geo_len(h)
% Return geometric length ~ geom(h) + 1, even if geornd is unavailable
if exist('geornd','file') == 2
    L = geornd(h) + 1;
else
    % Inverse CDF for geometric + parameter h
    u = max(eps, 1 - rand);            % avoid log(0)
    L = ceil( log(u) / log(1 - h) );   % equivalent to geornd(h)+1
end
end

function v = getfieldwithdefault(S,f,default)
if isfield(S,f) && ~isempty(S.(f)), v = S.(f); else, v = default; end
end