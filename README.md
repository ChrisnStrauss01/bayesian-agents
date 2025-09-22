# Bayesian Categorisation Agents (For MSc Dissertation)

This repository reproduces the simulations reported in the MSc dissertation.  
Agents A–D are evaluated on a 2AFC “threat vs. non-threat” task with asymmetric payoffs.  
The oracle uses true base-rates with a symmetric decision threshold (τ = 0.5).  
Relative advantage in the code is stored under the field name `regret`.

## Quick start (MATLAB):
1. Clone this repo and open it in MATLAB.
2. Ensure these files are on the MATLAB path:
   `config_bayesian_categorisation.m`, `restoreOGSchedule.m`, `generateStimuli.m`,
   `initAgents.m`, `simulateAgents.m`, `agent_logic.m`, `computeMetrics.m`,
   `run_repro.m`, `seedList.m`.
## Run:
From MATLAB, run:

```matlab
run_repro('multi')   % full 50-seed run
run_repro('single')  % one-seed smoke test