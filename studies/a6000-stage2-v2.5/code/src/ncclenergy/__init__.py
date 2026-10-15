"""nccl-energy-frontier: latency/energy characterization of NCCL collectives.

Subpackages are split by concern:
    metadata      machine/software metadata capture
    nvml_sampler  background NVML telemetry sampler (power, util, temp, clocks)
    energy        energy-counter reads, power integration, idle measurement
    nccl_parse    parsing of nccl-tests stdout and NCCL_DEBUG logs
    orchestrate   experiment planning and execution
    stats         bootstrap statistics and derived metrics
    analyze       raw -> processed tables
    plots         figures
"""

__version__ = "0.1.0"
