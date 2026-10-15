# Sources and what they support

External sources below were checked on 2026-10-14. They define APIs and explain relevant measurement/statistical principles. The study's numerical effects come from its own retained records and calculations; no paper independently validates the local +1.6454% estimate or its physical accuracy. Documentation linked as `latest` may change and is not a substitute for the pinned experimental implementation. [Stage 2 (2026/10), [raw sessions](../code/data/stage2/v2.5-heldout), [numerical results](../prep/heldout_final/contrasts.tsv), [native patch](../code/patches/nccl-tests-afd59ab-v2.5-stage2.patch)]

| Primary source | Supported principle | What it does not establish here |
|---|---|---|
| NVIDIA NVML, Device Queries | Energy counter API, units and reported scope | A6000 accuracy, a guaranteed 100 ms cadence, or physical reset/discard on a read |
| NVIDIA NCCL stream semantics and CUDA event API | Asynchronous enqueue, synchronization and event timing units | Whether this particular patch placed its boundaries correctly |
| Yang, Adámek and Armour, SC24 | Device-specific sensor timing, averaging/response and measurement characterization | The local cumulative-counter correction or a sub-percent error bound |
| Dauner et al., HotCarbon 2026 | Platform- and read-frequency-dependent reported energy, stale readings and counter/power disagreement | The local read-dose mechanism or universal superiority of either estimator |
| Kalibera and Jones, ISMM 2013 | Repetition levels and benchmark uncertainty | Independence of these same-evening sessions or a universal sufficient sample count |
| Lakens, 2017 | Equivalence testing, prespecified margins and distinction from nonsignificance | Physical accuracy, the scientific desirability of this project's margins or individual-reading bounds |

Source locations: NVML `nvmlDeviceGetTotalEnergyConsumption`; Yang et al. §§4–5; Dauner et al. §§3–4; Kalibera and Jones §3; Lakens Figure 1 and equivalence-bound discussion. These are a selective set of primary references, not an exhaustive systematic literature review. [NVIDIA (accessed 2026/10), [NVML](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html), [NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html), [CUDA](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html); Yang et al. (2024/12), [author manuscript](https://arxiv.org/html/2312.02741v3); Dauner et al. (2026/07), [paper](https://hotcarbon.org/assets/2026/paper-46.pdf); Kalibera and Jones (2013/06), [paper](https://kar.kent.ac.uk/33611/45/p63-kaliber.pdf); Lakens (2017/05), [paper](https://pure.tue.nl/ws/portalfiles/portal/80918653/lakeequi2017.pdf)]

## Reference list

[NVIDIA NVML, accessed 2026/10] NVIDIA. "Device Queries." NVML API Reference, `nvmlDeviceGetTotalEnergyConsumption`. [Official documentation](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html).

[NVIDIA NCCL, accessed 2026/10] NVIDIA. "CUDA Stream Semantics." NCCL User Guide. [Official documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html).

[NVIDIA CUDA, accessed 2026/10] NVIDIA. "Event Management." CUDA Runtime API. [Official documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html).

[Yang et al., 2024/11] Zeyu Yang, Karel Adámek and Wesley Armour. "Accurate and Convenient Energy Measurements for GPUs: A Detailed Study of NVIDIA GPU's Built-In Power Sensor." SC24, Article 22. DOI: 10.1109/SC41406.2024.00028. [Proceedings page](https://sc24.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap544.html). The authors' arXiv v3 (2024/12) is titled "Part-time Power Measurements: nvidia-smi's Lack of Attention" ([manuscript](https://arxiv.org/html/2312.02741v3)); it is not a second independent study.

[Dauner et al., 2026/07] Maximilian Dauner, Manuel Steinberg, Andreas Brunnert, Benedikt Schicker and Benedikt Zönnchen. "Evaluating the Influence of Measurement Frequency on Energy Readings Using Intel RAPL and NVIDIA NVML." HotCarbon 2026; ACM SIGENERGY Energy Informatics Review 6(2). [Workshop paper](https://hotcarbon.org/assets/2026/paper-46.pdf). Its evaluated GPU set is not the identical RTX A6000 deployment studied here; its external reference measures the whole platform.

[Kalibera and Jones, 2013/06] Tomas Kalibera and Richard Jones. "Rigorous Benchmarking in Reasonable Time." ISMM 2013, pp. 63–74. DOI: 10.1145/2464157.2464160. [University repository manuscript](https://kar.kent.ac.uk/33611/45/p63-kaliber.pdf).

[Lakens, 2017/05] Daniël Lakens. "Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses." Social Psychological and Personality Science 8(4), pp. 355–362. DOI: 10.1177/1948550617697177. [University-hosted final paper](https://pure.tue.nl/ws/portalfiles/portal/80918653/lakeequi2017.pdf).

## Local evidence map

| Files | Scientific role |
|---|---|
| [held-out protocol](../protocol/STAGE2_HELDOUT1_PROTOCOL.md), [calibration protocol](../protocol/STAGE2_CALIB1_PROTOCOL.md) | Prespecified scope, count, margin, readiness and stop rules |
| [acquisition source](../code/src/ncclenergy/v2/stage2_runner.py), [native patch](../code/patches/nccl-tests-afd59ab-v2.5-stage2.patch) | Actual boundary, counter-read and process implementation |
| [raw held-out session directories](../code/data/stage2/v2.5-heldout) | Per-launch raw counter and telemetry traces, profiler execution, correctness and process records |
| [calibration analysis](../prep/calib1/run/calibration_analysis.json) | Period/read-dose/readiness outcomes, including unbounded Q1 |
| [units](../prep/heldout_final/units.tsv), [exclusions](../prep/heldout_final/exclusions.tsv), [execution](../prep/heldout_final/execution.tsv) | Availability, launch acceptance, R and actual execution |
| [contrasts](../prep/heldout_final/contrasts.tsv), [thermal controls](../prep/heldout_final/thermal_control.tsv), [adequacy](../prep/heldout_final/adequacy.tsv) | Effects, uncertainty, thermal overrides and all retained verdicts |
| [public provenance](../provenance), [replay adapter](../scripts/reproduce_public.py) | Public-original mapping, declared omissions, hashes and offline derivative replay |

The presence of a source file or successful parse is not proof that every runtime path was validated. Arithmetic replay, real execution evidence, physical calibration and generalizability are reported separately. [Release (2026/10), [publication status](publication_status.md); Stage 2 (2026/10), [analysis limits](../code/src/ncclenergy/v2/stage2_heldout.py)]
