"""CPU-only tests for the offline Energy-SLO recommender and tuner-config generator (v2.2 semantics)."""

import copy

import numpy as np
import pandas as pd

from ncclenergy.v2.recommender import Query, build_profiles, lobo_validate, recommend, tuner_conf

SCOPE = {"gpu_model": "NVIDIA RTX A6000", "driver_version": "580.178.04", "cuda_driver_version": 13000,
         "nccl_version": "2.27.5+cuda12.9", "nccl_version_code": 22705, "dtype": "float32", "in_place": 0,
         "pci_bus_ids": "00000000:81:00.0,00000000:A1:00.0", "ops_by_coll": {"all_reduce": "sum"}}
META = {"P": {"fingerprint": "fp1", "env": "", "nranks": 2, "nnodes": 1, **SCOPE}}


def _units(sizes=(1024, 2048, 4096), better_at=(2048, 4096), coll="all_gather"):
    rng = np.random.default_rng(0)
    rows = []
    for b in range(1, 6):
        for s in sizes:
            for cfg, act, e, t in (("auto", "RING/LL/1", 1.00, 1.00), ("Tree/LL", "TREE/LL/1", 0.90, 1.02),
                                   ("Ring/Simple", "RING/SIMPLE/1", 0.85, 1.30)):
                if cfg == "Tree/LL" and s not in better_at:
                    e = 1.05
                je = e * (1 + rng.normal(0, 0.005))
                tt = t * (1 + rng.normal(0, 0.005))
                rows.append(dict(block=b, placement="P", coll=coll, size_bytes=s, config=cfg, actual=act,
                                 J_per_coll=je, t_gpu_s=tt, logE=np.log(je), logT=np.log(tt)))
    return pd.DataFrame(rows)


def _q(size, **kw):
    base = dict(collective="all_gather", message_size=size, number_of_ranks=2, number_of_nodes=1,
                topology_fingerprint="fp1", placement_env="", nccl_version="2.27.5+cuda12.9",
                cuda_driver_version=13000, gpu_model="NVIDIA RTX A6000", driver_version="580.178.04",
                dtype="float32", op=None, in_place=0)
    base.update(kw)
    return Query(**base)


def test_recommend_respects_slo_and_evidence():
    prof = build_profiles(_units(), META)
    r = recommend(prof, _q(2048))
    assert r["status"] == "RECOMMEND" and r["recommendation"] == "Tree/LL"     # Ring/Simple saves more but breaks the SLO
    assert r["E_ratio_ci"][1] < 1 and r["T_ratio_hi"] <= 1.05 and r["n_blocks"] == 5
    assert recommend(prof, _q(1024))["status"] == "FALLBACK"                     # nothing better than auto at 1 KiB
    r2 = recommend(prof, _q(2048, objective="max_bytes_per_joule"))
    assert r2["recommendation"] == "Ring/Simple"                                 # no SLO for bytes/J


def test_incompatible_or_unknown_query_metadata_falls_back_to_auto():
    prof = build_profiles(_units(), META)
    for q in (_q(2048, topology_fingerprint="other"), _q(2048, placement_env="NCCL_P2P_DISABLE=1"),
              _q(2048, nccl_version="2.28.3+cuda12.9"), _q(2048, number_of_ranks=4), _q(2048, number_of_nodes=2),
              _q(2048, cuda_driver_version=12080), _q(2048, gpu_model="NVIDIA GeForce GTX 1080 Ti"),
              _q(2048, driver_version="550.144.03"), _q(2048, dtype="bfloat16"), _q(2048, in_place=1),
              _q(512), _q(8192)):
        r = recommend(prof, q)
        assert r["status"] == "INSUFFICIENT_EVIDENCE", (q, r)
    for field in ("nccl_version", "cuda_driver_version", "gpu_model", "driver_version", "dtype", "in_place"):
        r = recommend(prof, _q(2048, **{field: None}))                            # unknown query metadata
        assert r["status"] == "INSUFFICIENT_EVIDENCE" and any("QUERY_METADATA_MISSING" in p for p in r["scope_problems"])


def test_missing_legacy_profile_metadata_is_not_filled_in():
    for field in ("driver_version", "cuda_driver_version", "gpu_model", "nccl_version", "dtype"):
        meta = copy.deepcopy(META)
        meta["P"][field] = None                                                  # not recorded -> stays None
        prof = build_profiles(_units(), meta)
        r = recommend(prof, _q(2048))
        assert r["status"] == "INSUFFICIENT_EVIDENCE"
        assert f"PROFILE_METADATA_MISSING:{field}" in r["scope_problems"]


def test_reduction_op_is_part_of_the_scope():
    v = _units(coll="all_reduce")
    prof = build_profiles(v, META)
    ok = recommend(prof, _q(2048, collective="all_reduce", op="sum"))
    assert ok["status"] == "RECOMMEND"
    assert recommend(prof, _q(2048, collective="all_reduce", op="max"))["status"] == "INSUFFICIENT_EVIDENCE"
    assert recommend(prof, _q(2048, collective="all_reduce", op=None))["status"] == "INSUFFICIENT_EVIDENCE"
    meta = copy.deepcopy(META)
    meta["P"]["ops_by_coll"] = {}                                                # op not recorded for the profile
    r = recommend(build_profiles(v, meta), _q(2048, collective="all_reduce", op="sum"))
    assert r["status"] == "INSUFFICIENT_EVIDENCE" and "PROFILE_METADATA_MISSING:op[all_reduce]" in r["scope_problems"]


def test_interpolation_disabled_by_default_and_experimental_opt_in_is_separate():
    prof = build_profiles(_units(), META)
    r = recommend(prof, _q(3000))                       # between two measured sizes that both recommend Tree/LL
    assert r["status"] == "INSUFFICIENT_EVIDENCE" and "interpolation" in r["reason"]
    x = recommend(prof, _q(3000), allow_interpolation=True)
    assert x["status"] == "EXPERIMENTAL_UNVALIDATED_INTERPOLATION" and x["candidate"] == "Tree/LL"
    assert x["E_ratio_vs_auto"] is None and x["E_ratio_ci"] is None              # no copied endpoint ratio/interval
    assert set(x["endpoints"]) == {"2048", "4096"}
    y = recommend(prof, _q(1500), allow_interpolation=True)                     # endpoints disagree
    assert y["candidate"] is None and y["status"] != "RECOMMEND"


def test_lobo_validation_scores_heldout_blocks():
    v = _units()
    res = lobo_validate(v, META)
    assert set(res.heldout_block) == {1, 2, 3, 4, 5} and len(res) == 15
    rec = res[res.size_bytes == 2048]
    assert (rec.recommendation == "Tree/LL").all() and (rec.measured_saving_vs_auto > 0.05).all()
    assert not rec.slo_violated.any() and (rec.regret.abs() < 0.03).all()


def test_tuner_conf_exact_sizes_with_scope_header():
    prof = build_profiles(_units(), META)
    lines = tuner_conf(prof, "P", META)
    rules = [l for l in lines if not l.startswith("#")]
    # one rule per measured size (total bytes, as NCCL passes them to the tuner); no range covers unmeasured sizes
    assert rules == ["allgather,2048,2048,tree,ll,-1,1,2,-1,-1", "allgather,4096,4096,tree,ll,-1,1,2,-1,-1"]
    scope = dict(l[len("#@scope "):].split("=", 1) for l in lines if l.startswith("#@scope "))
    assert scope["method"] == "v2.2" and scope["allow_ranges"] == "0" and scope["nccl_version_code"] == "22705"
    assert scope["gpu_model"] == "NVIDIA RTX A6000" and scope["placement_env"] == "none"
    assert scope["declared_scope"].startswith("dtype=float32;")
    exp = [l for l in tuner_conf(prof, "P", META, experimental_ranges=True) if not l.startswith("#")]
    assert exp == ["allgather,2048,4096,tree,ll,-1,1,2,-1,-1"]
    assert "#@scope allow_ranges=1" in tuner_conf(prof, "P", META, experimental_ranges=True)


def test_no_rules_for_unmeasured_gaps():
    v = _units(sizes=(1024, 65536, 131072), better_at=(1024, 65536, 131072))
    prof = build_profiles(v, META)
    assert recommend(prof, _q(4096))["status"] == "INSUFFICIENT_EVIDENCE"
    assert recommend(prof, _q(98304))["status"] == "INSUFFICIENT_EVIDENCE"      # v2.2: no interpolation by default
    rules = [l for l in tuner_conf(prof, "P", META) if not l.startswith("#")]
    assert rules == ["allgather,1024,1024,tree,ll,-1,1,2,-1,-1", "allgather,65536,65536,tree,ll,-1,1,2,-1,-1",
                     "allgather,131072,131072,tree,ll,-1,1,2,-1,-1"]
