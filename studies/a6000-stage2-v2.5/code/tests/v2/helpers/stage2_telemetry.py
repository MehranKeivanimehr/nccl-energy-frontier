"""Telemetry schema v3 with the fake NVML (CPU only). Prints one JSON line.
    python -W error stage2_telemetry.py <out_dir>"""
import csv
import gzip
import json
import sys
import time
from pathlib import Path

import pynvml
assert "fake_nvml" in pynvml.__file__
IDS = []
_fv = pynvml.nvmlDeviceGetFieldValues


def fv(h, ids):
    IDS.append(list(ids))
    return _fv(h, ids)


pynvml.nvmlDeviceGetFieldValues = fv
from ncclenergy.v2.telemetry import FIELDS, FIELDS_V3_EXTRA, Sampler  # noqa: E402

out = Path(sys.argv[1])
res = {}
for name, kw in (("default", {}), ("v3", {"schema_v3": True}), ("v3_tps", {"schema_v3": True, "total_power_samples": True})):
    IDS.clear()
    s = Sampler([4, 5], period_s=0.05, **kw).start()
    time.sleep(1.3)
    s.stop()
    s.write(out / f"{name}.csv.gz")
    with gzip.open(out / f"{name}.csv.gz", "rt") as f:
        rows = list(csv.reader(f))
    r = {"header": rows[0], "row_len": len(s.rows[0]), "array_cols": s.array().shape[1], "n": len(s.rows),
         "ids": IDS[0], "errors": len(s.errors)}
    if kw.get("schema_v3"):
        x = s.rows[-1]
        names = FIELDS + FIELDS_V3_EXTRA
        v = dict(zip(names, x))
        r.update(order=v["t_power_a_ns"] <= v["t_power_b_ns"] <= v["t_inst_a_ns"] <= v["t_inst_b_ns"],
                 field_ts_is_realtime_us=abs(v["inst_field_ts_us"] - time.time_ns() / 1000) < 5e6,
                 realtime_close=abs(v["t_realtime_ns"] - time.time_ns()) < 5e9, status=v["inst_status"],
                 latency=v["inst_latency_us"], mono_close=abs(v["t_mono_ns"] / 1e9 - x[0]) < 1.0)
    if kw.get("total_power_samples"):
        s.write_total_power_samples(out / "tps.csv.gz")
        r.update(tps_rows=len(s.tps_rows), tps_errors=len(s.tps_errors),
                 tps_ts_increasing=all(b[3] > a[3] for a, b in zip(s.tps_rows, s.tps_rows[1:]) if a[2] == b[2]))
    res[name] = r
print(json.dumps(res))
