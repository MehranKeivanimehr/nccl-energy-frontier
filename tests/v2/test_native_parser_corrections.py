"""Real C plugin with fake NVML: malformed input must leave NCCL defaults."""
import importlib.util
from pathlib import Path
import tempfile


def test_native_rejects_extra_empty_duplicate_and_truncated_fields():
    path = Path(__file__).with_name("test_v2_tuner_guard.py")
    spec = importlib.util.spec_from_file_location("guard_test_helper", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        helper._build(tmp)
        for rule in (helper.RULES[0] + ",extra",
                     helper.RULES[0].replace("ring,ll", "ring,,ll"),
                     helper.RULES[0].replace("524288,524288", ",524288"),
                     helper.RULES[0] + "x" * 400):
            out, _ = helper._run(tmp, rules=[rule])
            assert out.startswith("NO_RULE"), rule
        scope = dict(helper.SCOPE, pci_bus_ids="00000000:81:00.0,00000000:81:00.0")
        out, _ = helper._run(tmp, scope=scope)
        assert out.startswith("NO_RULE")
        out, _ = helper._run(tmp, extra_lines=["#@scope nranks=2"])
        assert out.startswith("NO_RULE")


def test_duplicate_visible_gpu_uuid_is_rejected():
    path = Path(__file__).with_name("test_v2_tuner_guard.py")
    spec = importlib.util.spec_from_file_location("guard_test_helper2", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        helper._build(tmp)
        out, _ = helper._run(tmp, env={"CUDA_VISIBLE_DEVICES": "GPU-aaa,GPU-aaa"})
        assert out.startswith("NO_RULE")
