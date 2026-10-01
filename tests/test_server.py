from pathlib import Path

from harness.server import build_cmd, free_port
from harness.spec import CandidateSpec


def test_build_cmd_pins_single_slot_and_disables_fit():
    spec = CandidateSpec(base_model="m.gguf", quant="Q4_K_M", kv_type_k="q8_0", kv_type_v="q8_0",
                          flash_attn=True)
    cmd = build_cmd(Path("models/cache/x.gguf"), spec, n_ctx=32768, port=8899, bin_dir="third_party/llama.cpp/build/bin")

    def flag(name):
        return cmd[cmd.index(name) + 1]

    # Regression coverage for the two bugs this module exists to avoid:
    # a server that silently shrinks context (--fit on by default) or splits
    # it across slots (-np auto) would invalidate exactly what gets measured.
    assert flag("-np") == "1"
    assert flag("--fit") == "off"
    assert flag("-c") == "32768"
    assert flag("-ctk") == "q8_0"
    assert flag("-ctv") == "q8_0"
    assert flag("-fa") == "on"
    assert "--jinja" in cmd


def test_build_cmd_reflects_flash_attn_off():
    spec = CandidateSpec(base_model="m.gguf", quant="Q4_K_M")
    cmd = build_cmd(Path("m.gguf"), spec, n_ctx=4096, port=8899)
    assert cmd[cmd.index("-fa") + 1] == "off"


def test_free_port_returns_a_usable_port():
    port = free_port()
    assert 1024 < port < 65536
