#!/usr/bin/env python3
"""Install DFlash2 onto the glm53-flash vLLM image (idempotent).

PORT NOTE (vLLM 83252ea89, 2026-09-09)
--------------------------------------
Upstream absorbed DFlash2 between Mia's day-0 base
(0.1.dev20051+g487ecf187) and this pin.  At 83252ea89 the vLLM tree already
ships, natively:

  * ``model_executor/models/qwen3_dflash2.py``
  * ``v1/worker/gpu/spec_decode/dflash2/{__init__,speculator}.py``
  * the ``DFlash2DraftModel`` registry entry
  * the ``method == "dflash"`` -> DFlash2Speculator dispatch
  * the ``decoder_layer_cls`` / ``model_cls`` indirection in qwen3_dflash.py

The two vendored sources in this overlay (``qwen3_dflash2.py``,
``dflash2_speculator.py``) exist, by their own docstrings, only to work
around two APIs that were **missing from the old base image**:
``LogitsProcessor.get_top_k_tokens`` and an exported
``gumbel_noised_argmax``.  Both are present at this pin, and upstream's
copies are strictly newer (they carry ``IS_DRAFTING=True`` in the Gumbel
draw and a ``draft_logits_spec`` hook that the vendored copies predate).

Therefore the file installs are now **capability-gated**: if the shimmed-for
APIs are present the upstream implementations are kept and the vendored
copies are skipped; if either API is missing the vendored copies are
installed exactly as before.  Anything else raises.  Nothing is silently
no-opped.
"""

from __future__ import annotations

from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
OPT = Path("/opt/glm53")


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text and old not in text:
        return
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{path}: expected one patch target, found {n}: {old!r}")
    path.write_text(text.replace(old, new))


def require_once(path: Path, snippet: str, why: str) -> None:
    """Fail-closed assertion that an already-upstream construct is present.

    Used where upstream has absorbed an edit this overlay used to make: the
    edit becomes a no-op, but the guarantee it encoded must still be
    checked, or a future upstream revert would silently drop the feature.
    """
    text = path.read_text()
    n = text.count(snippet)
    if n != 1:
        raise SystemExit(
            f"{path}: expected exactly one {why} (already-upstream check), "
            f"found {n}: {snippet!r}"
        )


def main() -> None:
    src_model = OPT / "qwen3_dflash2.py"
    src_spec = OPT / "dflash2_speculator.py"
    dst_model = SITE / "model_executor/models/qwen3_dflash2.py"
    dst_dir = SITE / "v1/worker/gpu/spec_decode/dflash2"
    dst_spec = dst_dir / "speculator.py"
    dst_init = dst_dir / "__init__.py"

    # PORT 83252ea89: capability probe -- see module docstring.  The vendored
    # sources are shims for two APIs the old base image lacked.  Probe the
    # target tree for those APIs rather than the vLLM version string.
    lp = SITE / "model_executor/layers/logits_processor.py"
    gumbel = SITE / "v1/worker/gpu/sample/gumbel.py"
    if not lp.is_file() or not gumbel.is_file():
        raise SystemExit(
            f"{lp} / {gumbel}: cannot probe DFlash2 shim prerequisites; "
            "the vLLM layout has drifted -- re-derive patch_dflash2.py."
        )
    has_top_k = "def get_top_k_tokens(" in lp.read_text()
    gumbel_src = gumbel.read_text()
    has_gumbel = (
        "def gumbel_noised_argmax(" in gumbel_src and "IS_DRAFTING" in gumbel_src
    )
    upstream_has_dflash2 = dst_model.is_file() and dst_spec.is_file()

    if has_top_k and has_gumbel and upstream_has_dflash2:
        # Upstream's own DFlash2 is newer than the vendored shims (it carries
        # IS_DRAFTING=True in the Gumbel draw and a draft_logits_spec hook).
        # Overwriting it would be a regression, so keep upstream's.
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not dst_init.exists():
            dst_init.write_text("# SPDX-License-Identifier: Apache-2.0\n")
        print(
            "dflash2: upstream ships DFlash2 natively and both shimmed APIs "
            "(get_top_k_tokens, gumbel_noised_argmax/IS_DRAFTING) are present "
            "-- keeping upstream qwen3_dflash2.py + dflash2/speculator.py, "
            "vendored overlay copies NOT installed"
        )
    elif not has_top_k or not has_gumbel:
        # Old-base behaviour: the shims are load-bearing, install them.
        dst_model.write_text(src_model.read_text())
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_spec.write_text(src_spec.read_text())
        if not dst_init.exists():
            dst_init.write_text("# SPDX-License-Identifier: Apache-2.0\n")
        print(
            "dflash2: shimmed APIs missing "
            f"(get_top_k_tokens={has_top_k} gumbel_noised_argmax={has_gumbel}) "
            "-- installed vendored qwen3_dflash2.py + dflash2_speculator.py"
        )
    else:
        raise SystemExit(
            "dflash2: both shimmed APIs are present but upstream does not ship "
            f"{dst_model} / {dst_spec}. Refusing to install the vendored shims "
            "against a tree whose APIs they were written to work around -- "
            "re-derive patch_dflash2.py against this tree."
        )

    qwen = SITE / "model_executor/models/qwen3_dflash.py"
    replace_once(
        qwen,
        'def _dflash_layer_causal(config: Qwen3Config, layer_idx: int) -> bool:\n'
        '    """``dflash_config.causal`` overrides all layers; else only SWA layers causal."""\n'
        '    override = (getattr(config, "dflash_config", None) or {}).get("causal")\n',
        'def _dflash_layer_causal(config: Qwen3Config, layer_idx: int) -> bool:\n'
        # PORT 83252ea89: upstream now ships this exact behaviour under its own
        # docstring. `new` below is upstream's text verbatim so replace_once()
        # takes its idempotent short-circuit (new present, old absent -> return).
        # Fail-closed is intact: if upstream ever drops the ``is_causal``
        # short-circuit, neither string is found and replace_once raises.
        '    """Resolve explicit causality before falling back to legacy layer defaults."""\n'
        '    is_causal = getattr(config, "is_causal", None)\n'
        "    if is_causal is not None:\n"
        "        return bool(is_causal)\n"
        '    override = (getattr(config, "dflash_config", None) or {}).get("causal")\n',
    )
    replace_once(
        qwen,
        "@support_torch_compile\n"
        "class DFlashQwen3Model(nn.Module):\n"
        "    hf_to_vllm_mapper = WeightsMapper(\n",
        # PORT 83252ea89: upstream ships decoder_layer_cls, followed by a
        # blank line. Match its text verbatim so replace_once() short-circuits.
        "@support_torch_compile\n"
        "class DFlashQwen3Model(nn.Module):\n"
        "    decoder_layer_cls = DFlashQwen3DecoderLayer\n"
        "\n"
        "    hf_to_vllm_mapper = WeightsMapper(\n",
    )
    replace_once(
        qwen,
        "        self.layers = nn.ModuleList(\n"
        "            [\n"
        "                DFlashQwen3DecoderLayer(\n",
        "        self.layers = nn.ModuleList(\n"
        "            [\n"
        "                self.decoder_layer_cls(\n",
    )
    replace_once(
        qwen,
        "class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):\n"
        "    def __init__(self, *, vllm_config: VllmConfig, prefix: str = \"\"):\n",
        "class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):\n"
        "    model_cls = DFlashQwen3Model\n"
        "\n"
        "    def __init__(self, *, vllm_config: VllmConfig, prefix: str = \"\"):\n",
    )
    replace_once(
        qwen,
        "        self.model = DFlashQwen3Model(\n"
        "            vllm_config=vllm_config,\n"
        '            prefix=maybe_prefix(prefix, "model"),\n'
        "            start_layer_id=target_layer_num,\n"
        "        )\n",
        "        self.model = self.model_cls(\n"
        "            vllm_config=vllm_config,\n"
        '            prefix=maybe_prefix(prefix, "model"),\n'
        "            start_layer_id=target_layer_num,\n"
        "        )\n",
    )

    registry = SITE / "model_executor/models/registry.py"
    # PORT 83252ea89: upstream registers DFlash2DraftModel itself. The old
    # replace_once() would have duplicated the line here (its `new` contains
    # its `old`, so the idempotency short-circuit can never fire). Downgraded
    # to a fail-closed presence assertion.
    require_once(
        registry,
        '    "DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM"),\n',
        "DFlash2DraftModel registry entry",
    )

    dflash_utils = SITE / "v1/worker/gpu/spec_decode/dflash/utils.py"
    replace_once(
        dflash_utils,
        "    speculative_config = vllm_config.speculative_config\n"
        "    assert speculative_config is not None\n"
        "    draft_model_config = speculative_config.draft_model_config\n"
        "    # Select an attention backend that supports the drafter's attention: mixing\n"
        "    # a non-causal layer onto a causal-only backend would fail.\n"
        "    draft_vllm_config = replace(\n"
        "        vllm_config,\n"
        "        attention_config=replace(\n"
        "            vllm_config.attention_config,\n"
        "            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),\n"
        "            backend=speculative_config.attention_backend,\n"
        "        ),\n"
        "        cache_config=(\n"
        "            replace(\n"
        "                vllm_config.cache_config,\n"
        "                cache_dtype=speculative_config.kv_cache_dtype,\n"
        "            )\n"
        "            if speculative_config.kv_cache_dtype is not None\n"
        "            else vllm_config.cache_config\n"
        "        ),\n"
        "    )\n",
        "    speculative_config = vllm_config.speculative_config\n"
        "    assert speculative_config is not None\n"
        "    draft_model_config = speculative_config.draft_model_config\n"
        "    # Dense DFlash2 attention cannot use the target's MLA-only fp8_ds_mla\n"
        "    # layout, and SM121 has no FA3/FA4 for plain FP8 KV. Keep draft KV in\n"
        "    # the model dtype unless speculative_config.kv_cache_dtype is set.\n"
        "    draft_kv = speculative_config.kv_cache_dtype\n"
        '    if draft_kv is None and vllm_config.cache_config.cache_dtype in (\n'
        '        "fp8_ds_mla",\n'
        '        "fp8",\n'
        '        "fp8_e4m3",\n'
        '        "fp8_e5m2",\n'
        '        "nvfp4",\n'
        "    ):\n"
        '        draft_kv = "auto"\n'
        "    # Select an attention backend that supports the drafter's attention: mixing\n"
        "    # a non-causal layer onto a causal-only backend would fail.\n"
        "    draft_vllm_config = replace(\n"
        "        vllm_config,\n"
        "        attention_config=replace(\n"
        "            vllm_config.attention_config,\n"
        "            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),\n"
        "            backend=speculative_config.attention_backend,\n"
        "        ),\n"
        "        cache_config=(\n"
        "            replace(\n"
        "                vllm_config.cache_config,\n"
        "                cache_dtype=draft_kv,\n"
        "            )\n"
        "            if draft_kv is not None\n"
        "            else vllm_config.cache_config\n"
        "        ),\n"
        "    )\n",
    )

    spec_init = SITE / "v1/worker/gpu/spec_decode/__init__.py"
    # PORT 83252ea89: upstream already dispatches DFlash2 from the "dflash"
    # method arm (and the arm is now `elif`, not `if`). Assert the dispatch
    # is present instead of inserting a duplicate of it.
    require_once(
        spec_init,
        '        if "DFlash2DraftModel" in speculative_config.draft_model_config.architectures:\n'
        "            from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (\n"
        "                DFlash2Speculator,\n"
        "            )\n"
        "\n"
        "            return DFlash2Speculator(vllm_config, device)\n",
        "DFlash2Speculator dispatch arm",
    )

    compile(dst_model.read_text(), str(dst_model), "exec")
    compile(dst_spec.read_text(), str(dst_spec), "exec")
    compile(qwen.read_text(), str(qwen), "exec")
    compile(dflash_utils.read_text(), str(dflash_utils), "exec")
    compile(spec_init.read_text(), str(spec_init), "exec")
    print("dflash2 overlay installed")


if __name__ == "__main__":
    main()
