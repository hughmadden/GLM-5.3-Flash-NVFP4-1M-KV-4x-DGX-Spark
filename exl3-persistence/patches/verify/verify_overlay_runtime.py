#!/usr/bin/env python3
"""Import-level verification (torch/vLLM are imported). qemu-slow; build-arg gated.

This is Mia's original build-time check (``Dockerfile`` lines 320-367) kept
intact, minus the source-text assertions that ``verify_overlay_static.py``
already covers without imports. It proves the patched modules actually import
and expose the patched attributes, which text matching cannot.

Under ``docker buildx --platform linux/arm64`` on an x86_64 builder this runs
under qemu-user and is minutes-slow. Set ``--build-arg RUN_IMPORT_CHECKS=0``
to skip it; if you do, this check MUST be run on a Spark before the image is
trusted (``docker run --rm --entrypoint python3 <tag>
/opt/glm53/verify/verify_overlay_runtime.py``).
"""

from __future__ import annotations

import inspect

from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
    FlashInferMLASparseSM120Impl as impl,
)

assert impl.supports_dense_mha_prefill is False
assert "do_kv_cache_update" in impl.__dict__
init_src = inspect.getsource(impl.__init__)
assert "self.rope_pad = 64" in init_src
fwd_src = inspect.getsource(impl.forward_mqa)
assert "torch.nn.functional.pad(q, (0, self.rope_pad))" in fwd_src
assert "qk_rope_head_dim=self.kernel_qk_rope_head_dim" in fwd_src
assert "return_valid_counts=True" in fwd_src
assert "sparse_mla_top_k=sparse_topk_capacity" in fwd_src
assert "seq_lens=topk_lengths" in fwd_src
assert "out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)" in fwd_src
assert "attn_metadata.topk_tokens" not in fwd_src

from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (  # noqa: E402
    FlashInferMLASparseSM120Backend as sm120_backend,
)

assert sm120_backend.get_supported_kernel_block_sizes() == [64]

from vllm.model_executor.layers.quantization import (  # noqa: E402
    QUANTIZATION_METHODS,
    get_quantization_config,
)

assert "exl3" in QUANTIZATION_METHODS, QUANTIZATION_METHODS
assert get_quantization_config("exl3").__name__ == "Exl3Config"

import recipe_persistence  # noqa: E402,F401
import startup_observer  # noqa: E402

assert callable(startup_observer.emit)

print("glm53 overlay runtime (import) verify OK")
