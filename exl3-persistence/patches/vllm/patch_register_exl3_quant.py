#!/usr/bin/env python3
"""Register ``exl3`` in vLLM's QuantizationMethods (lazy import).

Ported verbatim from Mia's ``Dockerfile`` heredoc (lines 386-415). Both anchors
re-confirmed ``count() == 1`` on the pinned tree ``83252ea89``.

``Exl3Config.override_quantization_method`` requires ``"exl3"`` to be in
ModelConfig's ordered overrides list, which is derived from this Literal.
Fail-closed: a drifted anchor raises and the build stops.
"""

from __future__ import annotations

import os
from pathlib import Path

SITE = Path(os.environ.get("GLM53_SITE", "/usr/local/lib/python3.12/dist-packages"))
init = SITE / "vllm/model_executor/layers/quantization/__init__.py"

text = init.read_text()

old = 'QuantizationMethods = Literal[\n    "awq",\n'
new = 'QuantizationMethods = Literal[\n    "exl3",\n    "awq",\n'
if text.count(old) != 1:
    raise RuntimeError("expected one QuantizationMethods Literal header")
text = text.replace(old, new)

lazy = (
    "    # Update the `method_to_config` with customized quantization methods.\n"
    "    method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)\n"
    "\n"
    "    return method_to_config[quantization]\n"
)
lazy_new = (
    "    from .exl3 import Exl3Config\n"
    '    method_to_config["exl3"] = Exl3Config\n'
    "    # Update the `method_to_config` with customized quantization methods.\n"
    "    method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)\n"
    "\n"
    "    return method_to_config[quantization]\n"
)
if text.count(lazy) != 1:
    raise RuntimeError("expected one get_quantization_config trailer")
text = text.replace(lazy, lazy_new)

init.write_text(text)
print("registered exl3 in QUANTIZATION_METHODS (lazy import)")
