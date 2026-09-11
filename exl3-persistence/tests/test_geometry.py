# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS
import subprocess
import sys

import pytest
from recipe_persistence.geometry import Geometry, key_group


def caches():
    return NS(tensors=[NS(page_size_bytes=16), NS(page_size_bytes=32)],
              group_data_refs=[[NS(tensor_idx=0, page_size_bytes=7),
                                NS(tensor_idx=1, page_size_bytes=21)],
                               [NS(tensor_idx=0, page_size_bytes=4)]])


def test_actual_refs_not_layer_count_or_padding():
    geometry = Geometry.from_canonical(caches())
    assert geometry.group_bytes == (28, 4)
    assert geometry.row_bytes == 48
    assert geometry.rows_for_budget(100, 9) == 2
    assert geometry.rows_for_budget(1000, 3) == 3
    with pytest.raises(ValueError):
        geometry.rows_for_budget(47, 10)


def test_repeated_ref_is_payload_but_not_new_physical_tensor():
    c = caches()
    c.group_data_refs[0].append(NS(tensor_idx=0, page_size_bytes=7))
    g = Geometry.from_canonical(c)
    assert g.group_bytes == (35, 4)
    assert g.row_bytes == 48


def test_geometry_validation():
    c = caches()
    c.group_data_refs[0][0].page_size_bytes = 17
    with pytest.raises(ValueError):
        Geometry.from_canonical(c)
    with pytest.raises(ValueError):
        key_group(b"hash" + (2).to_bytes(4, "big"), 2)
    with pytest.raises(ValueError):
        key_group(b"tiny", 2)


def test_metadata_import_does_not_import_vllm_or_torch():
    subprocess.run([sys.executable, "-c", (
        "import sys; import recipe_persistence.geometry; "
        "import recipe_persistence.handlers; import recipe_persistence.native; "
        "assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules")],
        check=True)
