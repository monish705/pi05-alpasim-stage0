from __future__ import annotations

import sys
import types


def _install_lerobot_compat() -> None:
    try:
        from lerobot.datasets import lerobot_dataset as modern_lerobot_dataset
    except Exception:
        return

    if "lerobot.common.datasets.lerobot_dataset" in sys.modules:
        return

    common_pkg = types.ModuleType("lerobot.common")
    datasets_pkg = types.ModuleType("lerobot.common.datasets")
    datasets_pkg.lerobot_dataset = modern_lerobot_dataset
    common_pkg.datasets = datasets_pkg

    sys.modules.setdefault("lerobot.common", common_pkg)
    sys.modules.setdefault("lerobot.common.datasets", datasets_pkg)
    sys.modules.setdefault("lerobot.common.datasets.lerobot_dataset", modern_lerobot_dataset)


_install_lerobot_compat()
