"""Rin: raw video in, curated and checked labels out, small student models trained on them.

    from rin import Master, core
    m = Master("workspace/store")
"""

from rin import core
from rin.master import Master
from rin.run import Run

__all__ = ["Master", "Run", "core"]
