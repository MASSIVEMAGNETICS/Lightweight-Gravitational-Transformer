"""
boxol_flower – fractal 3-D voxel-bloom simulation.

Refactored from ``BoxolFlowerMonolith v1.6`` (boxol_flower_mono.py) into a
clean, importable Python package.

Public API::

    from boxol_flower import BoxolFlower, GuiController, __version__

``BoxolFlower`` is a pure-NumPy headless core that works in any environment.
``GuiController`` wraps it with a Tkinter/Matplotlib 3-D scatter window and
is only functional when both libraries are available with a live display.
"""

from .boxol import BoxolFlower, GuiController, __version__

__all__ = ["BoxolFlower", "GuiController", "__version__"]
