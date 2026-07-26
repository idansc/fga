"""Compatibility shim: this module moved to `fga.tasks.visual_dialog.configuration_fga`.

The split separates the general attention layer from the Visual Dialog
application built on it. Aliasing the module object rather than re-exporting
names keeps `fga.configuration_fga` and its new home the same object, so `isinstance` checks
and pickles agree and nothing can drift out of sync.
"""

import sys

from .tasks.visual_dialog import configuration_fga as _moved

sys.modules[__name__] = _moved
