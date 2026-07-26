"""Compatibility shim: this module moved to `fga.tasks.visual_dialog.image_features`.

The split separates the general attention layer from the Visual Dialog
application built on it. Aliasing the module object rather than re-exporting
names keeps `fga.image_features` and its new home the same object, so `isinstance` checks
and pickles agree and nothing can drift out of sync.
"""

import sys

from .tasks.visual_dialog import image_features as _moved

sys.modules[__name__] = _moved
