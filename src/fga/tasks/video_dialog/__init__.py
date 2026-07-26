"""Audio-visual scene-aware dialog.

Six modalities attended jointly: the question, four spatio-temporal video streams
and the audio track. The streams are attended individually and then fused across
the stream axis, so the model can weigh moments against each other after deciding
what to look at within each.

Ported from https://github.com/idansc/simple-avsd.
"""

from .modeling_avsd import MODALITY_NAMES, AVSDConfig, AVSDEncoder, AVSDEncoderOutput

__all__ = ["MODALITY_NAMES", "AVSDConfig", "AVSDEncoder", "AVSDEncoderOutput"]
