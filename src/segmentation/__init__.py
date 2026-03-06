from .pipeline import SegmentationPipeline
from .rules import RuleBasedSegmenter
from .clustering import PersonaClusterer, BehavioralClusterer  # BehavioralClusterer deprecated
from .router import PersonaRouter

__all__ = [
    "SegmentationPipeline",
    "RuleBasedSegmenter",
    "PersonaClusterer",
    "PersonaRouter",
    "BehavioralClusterer",   # backward compat
]
