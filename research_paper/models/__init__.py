"""WetMamba model components for wetland segmentation research.

Architecture: Prithvi-EO-2.0 encoder + Mamba SSM decoder + Depression-Aware
Gating (DAG) + Multi-Temporal SSM fusion.

References:
    - Prithvi-EO-2.0 (Jakubik et al. 2024): NASA/IBM geospatial FM
    - Mamba (Gu & Dao 2024): Selective state space model
    - RS3Mamba (Ma et al. 2024): Mamba for remote sensing segmentation
    - Wu et al. (2019) RSE: LiDAR depression-based wetland mapping
"""

from research_paper.models.dag_module import DepressionAwareGating
from research_paper.models.mamba_decoder import MambaDecoderBlock, MambaDecoder
from research_paper.models.temporal_ssm import TemporalSSMFusion
from research_paper.models.wetmamba import WetMamba

__all__ = [
    "DepressionAwareGating",
    "MambaDecoderBlock",
    "MambaDecoder",
    "TemporalSSMFusion",
    "WetMamba",
]
