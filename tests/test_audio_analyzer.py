import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from modules.audio_analyzer import AudioAnalyzer


def test_audio_analyzer_not_implemented():
    """Phase 1 stub — 구현 전 NotImplementedError가 발생해야 함."""
    with pytest.raises(NotImplementedError):
        AudioAnalyzer()
