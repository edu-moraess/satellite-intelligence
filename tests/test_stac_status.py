"""STAC search status differentiation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.satellite.stac import search_scenes, SearchStatus, SearchResult


def test_invalid_aoi_status():
    result = search_scenes(aoi="not-a-geometry")
    assert isinstance(result, SearchResult)
    assert result.status == SearchStatus.INVALID_AOI
    assert len(result.scenes) == 0


def test_search_result_iterable():
    r = SearchResult(status=SearchStatus.NO_RESULTS, scenes=[], message="none")
    assert list(r) == []
    assert len(r) == 0
    assert not r
