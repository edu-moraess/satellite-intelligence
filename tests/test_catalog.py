"""Unit tests for location catalog (no internet)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.catalog.locations import (
    load_catalog,
    search_locations,
    get_location_by_id,
    list_categories,
    list_countries,
)


def test_load_catalog():
    df = load_catalog()
    assert len(df) > 100
    assert "latitude" in df.columns
    assert "longitude" in df.columns


def test_search_tokyo():
    results = search_locations("Tokyo")
    assert len(results) >= 1
    assert any("Tokyo" in r.location for r in results)


def test_search_sao_paulo():
    results = search_locations("São Paulo")
    assert len(results) >= 1


def test_get_by_id():
    df = load_catalog()
    first_id = df.iloc[0]["id"]
    loc = get_location_by_id(first_id)
    assert loc is not None
    assert loc.id == first_id


def test_categories():
    cats = list_categories()
    assert "CAPITAL" in cats
    assert "CITY" in cats


def test_countries():
    countries = list_countries()
    assert "Brazil" in countries or "Japan" in countries


def test_namibia_country_code():
    """ISO code NA must not be parsed as NaN."""
    from src.catalog.locations import load_catalog
    load_catalog.cache_clear()
    df = load_catalog()
    nam = df[df["country"] == "Namibia"]
    assert len(nam) >= 1
    assert nam.iloc[0]["country_code"] == "NA"
    assert df["country_code"].isnull().sum() == 0
