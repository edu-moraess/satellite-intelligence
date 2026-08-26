"""
Global Location Catalog for Satellite Intelligence.

Provides search and retrieval over a curated geospatial catalog
of countries, capitals, cities, landmarks, mountains, volcanoes,
ports, airports, mining, industrial, agricultural, forest, lake,
island and coastal areas.

Coordinates are sourced from publicly available authoritative datasets
(ISO 3166, national capital lists, public domain geographic databases).
See README for full attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "locations" / "global_locations.csv"


@dataclass(frozen=True)
class Location:
    """Immutable location record."""

    id: str
    country: str
    country_code: str
    region: str
    location: str
    category: str
    latitude: float
    longitude: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "country": self.country,
            "country_code": self.country_code,
            "region": self.region,
            "location": self.location,
            "category": self.category,
            "latitude": self.latitude,
            "longitude": self.longitude,
        }

    @property
    def display_name(self) -> str:
        return f"{self.location}, {self.country}"

    @property
    def coordinates_str(self) -> str:
        return f"{self.latitude:.4f}, {self.longitude:.4f}"


@lru_cache(maxsize=1)
def load_catalog() -> pd.DataFrame:
    """Load the global location catalog (cached)."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Location catalog not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, keep_default_na=False, na_values=[""])
    required = {"id", "country", "country_code", "region", "location", "category", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Catalog missing columns: {missing}")
    return df


def search_locations(query: str, limit: int = 15) -> List[Location]:
    """
    Search locations by name, country or category.

    Case-insensitive partial match on location, country and category.
    Returns ranked results (exact prefix first).
    """
    if not query or not query.strip():
        return []

    q = query.strip().lower()
    df = load_catalog()

    scores = []
    for idx, row in df.iterrows():
        loc_l = str(row["location"]).lower()
        country_l = str(row["country"]).lower()
        cat_l = str(row["category"]).lower()
        score = 0
        if loc_l == q:
            score = 100
        elif loc_l.startswith(q):
            score = 80
        elif q in loc_l:
            score = 60
        elif country_l.startswith(q) or q in country_l:
            score = 40
        elif q in cat_l:
            score = 20
        if score > 0:
            scores.append((score, idx))

    scores.sort(key=lambda x: (-x[0], str(df.loc[x[1], "location"])))
    results = []
    for _, idx in scores[:limit]:
        row = df.loc[idx]
        results.append(
            Location(
                id=str(row["id"]),
                country=str(row["country"]),
                country_code=str(row["country_code"]),
                region=str(row["region"]),
                location=str(row["location"]),
                category=str(row["category"]),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
            )
        )
    return results


def get_location_by_id(location_id: str) -> Optional[Location]:
    """Retrieve a single location by its id."""
    df = load_catalog()
    match = df[df["id"] == location_id]
    if match.empty:
        return None
    row = match.iloc[0]
    return Location(
        id=str(row["id"]),
        country=str(row["country"]),
        country_code=str(row["country_code"]),
        region=str(row["region"]),
        location=str(row["location"]),
        category=str(row["category"]),
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
    )


def list_categories() -> List[str]:
    """Return sorted unique categories."""
    df = load_catalog()
    return sorted(df["category"].unique().tolist())


def list_countries() -> List[str]:
    """Return sorted unique countries."""
    df = load_catalog()
    return sorted(df["country"].unique().tolist())


def get_all_locations() -> List[Location]:
    """Return every location in the catalog."""
    df = load_catalog()
    return [
        Location(
            id=str(row["id"]),
            country=str(row["country"]),
            country_code=str(row["country_code"]),
            region=str(row["region"]),
            location=str(row["location"]),
            category=str(row["category"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
        )
        for _, row in df.iterrows()
    ]
