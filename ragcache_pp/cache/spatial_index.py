"""Spatial index using geohash for metadata-aware cache policies.

Provides geohash encoding/decoding, neighbor lookup, and per-region
utility tracking for spatial-aware admission, eviction, and prefetch.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

# Geohash base32 alphabet
_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_DECODEMAP = {c: i for i, c in enumerate(_BASE32)}

# Neighbor lookup tables
_NEIGHBORS = {
    "right":  {"even": "bc01fg45telefonías89telefonías", "odd": "p0r21436x8zb9dcf5h7kjnmqesgutwvy"},
    "left":   {"even": "238967debc01telefonías45telefonías", "odd": "14365h7k9dcfesgtwuv238967debc01"},
    "top":    {"even": "p0r21436x8zb9dcf5h7kjnmqesgutwvy", "odd": "bc01fg45telefonías89telefonías"},
    "bottom": {"even": "14365h7k9dcfesgtwuv238967debc01", "odd": "238967debc01telefonías45telefonías"},
}
# Actually, let's use a proper geohash implementation


def encode_geohash(lat: float, lon: float, precision: int = 6) -> str:
    """Encode latitude/longitude to geohash string.

    Args:
        lat: Latitude in degrees [-90, 90].
        lon: Longitude in degrees [-180, 180].
        precision: Number of characters in the geohash (default 6 ≈ 1.2km).

    Returns:
        Geohash string of the given precision.
    """
    lat_range = (-90.0, 90.0)
    lon_range = (-180.0, 180.0)
    geohash_chars: list[str] = []
    bits = 0
    char_idx = 0
    is_lon = True  # alternate lon/lat bits

    while len(geohash_chars) < precision:
        if is_lon:
            mid = (lon_range[0] + lon_range[1]) / 2
            if lon >= mid:
                char_idx = (char_idx << 1) | 1
                lon_range = (mid, lon_range[1])
            else:
                char_idx = char_idx << 1
                lon_range = (lon_range[0], mid)
        else:
            mid = (lat_range[0] + lat_range[1]) / 2
            if lat >= mid:
                char_idx = (char_idx << 1) | 1
                lat_range = (mid, lat_range[1])
            else:
                char_idx = char_idx << 1
                lat_range = (lat_range[0], mid)

        is_lon = not is_lon
        bits += 1

        if bits == 5:
            geohash_chars.append(_BASE32[char_idx])
            bits = 0
            char_idx = 0

    return "".join(geohash_chars)


def decode_geohash(ghash: str) -> tuple[float, float]:
    """Decode geohash string to (latitude, longitude) center point."""
    lat_range = (-90.0, 90.0)
    lon_range = (-180.0, 180.0)
    is_lon = True

    for char in ghash:
        val = _DECODEMAP[char]
        for i in range(4, -1, -1):
            bit = (val >> i) & 1
            if is_lon:
                mid = (lon_range[0] + lon_range[1]) / 2
                if bit:
                    lon_range = (mid, lon_range[1])
                else:
                    lon_range = (lon_range[0], mid)
            else:
                mid = (lat_range[0] + lat_range[1]) / 2
                if bit:
                    lat_range = (mid, lat_range[1])
                else:
                    lat_range = (lat_range[0], mid)
            is_lon = not is_lon

    return (lat_range[0] + lat_range[1]) / 2, (lon_range[0] + lon_range[1]) / 2


def geohash_neighbors(ghash: str) -> list[str]:
    """Return the 8 neighboring geohash cells at the same precision.

    Uses the standard geohash adjacency algorithm.
    """
    if not ghash:
        return []

    neighbors: list[str] = []
    for dlat in (-1, 0, 1):
        for dlon in (-1, 0, 1):
            if dlat == 0 and dlon == 0:
                continue
            lat, lon = decode_geohash(ghash)
            # Approximate cell size at this precision
            precision = len(ghash)
            # Each geohash char encodes 5 bits; alternating lon/lat
            lat_bits = (precision * 5) // 2
            lon_bits = (precision * 5) - lat_bits
            lat_cell_size = 180.0 / (2 ** lat_bits)
            lon_cell_size = 360.0 / (2 ** lon_bits)

            new_lat = lat + dlat * lat_cell_size
            new_lon = lon + dlon * lon_cell_size

            # Clamp to valid ranges
            new_lat = max(-90.0, min(90.0, new_lat))
            new_lon = max(-180.0, min(180.0, new_lon))

            neighbor_hash = encode_geohash(new_lat, new_lon, precision)
            if neighbor_hash != ghash:
                neighbors.append(neighbor_hash)

    # Deduplicate (edge cells may produce duplicates)
    return list(set(neighbors))


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in meters between two lat/lon points."""
    R = 6_371_000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass
class RegionStats:
    """Per-geohash-cell statistics for spatial-aware caching."""

    geohash: str
    total_hits: int = 0
    total_prefill_savings_ms: float = 0.0
    cache_footprint_blocks: int = 0
    last_access_time: float = 0.0
    doc_ids: set[str] = field(default_factory=set)

    @property
    def utility(self) -> float:
        """Region utility: U(g) = total_savings / footprint.

        total_prefill_savings_ms already accumulates savings across all hits,
        so dividing by footprint gives savings-per-block (no double-counting).
        """
        if self.cache_footprint_blocks == 0:
            return 0.0
        return self.total_prefill_savings_ms / self.cache_footprint_blocks

    def record_hit(self, prefill_savings_ms: float) -> None:
        self.total_hits += 1
        self.total_prefill_savings_ms += prefill_savings_ms
        self.last_access_time = time.time()

    def apply_decay(self, decay_rate: float) -> None:
        """Apply temporal decay to accumulated statistics."""
        self.total_hits = int(self.total_hits * decay_rate)
        self.total_prefill_savings_ms *= decay_rate


class SpatialIndex:
    """Geohash-based spatial index for metadata-aware caching.

    Maintains per-region statistics and supports neighbor-based prefetch
    decisions and spatial-aware admission/eviction.
    """

    def __init__(self, precision: int = 6, region_decay_rate: float = 0.99):
        self.precision = precision
        self.region_decay_rate = region_decay_rate
        self._regions: dict[str, RegionStats] = {}
        self._doc_to_geohash: dict[str, str] = {}  # doc_id -> geohash
        self._step_count = 0

    def register_document(self, doc_id: str, lat: float, lon: float) -> str:
        """Register a document's geospatial metadata. Returns its geohash."""
        ghash = encode_geohash(lat, lon, self.precision)
        self._doc_to_geohash[doc_id] = ghash

        if ghash not in self._regions:
            self._regions[ghash] = RegionStats(geohash=ghash)
        self._regions[ghash].doc_ids.add(doc_id)

        return ghash

    def get_doc_geohash(self, doc_id: str) -> Optional[str]:
        return self._doc_to_geohash.get(doc_id)

    def get_region(self, geohash: str) -> Optional[RegionStats]:
        return self._regions.get(geohash)

    def get_region_utility(self, geohash: str) -> float:
        region = self._regions.get(geohash)
        return region.utility if region else 0.0

    def record_cache_hit(self, doc_id: str, prefill_savings_ms: float) -> None:
        """Record a cache hit for a document, updating its region stats."""
        ghash = self._doc_to_geohash.get(doc_id)
        if ghash and ghash in self._regions:
            self._regions[ghash].record_hit(prefill_savings_ms)

    def update_footprint(self, doc_id: str, num_blocks: int, delta: int) -> None:
        """Update cache footprint when a document is cached/evicted.

        Args:
            doc_id: Document identifier.
            num_blocks: Number of blocks for the document.
            delta: +1 for cache insertion, -1 for eviction.
        """
        ghash = self._doc_to_geohash.get(doc_id)
        if ghash and ghash in self._regions:
            self._regions[ghash].cache_footprint_blocks += num_blocks * delta
            self._regions[ghash].cache_footprint_blocks = max(0, self._regions[ghash].cache_footprint_blocks)

    def get_prefetch_candidates(self, query_lat: float, query_lon: float, budget: int) -> list[str]:
        """Get top-P document IDs to prefetch based on spatial proximity.

        Looks at the query's geohash cell and neighboring cells, returns
        the highest-priority documents from those regions.

        Args:
            query_lat: Query latitude.
            query_lon: Query longitude.
            budget: Max number of documents to prefetch.

        Returns:
            List of doc_ids to prefetch.
        """
        query_ghash = encode_geohash(query_lat, query_lon, self.precision)
        neighbor_hashes = geohash_neighbors(query_ghash)

        # Collect candidates from neighbor cells (not the query cell itself,
        # since those are likely already being retrieved)
        candidates: list[tuple[float, str]] = []
        for ghash in neighbor_hashes:
            region = self._regions.get(ghash)
            if region is None:
                continue
            region_util = region.utility
            for doc_id in region.doc_ids:
                candidates.append((region_util, doc_id))

        # Sort by region utility (descending), take top-P
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [doc_id for _, doc_id in candidates[:budget]]

    def check_admission(self, doc_id: str, retrieval_rank: int, tau_admit: float, k_hot: int) -> bool:
        """Check whether a document should be admitted to cache.

        Admission rule:
        - Admit if retrieval rank <= k_hot (high relevance regardless of region)
        - Admit if region utility U(g) > tau_admit (spatially hot region)
        - Fallback to rank-based when region has no utility data yet (bootstrap)

        Args:
            doc_id: Document identifier.
            retrieval_rank: 1-based rank in retrieval results.
            tau_admit: Region utility threshold.
            k_hot: Top-k rank threshold for unconditional admission.

        Returns:
            True if the document should be admitted.
        """
        if retrieval_rank <= k_hot:
            return True

        ghash = self._doc_to_geohash.get(doc_id)
        if ghash:
            utility = self.get_region_utility(ghash)
            if utility > 0:
                return utility > tau_admit
            # No utility data yet (cold start) — fall back to rank-based
            return retrieval_rank <= k_hot * 2

        # No spatial metadata — admit by default (fallback to rank-based)
        return retrieval_rank <= k_hot * 2

    def step(self) -> None:
        """Advance one time step; apply periodic decay to region stats."""
        self._step_count += 1
        # Apply decay every 100 steps to avoid overhead
        if self._step_count % 100 == 0:
            for region in self._regions.values():
                region.apply_decay(self.region_decay_rate)

    def get_stats(self) -> dict:
        """Return summary statistics for profiling."""
        active = [r for r in self._regions.values() if r.total_hits > 0]
        return {
            "total_regions": len(self._regions),
            "active_regions": len(active),
            "total_documents_indexed": len(self._doc_to_geohash),
            "avg_region_utility": (
                sum(r.utility for r in active) / len(active) if active else 0.0
            ),
        }
