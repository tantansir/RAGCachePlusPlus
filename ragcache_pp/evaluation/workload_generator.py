"""Workload generator for RAGCache++ evaluation.

Generates realistic RAG query traces following the pre-specified trace protocol:
- Mixed hotspot + tail (60/40 split)
- Time-varying demand (rotating hotspot phases)
- Top-k variation, chunk-length variation, retrieval noise
- Geospatial workloads with spatial constraints
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from ragcache_pp.cache.spatial_index import encode_geohash, haversine_distance
from ragcache_pp.serving.rag_controller import RAGRequest


@dataclass
class POI:
    """Point of Interest for geospatial workload."""

    doc_id: str
    name: str
    latitude: float
    longitude: float
    description: str = ""
    category: str = ""
    num_tokens: int = 256  # token count of the document chunk

    @property
    def geohash(self) -> str:
        return encode_geohash(self.latitude, self.longitude, precision=6)


@dataclass
class WorkloadConfig:
    """Configuration for workload generation."""

    num_queries: int = 1000
    num_pois: int = 5000
    top_k_choices: list[int] = field(default_factory=lambda: [3, 5, 10])
    chunk_length_range: tuple[int, int] = (64, 512)

    # Spatial distribution
    hotspot_fraction: float = 0.6
    num_hotspot_regions: int = 3
    phase_length: int = 1000  # queries per demand phase

    # Retrieval noise
    retrieval_noise_fraction: float = 0.1

    # Geographic bounds (NYC default)
    lat_min: float = 40.70
    lat_max: float = 40.82
    lon_min: float = -74.02
    lon_max: float = -73.93

    # Random seed for reproducibility
    seed: int = 42


class SyntheticPOIGenerator:
    """Generate synthetic POIs for geospatial workload."""

    # NYC hotspot regions (approximate centers)
    NYC_HOTSPOTS = [
        {"name": "Midtown", "lat": 40.7580, "lon": -73.9855, "radius_km": 1.0},
        {"name": "Lower Manhattan", "lat": 40.7128, "lon": -74.0060, "radius_km": 0.8},
        {"name": "Brooklyn Heights", "lat": 40.6959, "lon": -73.9957, "radius_km": 0.6},
        {"name": "Upper East Side", "lat": 40.7736, "lon": -73.9566, "radius_km": 0.7},
        {"name": "Williamsburg", "lat": 40.7081, "lon": -73.9571, "radius_km": 0.5},
    ]

    CATEGORIES = [
        "restaurant", "hotel", "museum", "park", "theater",
        "cafe", "bar", "shop", "attraction", "landmark",
    ]

    def __init__(self, config: WorkloadConfig):
        self.config = config
        self.rng = random.Random(config.seed)

    def generate_pois(self) -> list[POI]:
        """Generate synthetic POIs distributed across NYC."""
        pois: list[POI] = []
        for i in range(self.config.num_pois):
            # 70% of POIs in hotspot regions, 30% uniform
            if self.rng.random() < 0.7 and self.NYC_HOTSPOTS:
                hotspot = self.rng.choice(self.NYC_HOTSPOTS)
                lat = hotspot["lat"] + self.rng.gauss(0, hotspot["radius_km"] / 111)
                lon = hotspot["lon"] + self.rng.gauss(0, hotspot["radius_km"] / (111 * math.cos(math.radians(hotspot["lat"]))))
            else:
                lat = self.rng.uniform(self.config.lat_min, self.config.lat_max)
                lon = self.rng.uniform(self.config.lon_min, self.config.lon_max)

            # Clamp to bounds
            lat = max(self.config.lat_min, min(self.config.lat_max, lat))
            lon = max(self.config.lon_min, min(self.config.lon_max, lon))

            num_tokens = self.rng.randint(*self.config.chunk_length_range)
            category = self.rng.choice(self.CATEGORIES)

            poi = POI(
                doc_id=f"poi_{i:05d}",
                name=f"POI_{i}_{category}",
                latitude=lat,
                longitude=lon,
                description=f"A {category} located at ({lat:.4f}, {lon:.4f})",
                category=category,
                num_tokens=num_tokens,
            )
            pois.append(poi)

        return pois


class GeoSpatialWorkloadGenerator:
    """Generate geospatial RAG query traces.

    Follows the pre-specified trace protocol:
    - 60% from hotspot regions, 40% uniform
    - Time-varying demand with rotating hotspots
    - Top-k variation, retrieval noise
    """

    def __init__(self, config: WorkloadConfig, pois: list[POI]):
        self.config = config
        self.pois = pois
        self.rng = random.Random(config.seed + 1)

        # Build spatial index for retrieval simulation
        self._poi_by_id: dict[str, POI] = {p.doc_id: p for p in pois}
        self._poi_coords: list[tuple[float, float]] = [(p.latitude, p.longitude) for p in pois]

    def generate_trace(self) -> list[RAGRequest]:
        """Generate the full query trace."""
        requests: list[RAGRequest] = []

        hotspots = SyntheticPOIGenerator.NYC_HOTSPOTS[:self.config.num_hotspot_regions]

        for q_idx in range(self.config.num_queries):
            # Determine current phase and active hotspot
            phase = (q_idx // self.config.phase_length) % len(hotspots)
            active_hotspot = hotspots[phase]

            # Query location
            if self.rng.random() < self.config.hotspot_fraction:
                # Hotspot query
                lat = active_hotspot["lat"] + self.rng.gauss(0, active_hotspot["radius_km"] / 111)
                lon = active_hotspot["lon"] + self.rng.gauss(
                    0, active_hotspot["radius_km"] / (111 * math.cos(math.radians(active_hotspot["lat"])))
                )
            else:
                # Uniform tail query
                lat = self.rng.uniform(self.config.lat_min, self.config.lat_max)
                lon = self.rng.uniform(self.config.lon_min, self.config.lon_max)

            lat = max(self.config.lat_min, min(self.config.lat_max, lat))
            lon = max(self.config.lon_min, min(self.config.lon_max, lon))

            # Top-k for this query
            top_k = self.rng.choice(self.config.top_k_choices)

            # Spatial radius for retrieval
            spatial_radius = self.rng.choice([200, 500, 1000, 5000])

            # Simulate retrieval: find nearest POIs
            distances = [
                (haversine_distance(lat, lon, p.latitude, p.longitude), p)
                for p in self.pois
            ]
            distances.sort(key=lambda x: x[0])

            # Filter by spatial radius
            spatial_candidates = [(d, p) for d, p in distances if d <= spatial_radius]
            if len(spatial_candidates) < top_k:
                spatial_candidates = distances[:top_k * 2]  # fallback to nearest

            # Select top-k
            retrieved = spatial_candidates[:top_k]

            # Add retrieval noise
            num_noise = int(len(retrieved) * self.config.retrieval_noise_fraction)
            if num_noise > 0:
                noise_pois = self.rng.sample(self.pois, min(num_noise, len(self.pois)))
                for np_poi in noise_pois[:num_noise]:
                    if len(retrieved) > 0:
                        replace_idx = self.rng.randint(0, len(retrieved) - 1)
                        retrieved[replace_idx] = (99999.0, np_poi)

            request = RAGRequest(
                query_id=f"q_{q_idx:06d}",
                query_text=f"What are good places near ({lat:.4f}, {lon:.4f})?",
                latitude=lat,
                longitude=lon,
                spatial_radius_m=float(spatial_radius),
                retrieved_doc_ids=[p.doc_id for _, p in retrieved],
                retrieved_doc_tokens=[p.num_tokens for _, p in retrieved],
                retrieved_doc_geohashes=[p.geohash for _, p in retrieved],
                retrieval_ranks=list(range(1, len(retrieved) + 1)),
                arrival_time=q_idx * 0.1,  # 10 req/s baseline
            )
            requests.append(request)

        return requests

    def generate_uniform_trace(self, num_queries: Optional[int] = None) -> list[RAGRequest]:
        """Generate a uniform-random trace (negative control).

        No spatial clustering — for verifying spatial policy overhead is minimal.
        """
        n = num_queries or self.config.num_queries
        requests: list[RAGRequest] = []

        for q_idx in range(n):
            lat = self.rng.uniform(self.config.lat_min, self.config.lat_max)
            lon = self.rng.uniform(self.config.lon_min, self.config.lon_max)
            top_k = self.rng.choice(self.config.top_k_choices)

            distances = [
                (haversine_distance(lat, lon, p.latitude, p.longitude), p)
                for p in self.pois
            ]
            distances.sort(key=lambda x: x[0])
            retrieved = distances[:top_k]

            request = RAGRequest(
                query_id=f"uniform_q_{q_idx:06d}",
                query_text=f"Uniform query at ({lat:.4f}, {lon:.4f})",
                latitude=lat,
                longitude=lon,
                retrieved_doc_ids=[p.doc_id for _, p in retrieved],
                retrieved_doc_tokens=[p.num_tokens for _, p in retrieved],
                retrieved_doc_geohashes=[p.geohash for _, p in retrieved],
                retrieval_ranks=list(range(1, len(retrieved) + 1)),
                arrival_time=q_idx * 0.1,
            )
            requests.append(request)

        return requests

    def generate_no_locality_trace(self, num_queries: Optional[int] = None) -> list[RAGRequest]:
        """Generate a locality-destroyed trace (true negative control).

        Each query retrieves random docs (not nearest-neighbor), destroying
        spatial locality entirely. Compares spatial on/off fairly.
        """
        n = num_queries or self.config.num_queries
        requests: list[RAGRequest] = []

        for q_idx in range(n):
            lat = self.rng.uniform(self.config.lat_min, self.config.lat_max)
            lon = self.rng.uniform(self.config.lon_min, self.config.lon_max)
            top_k = self.rng.choice(self.config.top_k_choices)

            # Randomly sample docs instead of nearest-neighbor
            retrieved_pois = self.rng.sample(self.pois, min(top_k, len(self.pois)))

            request = RAGRequest(
                query_id=f"noloc_q_{q_idx:06d}",
                query_text=f"No-locality query at ({lat:.4f}, {lon:.4f})",
                latitude=lat,
                longitude=lon,
                retrieved_doc_ids=[p.doc_id for p in retrieved_pois],
                retrieved_doc_tokens=[p.num_tokens for p in retrieved_pois],
                retrieved_doc_geohashes=[p.geohash for p in retrieved_pois],
                retrieval_ranks=list(range(1, len(retrieved_pois) + 1)),
                arrival_time=q_idx * 0.1,
            )
            requests.append(request)

        return requests
