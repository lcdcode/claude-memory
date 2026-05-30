"""
ACT-R Cognitive Memory Scoring

Implements activation formula: A(m) = B(m) + w * cosine_sim + S(m) + epsilon

Based on the ACT-R cognitive architecture:
  B(m) - Base-level activation: frequency + power-law decay
  w * cosine - Semantic similarity weighted by context
  S(m) - Spreading activation via shared tags
  epsilon - Gaussian noise for probabilistic variability
"""

import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class ACTRConfig:
    d: float = 0.5          # Decay rate for power-law forgetting
    w: float = 11.0         # Weight for cosine similarity
    sigma: float = 1.2      # Noise std dev (0 = deterministic)
    tau: float = -2.0       # Retrieval threshold
    S: float = 2.0          # Max associative strength
    use_spreading: bool = True
    use_noise: bool = True
    prefetch_limit: int = 50
    use_actr: bool = True

    @classmethod
    def from_env(cls) -> "ACTRConfig":
        def _float(name: str, default: str, lo: float, hi: float) -> float:
            try:
                val = float(os.getenv(name, default))
            except ValueError:
                val = float(default)
            return max(lo, min(hi, val))

        def _int(name: str, default: str, lo: int, hi: int) -> int:
            try:
                val = int(os.getenv(name, default))
            except ValueError:
                val = int(default)
            return max(lo, min(hi, val))

        return cls(
            d=_float("ACTR_DECAY_D", "0.5", 0.0, 2.0),
            w=_float("ACTR_WEIGHT_W", "11.0", 0.0, 50.0),
            sigma=_float("ACTR_NOISE_SIGMA", "1.2", 0.0, 5.0),
            tau=_float("ACTR_THRESHOLD_TAU", "-2.0", -10.0, 10.0),
            S=_float("ACTR_SPREADING_S", "2.0", 0.0, 10.0),
            use_spreading=os.getenv("ACTR_USE_SPREADING", "true").lower() == "true",
            use_noise=os.getenv("ACTR_USE_NOISE", "true").lower() == "true",
            prefetch_limit=_int("ACTR_PREFETCH_LIMIT", "50", 1, 10000),
            use_actr=os.getenv("USE_ACTR_SCORING", "true").lower() == "true",
        )


def compute_base_level(
    access_timestamps: list[datetime],
    created_at: datetime,
    d: float = 0.5,
) -> float:
    """B(m) = ln(Sum_i (t_now - t_i)^(-d))"""
    now = datetime.now(timezone.utc)
    timestamps = list(access_timestamps) if access_timestamps else []
    if not timestamps:
        timestamps = [created_at]

    total = 0.0
    for ts in timestamps:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta_seconds = max((now - ts).total_seconds(), 1.0)
        total += delta_seconds ** (-d)

    if total <= 0:
        return -10.0
    return math.log(total)


def compute_spreading_activation(
    memory_tags: list[str],
    query_tags: list[str],
    tag_fan_counts: dict[str, int],
    S: float = 2.0,
) -> float:
    """Spreading activation via shared tags. Rarer tags = stronger signal."""
    if not memory_tags or not query_tags:
        return 0.0

    memory_set = {t.lower() for t in memory_tags}
    query_set = {t.lower() for t in query_tags}
    shared = memory_set & query_set

    if not shared:
        return 0.0

    W = 1.0 / max(len(query_set), 1)
    total = 0.0
    for tag in shared:
        fan = tag_fan_counts.get(tag, 1)
        sji = S - math.log(max(fan, 1))
        total += W * sji
    return total


def compute_noise(sigma: float = 1.2) -> float:
    if sigma <= 0:
        return 0.0
    return random.gauss(0, sigma)


def compute_activation(
    base_level: float,
    cosine_sim: float,
    spreading: float,
    noise: float,
    w: float = 11.0,
) -> float:
    """A(m) = B(m) + w * cosine_similarity + S(m) + epsilon"""
    return base_level + w * cosine_sim + spreading + noise


# Adaptive w based on query type
DEBUGGING_KEYWORDS = frozenset([
    "error", "bug", "fix", "debug", "exception", "traceback", "crash",
    "fail", "broken", "issue", "stack", "trace",
])

ARCHITECTURE_KEYWORDS = frozenset([
    "design", "architecture", "pattern", "system", "schema", "structure",
    "module", "component", "interface", "abstraction", "refactor", "migration",
])


def classify_query_type(
    query: str,
    tags: list[str] | None = None,
    category: str | None = None,
) -> str:
    words = set(query.lower().split())
    all_terms = words | {t.lower() for t in (tags or [])}

    if category in ("bugfix", "error_solution") or all_terms & DEBUGGING_KEYWORDS:
        return "debugging"
    if category in ("decision", "refactor") or all_terms & ARCHITECTURE_KEYWORDS:
        return "architecture"
    return "general"


def get_adaptive_w(query_type: str, base_w: float = 11.0) -> float:
    multipliers = {
        "debugging": 1.5,
        "architecture": 1.0,
        "recurrent": 0.6,
        "general": 1.0,
    }
    return base_w * multipliers.get(query_type, 1.0)


def score_and_rank_memories(
    rows: list[dict],
    query_tags: list[str] | None = None,
    tag_fan_counts: dict[str, int] | None = None,
    config: ACTRConfig | None = None,
    query: str = "",
    category: str | None = None,
) -> list[dict]:
    """Score and re-rank memories using ACT-R activation formula."""
    if config is None:
        config = ACTRConfig()
    if tag_fan_counts is None:
        tag_fan_counts = {}

    query_type = classify_query_type(query, query_tags, category)
    effective_w = get_adaptive_w(query_type, config.w)

    scored = []
    for row in rows:
        access_ts = row.get("access_timestamps") or []
        created_at = row.get("created_at", datetime.now(timezone.utc))
        if created_at and created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        cosine_sim = float(row.get("sim", 0.0))
        memory_tags = row.get("tags") or []

        B = compute_base_level(access_ts, created_at, config.d)
        S_val = 0.0
        if config.use_spreading and query_tags:
            S_val = compute_spreading_activation(
                memory_tags, query_tags, tag_fan_counts, config.S
            )
        epsilon = compute_noise(config.sigma) if config.use_noise else 0.0
        activation = compute_activation(B, cosine_sim, S_val, epsilon, effective_w)

        if activation >= config.tau:
            entry = dict(row)
            entry["activation_score"] = activation
            scored.append(entry)

    scored.sort(key=lambda x: x["activation_score"], reverse=True)
    return scored
