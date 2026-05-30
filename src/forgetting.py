"""
Strategic Forgetting Engine

Recalculates base-level activation and transitions memory status:
    A(m) > 0       -> active    (readily retrievable)
    -2 < A(m) <= 0 -> dormant   (retrievable but deprioritized)
    A(m) <= -2     -> forgotten  (excluded from default results)

forgotten != deleted: stays in DB, queryable with include_forgotten=true
"""

import logging
from datetime import datetime, timezone

from actr_scoring import ACTRConfig, compute_base_level

logger = logging.getLogger("claude-memory")

ACTIVE_THRESHOLD = 0.0
DORMANT_THRESHOLD = -2.0


def classify_memory_status(base_level: float) -> str:
    if base_level > ACTIVE_THRESHOLD:
        return "active"
    elif base_level > DORMANT_THRESHOLD:
        return "dormant"
    else:
        return "forgotten"


async def run_forgetting_cycle(pool, config: ACTRConfig | None = None) -> str:
    if config is None:
        config = ACTRConfig()

    counters = {"active": 0, "dormant": 0, "forgotten": 0, "unchanged": 0}
    now = datetime.now(timezone.utc)

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(42)")

            rows = await conn.fetch("""
                SELECT id, access_timestamps, created_at, memory_status
                FROM memories
                WHERE memory_status IS NULL
                   OR memory_status IN ('active', 'dormant', 'forgotten')
                FOR UPDATE
            """)

            updates = {"active": [], "dormant": [], "forgotten": []}

            for row in rows:
                access_ts = row["access_timestamps"] or []
                created_at = row["created_at"]
                if created_at and created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)

                base_level = compute_base_level(access_ts, created_at, config.d)
                new_status = classify_memory_status(base_level)
                old_status = row["memory_status"]

                if old_status != new_status:
                    updates[new_status].append((row["id"], base_level))
                    counters[new_status] += 1
                else:
                    counters["unchanged"] += 1

            for status, memory_list in updates.items():
                if not memory_list:
                    continue
                await conn.executemany("""
                    UPDATE memories
                    SET memory_status = $1,
                        actr_activation = $2,
                        activation_updated_at = $3
                    WHERE id = $4
                """, [(status, act, now, mid) for mid, act in memory_list])

    total = sum(counters.values())
    return (
        f"Forgetting cycle complete.\n"
        f"Total memories processed: {total}\n"
        f"  Active: {counters['active']} transitions\n"
        f"  Dormant: {counters['dormant']} transitions\n"
        f"  Forgotten: {counters['forgotten']} transitions\n"
        f"  Unchanged: {counters['unchanged']}"
    )
