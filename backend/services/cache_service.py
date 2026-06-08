"""
Cache Service — In-memory LRU Cache
=====================================
Cache hasil prediksi berdasarkan file hash.
Tidak perlu Redis — pure Python, ringan untuk Mac.
"""

import hashlib
from collections import OrderedDict
from typing import Optional
from loguru import logger


class LRUCache:
    def __init__(self, max_size: int = 50):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size

    def _hash(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def get(self, image_bytes: bytes) -> Optional[dict]:
        key = self._hash(image_bytes)
        if key in self._cache:
            self._cache.move_to_end(key)
            logger.debug(f"[CACHE] HIT {key[:12]}")
            return self._cache[key]
        return None

    def set(self, image_bytes: bytes, result: dict) -> None:
        key = self._hash(image_bytes)
        self._cache[key] = result
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            evicted = self._cache.popitem(last=False)
            logger.debug(f"[CACHE] EVICT {list(evicted)[0][:12]}")
        logger.debug(f"[CACHE] SET {key[:12]}")

    def clear(self) -> None:
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


# Singleton
prediction_cache = LRUCache(max_size=50)