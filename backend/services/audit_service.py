"""
Audit Service — Append-only JSON audit log
============================================
Mencatat semua prediksi untuk compliance dan monitoring.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from loguru import logger

AUDIT_LOG_PATH = Path("outputs/audit_log.jsonl")


class AuditService:

    def __init__(self):
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        event: str,
        file_id: str,
        filename: str,
        result: Optional[str] = None,
        confidence: Optional[float] = None,
        risk_level: Optional[str] = None,
        rejected: bool = False,
        rejection_reason: Optional[str] = None,
        inference_ms: Optional[float] = None,
        cached: bool = False,
    ) -> None:
        entry = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "file_id": file_id,
            "filename": filename,
            "result": result,
            "confidence": confidence,
            "risk_level": risk_level,
            "rejected": rejected,
            "rejection_reason": rejection_reason,
            "inference_ms": inference_ms,
            "cached": cached,
        }
        try:
            with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"[AUDIT] Failed to write log: {e}")

    def get_recent(self, limit: int = 100) -> list[dict]:
        try:
            if not AUDIT_LOG_PATH.exists():
                return []
            lines = AUDIT_LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
            entries = [json.loads(l) for l in lines[-limit:]]
            return list(reversed(entries))
        except Exception as e:
            logger.error(f"[AUDIT] Failed to read log: {e}")
            return []

    def get_stats(self) -> dict:
        entries = self.get_recent(limit=10000)
        if not entries:
            return {}
        predictions = [e for e in entries if not e.get("rejected")]
        rejections  = [e for e in entries if e.get("rejected")]
        class_counts = {}
        for e in predictions:
            r = e.get("result")
            if r:
                class_counts[r] = class_counts.get(r, 0) + 1
        return {
            "total_analyses": len(entries),
            "total_predictions": len(predictions),
            "total_rejections": len(rejections),
            "rejection_rate": round(len(rejections) / max(len(entries), 1) * 100, 1),
            "class_distribution": class_counts,
            "avg_confidence": round(
                sum(e["confidence"] for e in predictions if e.get("confidence")) /
                max(len(predictions), 1), 3
            ),
        }


# Singleton
audit_service = AuditService()