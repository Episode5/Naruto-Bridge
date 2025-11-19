"""Dataclasses for the Naruto-Bridge world randomization engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Unit:
    """A single lexical element discovered while traveling."""

    id: int
    word: str
    zone_id: Optional[int] = None
    origin: str = "lexicon"
    timestamp: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "word": self.word,
            "zone_id": self.zone_id,
            "origin": self.origin,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Unit":
        return cls(
            id=data["id"],
            word=data.get("word") or data.get("lexeme", ""),
            zone_id=data.get("zone_id"),
            origin=data.get("origin", "lexicon"),
            timestamp=data.get("timestamp", 0.0),
        )


@dataclass
class Zone:
    """A cluster of units with an emergent identity."""

    id: int
    unit_ids: List[int] = field(default_factory=list)
    name: Optional[str] = None
    region_id: Optional[int] = None
    status: str = "forming"  # forming | sealed
    meta: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "unit_ids": list(self.unit_ids),
            "name": self.name,
            "region_id": self.region_id,
            "status": self.status,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Zone":
        return cls(
            id=data.get("id") or data.get("zone_id"),
            unit_ids=list(data.get("unit_ids", [])),
            name=data.get("name"),
            region_id=data.get("region_id"),
            status=data.get("status", "forming"),
            meta=dict(data.get("meta", {})),
        )


@dataclass
class Region:
    """A grouping of zones influenced by mined tokens."""

    id: int
    zone_ids: List[int] = field(default_factory=list)
    name: Optional[str] = None
    status: str = "forming"  # forming | sealed
    meta: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "zone_ids": list(self.zone_ids),
            "name": self.name,
            "status": self.status,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Region":
        return cls(
            id=data.get("id") or data.get("region_id"),
            zone_ids=list(data.get("zone_ids", [])),
            name=data.get("name"),
            status=data.get("status", "forming"),
            meta=dict(data.get("meta", {})),
        )
