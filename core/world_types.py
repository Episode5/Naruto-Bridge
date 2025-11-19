"""Dataclasses for the Naruto-Bridge world randomization engine."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class Unit:
    """A single lexical element discovered while traveling."""

    unit_id: int
    lexeme: str
    origin: str = "lexicon"
    timestamp: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Unit":
        return cls(**data)


@dataclass
class Zone:
    """A cluster of units with an emergent identity."""

    zone_id: int
    target_units: int
    units: List[Unit] = field(default_factory=list)
    name: Optional[str] = None
    region_id: Optional[int] = None
    status: str = "forming"  # forming | sealed

    def to_dict(self) -> Dict:
        return {
            "zone_id": self.zone_id,
            "target_units": self.target_units,
            "units": [unit.to_dict() for unit in self.units],
            "name": self.name,
            "region_id": self.region_id,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Zone":
        units = [Unit.from_dict(u) for u in data.get("units", [])]
        return cls(
            zone_id=data["zone_id"],
            target_units=data.get("target_units", len(units)),
            units=units,
            name=data.get("name"),
            region_id=data.get("region_id"),
            status=data.get("status", "forming"),
        )


@dataclass
class Region:
    """A grouping of zones influenced by mined tokens."""

    region_id: int
    target_zones: int
    zone_ids: List[int] = field(default_factory=list)
    name: Optional[str] = None
    tokens_used: List[str] = field(default_factory=list)
    status: str = "forming"  # forming | sealed

    def to_dict(self) -> Dict:
        return {
            "region_id": self.region_id,
            "target_zones": self.target_zones,
            "zone_ids": list(self.zone_ids),
            "name": self.name,
            "tokens_used": list(self.tokens_used),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Region":
        return cls(
            region_id=data["region_id"],
            target_zones=data.get("target_zones", len(data.get("zone_ids", []))),
            zone_ids=list(data.get("zone_ids", [])),
            name=data.get("name"),
            tokens_used=list(data.get("tokens_used", [])),
            status=data.get("status", "forming"),
        )
