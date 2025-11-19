"""World randomization engine for Naruto-Bridge."""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .world_types import Region, Unit, Zone


class RandomizationEngine:
    """Creates units, zones, and regions based on the tuning config."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        *,
        base_dir: Optional[Path] = None,
    ) -> None:
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.config_path = config_path or self.base_dir / "data" / "tuning" / "world_config.json"
        self.config = self._load_config()
        self.lexicon_path = self._resolve_path(self.config.get("paths", {}).get("lexicon"))
        self.default_state_path = self._resolve_path(self.config.get("paths", {}).get("world_state"))

        seed = self.config.get("rng_seed")
        self.rng = random.Random(seed)

        self.base_words = self._load_lexicon()

        self.units: Dict[int, Unit] = {}
        self.zones: Dict[int, Zone] = {}
        self.regions: Dict[int, Region] = {}

        self.unit_counter = 0
        self.zone_counter = 0
        self.region_counter = 0

        self.mined_token_words: List[str] = []

        if self.default_state_path:
            self.load_state(self.default_state_path)
        self.paths = self._resolve_paths()
        self.lexicon = self._load_lexicon()
        seed = self.config.get("rng_seed")
        self.rng = random.Random(seed)
        self.autosave = self.config.get("persistence", {}).get("autosave", True)

        self.next_ids = {"unit": 1, "zone": 1, "region": 1}
        self.zones: List[Zone] = []
        self.regions: List[Region] = []

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def travel_units(self, steps: int = 1) -> List[Unit]:
        """Advance the engine by a number of units, creating topology as needed."""

        created: List[Unit] = []
        if steps < 1:
            return created

        for _ in range(steps):
            region = self._maybe_create_region()
            zone = self._maybe_create_zone(region)
            unit = self._create_unit(zone.id)
            created.append(unit)

            self.units[unit.id] = unit
            zone.unit_ids.append(unit.id)

            target_units = zone.meta.get("target_units")
            if target_units is not None and len(zone.unit_ids) >= target_units:
                self._seal_zone(zone)
                if zone.id not in region.zone_ids:
                    region.zone_ids.append(zone.id)
                self._maybe_form_region(region)

        return created

    def get_current_zone(self):
        """Return the most recent zone, or None if no zones exist."""
        if not self.zones:
            return None
        return self.zones[max(self.zones.keys())]

    def get_current_region(self):
        """Return the most recent region, or None if no regions exist."""
        if not self.regions:
            return None
        return self.regions[max(self.regions.keys())]

    def snapshot(self):
        """Return a serializable snapshot of world state for persistence or debugging."""
        return {
            "units": {uid: {"word": u.word, "zone_id": u.zone_id} for uid, u in self.units.items()},
            "zones": {
                zid: {
                    "name": z.name,
                    "unit_ids": z.unit_ids,
                    "region_id": z.region_id,
                    "meta": z.meta,
                }
                for zid, z in self.zones.items()
            },
            "regions": {
                rid: {
                    "name": r.name,
                    "zone_ids": r.zone_ids,
                    "meta": r.meta,
                }
                for rid, r in self.regions.items()
            },
        }

    def save_state(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "units": [
                {"id": u.id, "word": u.word, "zone_id": u.zone_id}
                for u in self.units.values()
            ],
            "zones": [
                {
                    "id": z.id,
                    "unit_ids": z.unit_ids,
                    "name": z.name,
                    "region_id": z.region_id,
                    "meta": z.meta,
                }
                for z in self.zones.values()
            ],
            "regions": [
                {
                    "id": r.id,
                    "zone_ids": r.zone_ids,
                    "name": r.name,
                    "meta": r.meta,
                }
                for r in self.regions.values()
            ],
            "counters": {
                "unit_counter": self.unit_counter,
                "zone_counter": self.zone_counter,
                "region_counter": self.region_counter,
            },
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_state(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            return  # no state yet

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.units.clear()
        self.zones.clear()
        self.regions.clear()

        for u in data.get("units", []):
            unit = Unit(
                id=u["id"],
                word=u["word"],
                zone_id=u.get("zone_id"),
            )
            self.units[unit.id] = unit

        for z in data.get("zones", []):
            zone = Zone(
                id=z["id"],
                unit_ids=z.get("unit_ids", []),
                name=z.get("name", ""),
                region_id=z.get("region_id"),
                meta=z.get("meta", {}) or {},
                status="sealed" if z.get("name") else z.get("status", "forming"),
            )
            self.zones[zone.id] = zone

        for r in data.get("regions", []):
            region = Region(
                id=r["id"],
                zone_ids=r.get("zone_ids", []),
                name=r.get("name", ""),
                status="sealed" if r.get("name") else r.get("status", "forming"),
                meta=r.get("meta", {}) or {},
            )
            self.regions[region.id] = region

        counters = data.get("counters", {})
        self.unit_counter = max(counters.get("unit_counter", 0), max(self.units.keys(), default=0))
        self.zone_counter = max(counters.get("zone_counter", 0), max(self.zones.keys(), default=0))
        self.region_counter = max(counters.get("region_counter", 0), max(self.regions.keys(), default=0))

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------
    def register_mined_token(self, token_obj: dict):
        """
        Accept a mining token (from tron_console) and extract meaningful words from its label/themes.
        """
        label = token_obj.get("label", "") or ""
        themes = token_obj.get("themes", []) or []

        raw_words: List[str] = []
        raw_words.extend(label.split())
        for t in themes:
            raw_words.extend(str(t).split())

        for w in raw_words:
            w_clean = "".join(ch for ch in w.lower() if ch.isalpha())
            if len(w_clean) >= 4:
                self.mined_token_words.append(w_clean)

    def _generate_zone_name(self, unit_ids: Sequence[int]) -> str:
        naming_cfg = self.config.get("naming", {})
        min_words = naming_cfg.get("zone_name_word_count_min", naming_cfg.get("zone", {}).get("min_words", 2))
        max_words = naming_cfg.get("zone_name_word_count_max", naming_cfg.get("zone", {}).get("max_words", 3))

        words = [self.units[uid].word for uid in unit_ids if uid in self.units]
        freq = Counter(words)
        common = [w for w, _ in freq.most_common(max_words)]

        if len(common) < min_words and self.base_words:
            needed = min_words - len(common)
            filler = self.rng.sample(self.base_words, min(needed, len(self.base_words)))
            common.extend(filler)

        chosen = common[:max_words]
        return " ".join(w.capitalize() for w in chosen)

    def _generate_region_name(self, zone_ids: Sequence[int]) -> str:
        naming_cfg = self.config.get("naming", {})
        min_words = naming_cfg.get("region_name_word_count_min", naming_cfg.get("region", {}).get("min_words", 2))
        max_words = naming_cfg.get("region_name_word_count_max", naming_cfg.get("region", {}).get("max_words", 4))
        use_roman = naming_cfg.get("region_use_roman_suffix", naming_cfg.get("region", {}).get("roman_suffix", True))

        local_words: List[str] = []
        for zid in zone_ids:
            zone = self.zones.get(zid)
            if not zone:
                continue
            for uid in zone.unit_ids:
                unit = self.units.get(uid)
                if unit:
                    local_words.append(unit.word)

        freq_local = Counter(local_words)
        freq_tokens = Counter(self.mined_token_words)

        merged = list(freq_local.most_common(max_words)) + list(freq_tokens.most_common(max_words))
        if not merged:
            candidates = self.base_words or ["region"]
            chosen = self.rng.sample(candidates, min(max_words, len(candidates)))
        else:
            seen = set()
            ordered: List[str] = []
            for word, _ in merged:
                if word not in seen:
                    seen.add(word)
                    ordered.append(word)
            chosen = ordered[:max_words]

        if len(chosen) < min_words and self.base_words:
            needed = min_words - len(chosen)
            chosen.extend(self.rng.sample(self.base_words, min(needed, len(self.base_words))))

        title_core = " ".join(w.capitalize() for w in chosen[:max_words])

        if use_roman:
            index = self.region_counter + 1
            romans = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
            suffix = romans[(index - 1) % len(romans)]
            return f"{title_core} {suffix}"

        return title_core

    # ------------------------------------------------------------------
    # Internal creation helpers
    # ------------------------------------------------------------------
    def _maybe_create_region(self) -> Region:
        region = self.get_current_region()
        if region and region.status != "sealed":
            if region.meta.get("target_zones") is None:
                region.meta["target_zones"] = self._sample_region_target()
            return region

        target = self._sample_region_target()
        region = Region(id=self._next_region_id(), zone_ids=[], status="forming", meta={"target_zones": target})
        self.regions[region.id] = region
        return region

    def _maybe_create_zone(self, region: Region) -> Zone:
        zone = self.get_current_zone()
        if zone and zone.status != "sealed":
            zone.region_id = region.id
            if zone.meta.get("target_units") is None:
                zone.meta["target_units"] = self._sample_zone_target()
            return zone

        target_units = self._sample_zone_target()
        zone = Zone(
            id=self._next_zone_id(),
            unit_ids=[],
            name=None,
            region_id=region.id,
            status="forming",
            meta={"target_units": target_units},
        )
        self.zones[zone.id] = zone
        if zone.id not in region.zone_ids:
            region.zone_ids.append(zone.id)
        return zone

    def _maybe_form_region(self, region: Region) -> None:
        target_zones = region.meta.get("target_zones")
        if target_zones is None:
            target_zones = self._sample_region_target()
            region.meta["target_zones"] = target_zones

        sealed_zone_ids = [zid for zid in region.zone_ids if self.zones.get(zid) and self.zones[zid].status == "sealed"]
        if len(sealed_zone_ids) >= target_zones and region.status != "sealed":
            self._seal_region(region, sealed_zone_ids)

    def _create_unit(self, zone_id: int) -> Unit:
        lexeme = self.rng.choice(self.base_words)
        unit = Unit(id=self._next_unit_id(), word=lexeme, zone_id=zone_id, timestamp=time.time())
        return unit

    def _seal_zone(self, zone: Zone) -> None:
        zone.name = self._generate_zone_name(zone.unit_ids)
        zone.status = "sealed"

    def _seal_region(self, region: Region, zone_ids: Iterable[int]) -> None:
        name = self._generate_region_name(zone_ids)
        region.name = name
        region.status = "sealed"

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _load_config(self) -> Dict:
        config_path = Path(self.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"World config not found at {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_path(self, raw: Optional[str]) -> Optional[Path]:
        if raw is None:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = self.base_dir / path
        return path

    def _load_lexicon(self) -> List[str]:
        if not self.lexicon_path or not self.lexicon_path.exists():
            raise FileNotFoundError("Lexicon file missing; ensure data/lexicon_base.txt exists.")
        with self.lexicon_path.open("r", encoding="utf-8") as f:
            words = [line.strip() for line in f.readlines() if line.strip()]
        if not words:
            raise ValueError("Lexicon file is empty.")
        return words

    def _sample_int_normal(self, mean: float, std: float, min_val: int, max_val: int) -> int:
        val = int(self.rng.gauss(mean, std)) if std > 0 else int(mean)
        return max(min_val, min(max_val, val))

    def _sample_zone_target(self) -> int:
        clusters = self.config.get("clusters", {})
        mean = clusters.get("target_units_per_zone_mean", clusters.get("units_per_zone", {}).get("mean", 20))
        std = clusters.get("target_units_per_zone_std", clusters.get("units_per_zone", {}).get("std_dev", 4))
        min_units = clusters.get("min_units_per_zone", clusters.get("units_per_zone", {}).get("min", 1))
        max_units = clusters.get("max_units_per_zone", clusters.get("units_per_zone", {}).get("max", max(1, int(mean))))
        return self._sample_int_normal(mean, std, min_units, max_units)

    def _sample_region_target(self) -> int:
        clusters = self.config.get("clusters", {})
        mean = clusters.get("target_zones_per_region_mean", clusters.get("zones_per_region", {}).get("mean", 8))
        std = clusters.get("target_zones_per_region_std", clusters.get("zones_per_region", {}).get("std_dev", 2))
        min_zones = clusters.get("min_zones_per_region", clusters.get("zones_per_region", {}).get("min", 1))
        max_zones = clusters.get("max_zones_per_region", clusters.get("zones_per_region", {}).get("max", max(1, int(mean))))
        return self._sample_int_normal(mean, std, min_zones, max_zones)

    def _next_unit_id(self) -> int:
        self.unit_counter += 1
        return self.unit_counter

    def _next_zone_id(self) -> int:
        self.zone_counter += 1
        return self.zone_counter

    def _next_region_id(self) -> int:
        self.region_counter += 1
        return self.region_counter

    def _zones_for_region(self, region_id: int) -> Iterable[Zone]:
        for zone in self.zones.values():
            if zone.region_id == region_id:
                yield zone
        if steps < 1:
            return []

        created: List[Unit] = []
        for _ in range(steps):
            region = self._ensure_region()
            zone = self._ensure_zone(region)
            unit = self._create_unit()
            zone.units.append(unit)
            created.append(unit)
            if len(zone.units) >= zone.target_units:
                self._seal_zone(zone)
                if zone.zone_id not in region.zone_ids:
                    region.zone_ids.append(zone.zone_id)
                if len(region.zone_ids) >= region.target_zones:
                    self._seal_region(region)

        if self.autosave:
            self.save_state()

        return created

    def get_current_zone(self) -> Optional[Zone]:
        zone = self.zones[-1] if self.zones else None
        if zone and zone.status != "sealed":
            return zone
        return None

    def get_current_region(self) -> Optional[Region]:
        region = self.regions[-1] if self.regions else None
        if region and region.status != "sealed":
            return region
        return None

    def snapshot(self) -> Dict:
        """Return a JSON-safe snapshot of the world state."""

        return {
            "next_ids": dict(self.next_ids),
            "zones": [zone.to_dict() for zone in self.zones],
            "regions": [region.to_dict() for region in self.regions],
        }

    def save_state(self) -> None:
        data = {
            "version": self.config.get("version"),
            "next_ids": dict(self.next_ids),
            "zones": [zone.to_dict() for zone in self.zones],
            "regions": [region.to_dict() for region in self.regions],
        }
        path = self.paths["world_state"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def _load_config(self) -> Dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"World config not found at {self.config_path}")
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def _resolve_paths(self) -> Dict[str, Path]:
        raw_paths = self.config.get("paths", {})
        resolved = {}
        for key, value in raw_paths.items():
            path = Path(value)
            if not path.is_absolute():
                path = self.base_dir / path
            resolved[key] = path
        return resolved

    def _load_lexicon(self) -> List[str]:
        lexicon_path = self.paths.get("lexicon")
        if not lexicon_path or not lexicon_path.exists():
            raise FileNotFoundError("Lexicon file missing; ensure data/lexicon_base.txt exists.")
        words = [line.strip() for line in lexicon_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not words:
            raise ValueError("Lexicon file is empty.")
        return words

    def _load_state(self) -> None:
        path = self.paths["world_state"]
        if not path.exists():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.next_ids.update(raw.get("next_ids", {}))
        self.zones = [Zone.from_dict(item) for item in raw.get("zones", [])]
        self.regions = [Region.from_dict(item) for item in raw.get("regions", [])]

    # ------------------------------------------------------------------
    # Creation helpers
    # ------------------------------------------------------------------
    def _ensure_region(self) -> Region:
        region = self.get_current_region()
        if region:
            return region
        target = self._sample_target(self.config["clusters"]["zones_per_region"])
        region = Region(region_id=self._next_id("region"), target_zones=target)
        self.regions.append(region)
        return region

    def _ensure_zone(self, region: Region) -> Zone:
        zone = self.get_current_zone()
        if zone:
            zone.region_id = region.region_id
            return zone
        target = self._sample_target(self.config["clusters"]["units_per_zone"])
        zone = Zone(zone_id=self._next_id("zone"), target_units=target, region_id=region.region_id)
        self.zones.append(zone)
        return zone

    def _create_unit(self) -> Unit:
        lexeme = self.rng.choice(self.lexicon)
        unit = Unit(unit_id=self._next_id("unit"), lexeme=lexeme, timestamp=time.time())
        return unit

    def _seal_zone(self, zone: Zone) -> None:
        zone.name = self._build_zone_name(zone)
        zone.status = "sealed"

    def _seal_region(self, region: Region) -> None:
        name, used_tokens = self._build_region_name(region)
        region.name = name
        region.tokens_used = used_tokens
        region.status = "sealed"

    # ------------------------------------------------------------------
    # Naming helpers
    # ------------------------------------------------------------------
    def _build_zone_name(self, zone: Zone) -> str:
        cfg = self.config.get("naming", {}).get("zone", {})
        min_words = max(1, cfg.get("min_words", 1))
        max_words = max(min_words, cfg.get("max_words", min_words))
        candidate_pool = max(max_words, cfg.get("candidate_pool", max_words))

        counter = Counter(unit.lexeme for unit in zone.units)
        candidates = counter.most_common(candidate_pool)
        if not candidates:
            return f"Zone-{zone.zone_id}"

        words = self._weighted_unique_words(candidates, self.rng.randint(min_words, max_words))
        return " ".join(self._format_word(word) for word in words)

    def _build_region_name(self, region: Region) -> Tuple[str, List[str]]:
        cfg = self.config.get("naming", {}).get("region", {})
        min_words = max(2, cfg.get("min_words", 2))
        max_words = max(min_words, cfg.get("max_words", min_words))
        token_weight = cfg.get("token_weight", 1.0)
        unit_weight = cfg.get("unit_weight", 1.0)

        unit_words: Counter = Counter()
        for zone in self._zones_for_region(region.region_id):
            unit_words.update(unit.lexeme for unit in zone.units)

        token_words = Counter()
        token_word_sources: Dict[str, List[str]] = {}
        for token in self._load_mined_tokens():
            label = token.get("label", "")
            for word in self._split_words(label):
                token_words[word] += token_weight
                token_word_sources.setdefault(word, []).append(token.get("token_id") or token.get("label", ""))
            for theme in token.get("themes", []) or []:
                for word in self._split_words(theme):
                    token_words[word] += token_weight
                    token_word_sources.setdefault(word, []).append(token.get("token_id") or theme)

        if not unit_words:
            unit_words.update({self.rng.choice(self.lexicon): 1.0})

        combined: List[tuple[str, float]] = []
        combined.extend((word, freq * unit_weight) for word, freq in unit_words.items())
        combined.extend((word, weight) for word, weight in token_words.items())

        word_count = self.rng.randint(min_words, max_words)
        words = self._weighted_unique_words(combined, word_count)
        name = " ".join(self._format_word(word) for word in words)

        tokens_used: List[str] = []
        for word in words:
            tokens_used.extend(token_word_sources.get(word.lower(), []))

        if cfg.get("roman_suffix", False) and len(region.zone_ids) >= cfg.get("roman_after_zones", 0):
            suffix = self._roman(region.region_id + cfg.get("roman_offset", 0))
            if suffix:
                name = f"{name} {suffix}"

        return name, [token for token in tokens_used if token]

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _weighted_unique_words(self, pairs: Sequence[Tuple[str, float]], count: int) -> List[str]:
        pool = list(pairs)
        selected: List[str] = []
        while pool and len(selected) < count:
            total = sum(max(weight, 0.01) for _, weight in pool)
            pick = self.rng.uniform(0, total)
            upto = 0.0
            for idx, (word, weight) in enumerate(pool):
                upto += max(weight, 0.01)
                if upto >= pick:
                    selected.append(word)
                    pool.pop(idx)
                    break
        if not selected:
            return ["untitled"]
        return selected

    def _sample_target(self, cfg: Dict[str, float]) -> int:
        mean = cfg.get("mean", 1)
        std_dev = cfg.get("std_dev", 0)
        size = int(round(self.rng.gauss(mean, std_dev))) if std_dev > 0 else int(mean)
        size = max(cfg.get("min", 1), size)
        max_value = cfg.get("max")
        if max_value is not None:
            size = min(max_value, size)
        return max(1, size)

    def _next_id(self, key: str) -> int:
        value = self.next_ids.get(key, 1)
        self.next_ids[key] = value + 1
        return value

    def _zones_for_region(self, region_id: int) -> Iterable[Zone]:
        for zone in self.zones:
            if zone.region_id == region_id:
                yield zone

    def _split_words(self, text: str) -> List[str]:
        cleaned = []
        for raw in text.replace("-", " ").replace("_", " ").split():
            word = "".join(ch for ch in raw if ch.isalpha())
            if word:
                cleaned.append(word.lower())
        return cleaned

    def _format_word(self, word: str) -> str:
        if not word:
            return word
        return word[0].upper() + word[1:]

    def _load_mined_tokens(self) -> List[Dict]:
        path = self.paths.get("mining_tokens")
        if not path or not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return payload.get("tokens", [])

    def _roman(self, value: int) -> str:
        if value <= 0 or value >= 4000:
            return ""
        numerals = [
            (1000, "M"),
            (900, "CM"),
            (500, "D"),
            (400, "CD"),
            (100, "C"),
            (90, "XC"),
            (50, "L"),
            (40, "XL"),
            (10, "X"),
            (9, "IX"),
            (5, "V"),
            (4, "IV"),
            (1, "I"),
        ]
        result = []
        remaining = value
        for integer, numeral in numerals:
            count = remaining // integer
            result.append(numeral * count)
            remaining -= integer * count
        return "".join(result)
