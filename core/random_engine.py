"""World randomization engine for Naruto-Bridge."""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path
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
