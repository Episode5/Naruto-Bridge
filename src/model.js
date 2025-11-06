function createNinjaModel(raw) { 
	return {
		name: raw.name,
		rank: raw.rank,
		mainSkill: raw.skills?.[0] ?? null,
		chakra: raw.stats?.chakra ?? null,
	};
}

module.exports = { createNinjaModel };
