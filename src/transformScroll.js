const { createNinjaModel } = require("./model");

function forwardTransform(raw) {
  return createNinjaModel(raw);
}

function reverseTransform(model) {
  return {
    ninja_name: model.name,
    ninja_rank: model.rank,
    primary_skill: model.mainSkill,
    chakra_level: model.chakra,
  };
}

module.exports = { forwardTransform, reverseTransform };

