import { loadBenefitsConfig } from '../utils/configLoader.js';

const DEFAULT_TAX_YEAR = 2024;

function getConfig(taxYear = DEFAULT_TAX_YEAR) {
  return loadBenefitsConfig(taxYear).rdsp;
}

function roundToCents(value) {
  return Math.round((value + Number.EPSILON) * 100) / 100;
}

export function calculateRDSPGrants({ contribution = 0, familyIncome = 0, taxYear = DEFAULT_TAX_YEAR } = {}) {
  const config = getConfig(taxYear);
  if (contribution <= 0) {
    return { grant: 0 };
  }
  const [low, middle] = config.grantThresholds;
  if (familyIncome <= low.income) {
    const caps = config.grantContributionCaps;
    const firstPortion = Math.min(contribution, caps[0]);
    const secondPortion = Math.min(Math.max(contribution - caps[0], 0), caps[1]);
    const grant = firstPortion * low.rates[0] + secondPortion * low.rates[1];
    return { grant: roundToCents(grant) };
  }
  if (familyIncome <= middle.income) {
    const grant = Math.min(contribution, 1000) * middle.rates[0];
    return { grant: roundToCents(grant) };
  }
  return { grant: 0 };
}

export function calculateRDSPBonds({ familyIncome = 0, taxYear = DEFAULT_TAX_YEAR } = {}) {
  const config = getConfig(taxYear);
  const { maxAnnual, maxIncome, phaseOutIncome } = config.bond;
  if (familyIncome <= maxIncome) {
    return { bond: roundToCents(maxAnnual) };
  }
  if (familyIncome <= phaseOutIncome) {
    const ratio = (phaseOutIncome - familyIncome) / (phaseOutIncome - maxIncome);
    return { bond: roundToCents(maxAnnual * Math.max(0, ratio)) };
  }
  return { bond: 0 };
}

export function calculateRDSPClawback({
  withdrawal = 0,
  grants = 0,
  bonds = 0,
  totalAssistance = 0,
} = {}) {
  if (withdrawal <= 0) {
    return 0;
  }
  const pool = Math.max(totalAssistance, grants + bonds);
  if (pool <= 0) {
    return 0;
  }
  const proportion = Math.min(1, withdrawal / Math.max(withdrawal, totalAssistance || withdrawal));
  return roundToCents(pool * proportion);
}
