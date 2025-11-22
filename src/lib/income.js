import { calculateTotalTax } from './tax.js';
import { calculateCPP, calculateEI, getMaxCPPContribution } from './benefits.js';

const DEFAULT_TAX_YEAR = 2024;
const CAPITAL_GAINS_INCLUSION_RATE = 0.5;
const CREDIT_VALUES = {
  DTC: 1500,
};

function roundToCents(value) {
  return Math.round((value + Number.EPSILON) * 100) / 100;
}

function resolveCredits(credits = []) {
  if (!Array.isArray(credits)) {
    return Object.values(credits ?? {}).reduce((sum, value) => sum + Number(value || 0), 0);
  }
  return credits.reduce((sum, credit) => sum + (CREDIT_VALUES[credit] ?? 0), 0);
}

export function calculateTaxableIncome({
  gross = 0,
  rrspCont = 0,
  rrspWithdrawal = 0,
  capitalGains = 0,
  tfsaWithdrawal = 0,
  otherIncome = 0,
  capitalGainsInclusionRate = CAPITAL_GAINS_INCLUSION_RATE,
} = {}) {
  const taxableCapitalGains = Math.max(0, capitalGains) * capitalGainsInclusionRate;
  const taxable = gross - rrspCont + rrspWithdrawal + taxableCapitalGains + Math.max(0, otherIncome);
  return Math.max(0, taxable);
}

export function calculateDeductions(gross, { taxYear = DEFAULT_TAX_YEAR } = {}) {
  const cpp = calculateCPP(gross, { taxYear });
  const ei = calculateEI(gross, { taxYear });
  return {
    cpp,
    ei,
    total: roundToCents(cpp + ei),
  };
}

export function calculateNetEarnings({
  gross = 0,
  rrsp = 0,
  credits = [],
  province,
  taxYear = DEFAULT_TAX_YEAR,
} = {}) {
  if (!province) {
    throw new Error('Province is required to calculate net earnings');
  }
  const deductions = calculateDeductions(gross, { taxYear });
  const taxableBeforeRRSP = calculateTaxableIncome({ gross });
  const taxBeforeRRSP = calculateTotalTax(taxableBeforeRRSP, province, { taxYear });
  const taxableAfterRRSP = calculateTaxableIncome({ gross, rrspCont: rrsp });
  const baseTax = calculateTotalTax(taxableAfterRRSP, province, { taxYear });
  const creditValue = resolveCredits(credits);
  const netTax = Math.max(0, roundToCents(baseTax - creditValue));
  const netEarnings = roundToCents(gross - deductions.total - netTax);
  const refund = roundToCents(taxBeforeRRSP - netTax);
  return {
    netEarnings,
    net: netEarnings,
    netTax,
    deductions,
    refund,
    taxableIncome: taxableAfterRRSP,
    taxBeforeRRSP,
  };
}

export function solveNetEarningsLoop({
  gross,
  savingsRate,
  priority = [],
  province,
  taxYear = DEFAULT_TAX_YEAR,
  allocator,
  maxIterations = 10,
} = {}) {
  if (typeof allocator !== 'function') {
    throw new Error('Allocator function is required for solveNetEarningsLoop');
  }
  let rrspContribution = 0;
  let stable = false;
  let finalNet = 0;
  let contributions = {};
  for (let i = 0; i < maxIterations; i += 1) {
    const { netEarnings } = calculateNetEarnings({
      gross,
      rrsp: rrspContribution,
      province,
      taxYear,
    });
    finalNet = netEarnings;
    const totalSavings = netEarnings * savingsRate;
    contributions = allocator(totalSavings, { priority });
    const nextRRSP = contributions.rrsp ?? 0;
    if (Math.abs(nextRRSP - rrspContribution) < 1) {
      stable = true;
      rrspContribution = nextRRSP;
      break;
    }
    rrspContribution = nextRRSP;
  }
  return {
    stable,
    rrspCont: rrspContribution,
    contributions,
    netEarnings: finalNet,
  };
}

export const MAX_CPP_PREMIUM = getMaxCPPContribution();
