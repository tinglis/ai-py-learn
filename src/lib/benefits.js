import { loadBenefitsConfig } from '../utils/configLoader.js';

const DEFAULT_TAX_YEAR = 2024;

function roundToCents(value) {
  return Math.round((value + Number.EPSILON) * 100) / 100;
}

function getConfig(taxYear = DEFAULT_TAX_YEAR) {
  return loadBenefitsConfig(taxYear);
}

export function calculateOASClawback(income, { taxYear = DEFAULT_TAX_YEAR } = {}) {
  const config = getConfig(taxYear);
  const { clawbackThreshold, clawbackRate, maxAnnual } = config.oas;
  if (income <= clawbackThreshold) {
    return 0;
  }
  const clawback = Math.min(maxAnnual, (income - clawbackThreshold) * clawbackRate);
  return roundToCents(clawback);
}

export function getOASAnnualAmount({ taxYear = DEFAULT_TAX_YEAR } = {}) {
  const config = getConfig(taxYear);
  return config.oas.maxAnnual;
}

export function calculateCPP(earnings, { taxYear = DEFAULT_TAX_YEAR } = {}) {
  const config = getConfig(taxYear);
  const { ympe, basicExemption, employeeRate, maxContribution } = config.cpp;
  const pensionable = Math.max(0, Math.min(earnings, ympe) - basicExemption);
  const contribution = Math.min(maxContribution, pensionable * employeeRate);
  return roundToCents(contribution);
}

export function calculateEI(earnings, { taxYear = DEFAULT_TAX_YEAR } = {}) {
  const config = getConfig(taxYear);
  const { maxInsurableEarnings, rate, maxContribution } = config.ei;
  const insurable = Math.min(earnings, maxInsurableEarnings);
  const contribution = Math.min(maxContribution, insurable * rate);
  return roundToCents(contribution);
}

export function getMaxCPPContribution({ taxYear = DEFAULT_TAX_YEAR } = {}) {
  const config = getConfig(taxYear);
  return config.cpp.maxContribution;
}

export function calculateGIS({
  income = 0,
  cppIncome = 0,
  otherIncome = 0,
  lastYearIncome,
  taxYear = DEFAULT_TAX_YEAR,
} = {}) {
  const config = getConfig(taxYear);
  const { maxAnnual, incomeThreshold, clawbackRate } = config.gis.single;
  const testIncome =
    lastYearIncome ?? Math.max(0, income + cppIncome + Math.max(0, otherIncome));
  if (testIncome <= 0) {
    return roundToCents(maxAnnual);
  }
  if (testIncome <= incomeThreshold) {
    return roundToCents(maxAnnual);
  }
  const clawback = (testIncome - incomeThreshold) * clawbackRate;
  const benefit = Math.max(0, maxAnnual - clawback);
  return roundToCents(benefit);
}

export function getRrifFactor(age, { taxYear = DEFAULT_TAX_YEAR } = {}) {
  const config = getConfig(taxYear);
  const { factors } = config.rrif;
  const key = String(age);
  if (factors[key] != null) {
    return factors[key];
  }
  const sorted = Object.keys(factors)
    .map(Number)
    .sort((a, b) => a - b);
  let applicable = sorted[sorted.length - 1];
  for (const candidate of sorted) {
    if (candidate <= age) {
      applicable = candidate;
    }
  }
  return factors[String(applicable)] ?? 0.05;
}
