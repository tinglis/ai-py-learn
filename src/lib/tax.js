import { loadFederalConfig, loadProvincialConfig } from '../utils/configLoader.js';
import { calculateBracketTax } from '../utils/taxHelpers.js';

const DEFAULT_TAX_YEAR = 2024;

function roundToCents(value) {
  return Math.round((value + Number.EPSILON) * 100) / 100;
}

export function calculateFederalTax(income, { taxYear = DEFAULT_TAX_YEAR, includePersonalAmount = true } = {}) {
  const config = loadFederalConfig(taxYear);
  const basicPersonalAmount = includePersonalAmount ? config.basicPersonalAmount ?? 0 : 0;
  const tax = calculateBracketTax(income, config.brackets, basicPersonalAmount);
  return roundToCents(tax);
}

export function calculateProvincialTax(income, province, { taxYear = DEFAULT_TAX_YEAR, includePersonalAmount = true } = {}) {
  const config = loadProvincialConfig(taxYear, province);
  const basicPersonalAmount = includePersonalAmount ? config.basicPersonalAmount ?? 0 : 0;
  const tax = calculateBracketTax(income, config.brackets, basicPersonalAmount);
  return roundToCents(tax);
}

export function calculateTotalTax(income, province, options = {}) {
  const taxYear = options.taxYear ?? DEFAULT_TAX_YEAR;
  const includePersonalAmount = options.includePersonalAmount ?? true;
  const federal = calculateFederalTax(income, { taxYear, includePersonalAmount });
  const provincial = calculateProvincialTax(income, province, { taxYear, includePersonalAmount });
  return roundToCents(federal + provincial);
}
