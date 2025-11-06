import test from 'node:test';
import assert from 'node:assert/strict';
import {
  calculateOASClawback,
  getOASAnnualAmount,
  calculateCPP,
  calculateGIS,
} from '../src/lib/benefits.js';

const TAX_YEAR = 2024;

test('OAS clawback triggers above threshold', () => {
  const clawback = calculateOASClawback(91000, { taxYear: TAX_YEAR });
  assert.ok(clawback > 0);
});

test('OAS clawback zero below threshold', () => {
  const clawback = calculateOASClawback(80000, { taxYear: TAX_YEAR });
  assert.equal(clawback, 0);
});

test('Full OAS clawback equals annual amount', () => {
  const full = getOASAnnualAmount({ taxYear: TAX_YEAR });
  const clawback = calculateOASClawback(200000, { taxYear: TAX_YEAR });
  assert.equal(clawback, full);
});

test('CPP contribution positive for middle income', () => {
  const cpp = calculateCPP(60000, { taxYear: TAX_YEAR });
  assert.ok(cpp > 0);
});

test('GIS reduced to zero when income exceeds threshold', () => {
  const gis = calculateGIS({ income: 30000, cppIncome: 20000, taxYear: TAX_YEAR });
  assert.equal(gis, 0);
});
