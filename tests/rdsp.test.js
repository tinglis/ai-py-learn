import test from 'node:test';
import assert from 'node:assert/strict';
import {
  calculateRDSPGrants,
  calculateRDSPBonds,
  calculateRDSPClawback,
} from '../src/lib/rdsp.js';

const TAX_YEAR = 2024;

test('Low income RDSP grant includes enhanced match', () => {
  const { grant } = calculateRDSPGrants({ contribution: 1500, familyIncome: 30000, taxYear: TAX_YEAR });
  assert.equal(grant, 3500);
});

test('Mid income RDSP grant capped at contribution', () => {
  const { grant } = calculateRDSPGrants({ contribution: 1000, familyIncome: 80000, taxYear: TAX_YEAR });
  assert.equal(grant, 1000);
});

test('High income RDSP grant zero', () => {
  const { grant } = calculateRDSPGrants({ contribution: 1000, familyIncome: 120000, taxYear: TAX_YEAR });
  assert.equal(grant, 0);
});

test('Low income RDSP bond maximum awarded', () => {
  const { bond } = calculateRDSPBonds({ familyIncome: 30000, taxYear: TAX_YEAR });
  assert.equal(bond, 1000);
});

test('RDSP clawback triggered on withdrawal with recent grants', () => {
  const clawback = calculateRDSPClawback({ withdrawal: 1000, grants: 5000, totalAssistance: 5000 });
  assert.ok(clawback > 0);
});
