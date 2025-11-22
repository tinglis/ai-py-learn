import test from 'node:test';
import assert from 'node:assert/strict';
import {
  calculateTaxableIncome,
  calculateNetEarnings,
  calculateDeductions,
  solveNetEarningsLoop,
  MAX_CPP_PREMIUM,
} from '../src/lib/income.js';
import { allocateSavings } from '../src/lib/savings.js';

const PROVINCE = 'BC';

function simpleAllocator(total, { priority }) {
  return allocateSavings(total, {
    priority,
    limits: {
      tfsaRoom: Number.POSITIVE_INFINITY,
      rrspRoom: Number.POSITIVE_INFINITY,
      rdspRoom: Number.POSITIVE_INFINITY,
      fhsaRoom: Number.POSITIVE_INFINITY,
    },
  });
}

test('Taxable income reflects RRSP deduction', () => {
  assert.equal(
    calculateTaxableIncome({ gross: 70000, rrspCont: 10000 }),
    60000,
  );
});

test('Capital gains inclusion rate respected', () => {
  assert.equal(
    calculateTaxableIncome({ gross: 50000, capitalGains: 10000 }),
    55000,
  );
});

test('TFSA withdrawals excluded from taxable income', () => {
  assert.equal(
    calculateTaxableIncome({ gross: 50000, tfsaWithdrawal: 20000 }),
    50000,
  );
});

test('Net earnings lower than gross after deductions', () => {
  const result = calculateNetEarnings({ gross: 60000, province: PROVINCE });
  assert.ok(result.netEarnings < 48000);
});

test('RRSP contribution increases net earnings via refund', () => {
  const withRRSP = calculateNetEarnings({ gross: 60000, rrsp: 10000, province: PROVINCE });
  const withoutRRSP = calculateNetEarnings({ gross: 60000, province: PROVINCE });
  assert.ok(withRRSP.netEarnings > withoutRRSP.netEarnings);
  assert.ok(withRRSP.refund > 1500);
});

test('Disability tax credit raises net earnings', () => {
  const base = calculateNetEarnings({ gross: 60000, province: PROVINCE });
  const enhanced = calculateNetEarnings({ gross: 60000, province: PROVINCE, credits: ['DTC'] });
  assert.ok(enhanced.netEarnings > base.netEarnings);
});

test('CPP contribution capped at statutory maximum', () => {
  const { cpp } = calculateDeductions(200000);
  assert.equal(cpp, MAX_CPP_PREMIUM);
});

test('Net earnings loop converges for RRSP priority', () => {
  const result = solveNetEarningsLoop({
    gross: 80000,
    savingsRate: 0.2,
    priority: ['RRSP'],
    province: 'ON',
    allocator: (total, ctx) => simpleAllocator(total, ctx),
  });
  assert.ok(result.stable);
  assert.ok(result.rrspCont > 10000);
});

test('Net earnings loop handles TFSA priority', () => {
  const result = solveNetEarningsLoop({
    gross: 80000,
    savingsRate: 0.2,
    priority: ['TFSA'],
    province: 'ON',
    allocator: (total, ctx) => simpleAllocator(total, ctx),
  });
  assert.equal(result.contributions.rrsp ?? 0, 0);
});
