import test from 'node:test';
import assert from 'node:assert/strict';
import { syncSavings, allocateSavings } from '../src/lib/savings.js';

test('Sync savings distributes by ratios', () => {
  const result = syncSavings({
    net: 50000,
    percent: 0.2,
    dollarRatios: [
      { acc: 'TFSA', ratio: 0.5 },
      { acc: 'RRSP', ratio: 0.5 },
    ],
  });
  assert.equal(result.dollars.tfsa, 5000);
  assert.equal(result.dollars.rrsp, 5000);
});

test('Allocate savings respects TFSA room cap', () => {
  const allocation = allocateSavings(20000, {
    priority: ['TFSA', 'RRSP'],
    limits: { tfsaRoom: 5000, rrspRoom: 10000 },
  });
  assert.equal(allocation.tfsa, 5000);
  assert.equal(allocation.rrsp, 10000);
  assert.ok(allocation.unallocated > 0);
});

test('Allocate savings defaults to non-registered for leftovers', () => {
  const allocation = allocateSavings(5000, {
    priority: ['NON-REGISTERED'],
  });
  assert.equal(allocation.nonRegistered, 5000);
});
