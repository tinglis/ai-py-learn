import test from 'node:test';
import assert from 'node:assert/strict';

import { calculateFireNumber, calculateCoastFireNumber } from '../src/lib/fire.js';

test('calculateFireNumber basic case', () => {
  assert.equal(calculateFireNumber(40000, 0.04), 1000000);
});

test('calculateFireNumber zero swr', () => {
  assert.equal(calculateFireNumber(40000, 0), Infinity);
});

test('calculateFireNumber zero spending', () => {
  assert.equal(calculateFireNumber(0, 0.04), 0);
});

test('calculateCoastFireNumber with growth', () => {
  const result = calculateCoastFireNumber({ fireNumber: 1000000, yearsToGrow: 20, inflationAdjustedGrowthRate: 0.05 });
  assert.equal(Number(result.toFixed(2)), 376889.48);
});

test('calculateCoastFireNumber zero growth', () => {
  const result = calculateCoastFireNumber({ fireNumber: 1000000, yearsToGrow: 20, inflationAdjustedGrowthRate: 0 });
  assert.equal(result, 1000000);
});
