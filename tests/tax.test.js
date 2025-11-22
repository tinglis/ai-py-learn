import test from 'node:test';
import assert from 'node:assert/strict';

import { calculateFederalTax, calculateProvincialTax, calculateTotalTax } from '../src/lib/tax.js';

const options = { taxYear: 2024, includePersonalAmount: false };
const TOLERANCE = 0.01;

function approxEqual(actual, expected) {
  assert.ok(Math.abs(actual - expected) <= TOLERANCE, `expected ${expected}, received ${actual}`);
}

test('federal tax bracket 1', () => {
  approxEqual(calculateFederalTax(30000, options), 4500);
});

test('federal tax bracket 2', () => {
  approxEqual(calculateFederalTax(60000, options), 9227.31);
});

test('federal tax bracket 3', () => {
  approxEqual(calculateFederalTax(120000, options), 21982);
});

test('federal tax zero income', () => {
  approxEqual(calculateFederalTax(0, options), 0);
});

test('BC tax bracket 1', () => {
  approxEqual(calculateProvincialTax(40000, 'BC', options), 2024);
});

test('BC tax bracket 2', () => {
  approxEqual(calculateProvincialTax(100000, 'BC', options), 6549.96);
});

test('ON tax bracket 1', () => {
  approxEqual(calculateProvincialTax(40000, 'ON', options), 2020);
});

test('ON tax bracket 2', () => {
  approxEqual(calculateProvincialTax(100000, 'ON', options), 7040.71);
});

test('QC tax bracket 1', () => {
  approxEqual(calculateProvincialTax(40000, 'QC', options), 5600);
});

test('AB tax bracket 1', () => {
  approxEqual(calculateProvincialTax(100000, 'AB', options), 10000);
});

test('total tax calculation', () => {
  approxEqual(calculateTotalTax(60000, 'BC', options), 12581.77);
});
