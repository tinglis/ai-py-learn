import test from 'node:test';
import assert from 'node:assert/strict';
import { runStartupChecks } from '../src/lib/startupChecks.js';

test('startup checks execute successfully', () => {
  const result = runStartupChecks();
  assert.ok(result.passed);
  assert.ok(result.count >= 10);
});
