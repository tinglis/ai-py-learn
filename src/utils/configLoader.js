import fs from 'node:fs';
import path from 'node:path';

const CACHE = new Map();

function readJson(filePath) {
  const absolute = path.resolve(filePath);
  if (CACHE.has(absolute)) {
    return CACHE.get(absolute);
  }
  const data = JSON.parse(fs.readFileSync(absolute, 'utf-8'));
  CACHE.set(absolute, data);
  return data;
}

export function loadFederalConfig(taxYear) {
  const file = path.join('src', 'config', 'tax', String(taxYear), 'federal.json');
  return readJson(file);
}

export function loadProvincialConfig(taxYear, province) {
  const file = path.join('src', 'config', 'tax', String(taxYear), 'provinces.json');
  const provinces = readJson(file);
  const data = provinces[province];
  if (!data) {
    throw new Error(`Unknown province code: ${province}`);
  }
  return data;
}

export function loadBenefitsConfig(taxYear) {
  const file = path.join('src', 'config', 'benefits', String(taxYear), 'core.json');
  return readJson(file);
}
