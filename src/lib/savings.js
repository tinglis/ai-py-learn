const ACCOUNT_KEYS = {
  TFSA: 'tfsa',
  RRSP: 'rrsp',
  RDSP: 'rdsp',
  FHSA: 'fhsa',
  'NON-REGISTERED': 'nonRegistered',
  NONREGISTERED: 'nonRegistered',
  NONREG: 'nonRegistered',
};

function normaliseKey(name = '') {
  const upper = name.toUpperCase();
  for (const [label, key] of Object.entries(ACCOUNT_KEYS)) {
    if (upper === label) {
      return key;
    }
  }
  return upper.toLowerCase();
}

export function syncSavings({ net, percent, dollarRatios = [] } = {}) {
  const total = net * percent;
  const ratios = dollarRatios.length > 0 ? dollarRatios : [{ acc: 'TFSA', ratio: 1 }];
  const allocations = {};
  let ratioSum = 0;
  for (const entry of ratios) {
    ratioSum += entry.ratio;
  }
  for (const entry of ratios) {
    const key = normaliseKey(entry.acc);
    allocations[key] = total * (entry.ratio / ratioSum);
  }
  return {
    total,
    dollars: allocations,
  };
}

export function allocateSavings(total, { priority = [], limits = {} } = {}) {
  let remaining = total;
  const allocations = {};
  for (const slot of priority) {
    if (remaining <= 0) {
      break;
    }
    const key = normaliseKey(slot);
    const limitKey = `${key}Room`;
    const available = limits[limitKey] ?? Number.POSITIVE_INFINITY;
    const already = allocations[key] ?? 0;
    const roomLeft = Math.max(0, available - already);
    const contribution = Math.min(remaining, roomLeft || remaining);
    if (contribution > 0) {
      allocations[key] = already + contribution;
      remaining -= contribution;
    }
  }
  if (remaining > 0) {
    allocations.unallocated = (allocations.unallocated ?? 0) + remaining;
  }
  return allocations;
}
