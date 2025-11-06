export function calculateFireNumber(spending, swr) {
  if (swr === 0) {
    return spending === 0 ? 0 : Infinity;
  }
  return spending / swr;
}

export function calculateCoastFireNumber({ fireNumber, yearsToGrow, inflationAdjustedGrowthRate }) {
  if (yearsToGrow <= 0) {
    return fireNumber;
  }
  if (inflationAdjustedGrowthRate === 0) {
    return fireNumber;
  }
  const growthFactor = (1 + inflationAdjustedGrowthRate) ** yearsToGrow;
  return fireNumber / growthFactor;
}
