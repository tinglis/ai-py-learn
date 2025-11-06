export function calculateBracketTax(income, brackets, basicPersonalAmount = 0) {
  const taxableIncome = Math.max(0, income - basicPersonalAmount);
  let previousLimit = 0;
  let tax = 0;

  for (const { limit, rate } of brackets) {
    const upper = limit ?? Number.POSITIVE_INFINITY;
    if (taxableIncome <= previousLimit) {
      break;
    }
    const taxablePortion = Math.min(upper, taxableIncome) - previousLimit;
    if (taxablePortion > 0) {
      tax += taxablePortion * rate;
    }
    previousLimit = upper;
    if (taxableIncome <= upper) {
      break;
    }
  }

  return tax;
}
