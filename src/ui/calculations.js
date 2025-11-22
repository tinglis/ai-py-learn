import { calculateFireNumber, calculateCoastFireNumber } from '../lib/fire.js';
import { calculateNetEarnings, solveNetEarningsLoop } from '../lib/income.js';
import { allocateSavings } from '../lib/savings.js';
import {
  runFullProjection,
  runFullSimulation,
  runMonteCarloSimulation,
  getGraphData,
} from '../lib/projection.js';

const FREQ_MULTIPLIERS = {
  annually: 1,
  monthly: 12,
  semiMonthly: 24,
  weekly: 52,
};

function normaliseAmount(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function toAnnual(amount = 0, frequency = 'annually') {
  const multiplier = FREQ_MULTIPLIERS[frequency] ?? 1;
  return normaliseAmount(amount) * multiplier;
}

function computeDollarContributions(dollar = {}) {
  return {
    tfsa: toAnnual(dollar.tfsa?.amount, dollar.tfsa?.frequency),
    rrsp: toAnnual(dollar.rrsp?.amount, dollar.rrsp?.frequency),
    rdsp: toAnnual(dollar.rdsp?.amount, dollar.rdsp?.frequency),
    fhsa: toAnnual(dollar.fhsa?.amount, dollar.fhsa?.frequency),
    nonReg: toAnnual(dollar.nonReg?.amount, dollar.nonReg?.frequency),
  };
}

function sumContributions(contributions = {}) {
  return Object.entries(contributions).reduce((sum, [, value]) => sum + normaliseAmount(value), 0);
}

function extractContribution(contributions = {}, key) {
  return normaliseAmount(contributions[key] ?? contributions[key?.toLowerCase?.()] ?? 0);
}

function calcRealGrowth(preRet, inflation) {
  return (1 + preRet) / (1 + inflation) - 1;
}

function findFirstAge(portfolio = [], targetByAge = {}, startAge, endAge) {
  for (let age = startAge; age <= endAge; age += 1) {
    const row = portfolio[age];
    if (!row) continue;
    const threshold = targetByAge[age];
    if (threshold != null && row.balance >= threshold) {
      return age;
    }
  }
  return null;
}

function buildTargetSeries({ currentAge, inflation, base, startAge, endAge }) {
  const targets = {};
  for (let age = startAge; age <= endAge; age += 1) {
    const years = age - currentAge;
    targets[age] = base * Math.pow(1 + inflation, Math.max(0, years));
  }
  return targets;
}

export function buildDashboard(state) {
  const {
    profile,
    economy,
    savings,
    income,
    accounts,
    housing,
    benefits,
    retirement,
    limits,
    simulation,
  } = state;

  const limitsForAllocation = {
    tfsaRoom: normaliseAmount(accounts.tfsaRoom),
    rrspRoom: normaliseAmount(accounts.rrspRoom),
    rdspRoom: Number.isFinite(accounts.rdspRoom) ? accounts.rdspRoom : Number.POSITIVE_INFINITY,
    fhsaRoom: Number.isFinite(accounts.fhsaRoom) ? accounts.fhsaRoom : Number.POSITIVE_INFINITY,
  };

  let savingsTotal = 0;
  let contributions = {};
  let percentEquivalent = savings.percent;
  let netBreakdown = null;
  let rrspContribution = 0;

  if (savings.mode === 'percent') {
    const allocator = (total, { priority }) =>
      allocateSavings(total, { priority, limits: limitsForAllocation });
    const loop = solveNetEarningsLoop({
      gross: income.gross,
      savingsRate: savings.percent,
      priority: savings.priority,
      province: profile.province,
      allocator,
    });
    contributions = loop.contributions;
    savingsTotal = sumContributions(contributions) - normaliseAmount(contributions.unallocated);
    rrspContribution = extractContribution(contributions, 'rrsp');
    netBreakdown = calculateNetEarnings({
      gross: income.gross,
      rrsp: rrspContribution,
      province: profile.province,
      credits: income.credits,
    });
    percentEquivalent = savings.percent;
  } else {
    const annual = computeDollarContributions(savings.dollar);
    contributions = {
      tfsa: annual.tfsa,
      rrsp: annual.rrsp,
      rdsp: annual.rdsp,
      fhsa: annual.fhsa,
      nonRegistered: annual.nonReg,
    };
    savingsTotal = sumContributions(annual);
    rrspContribution = annual.rrsp;
    netBreakdown = calculateNetEarnings({
      gross: income.gross,
      rrsp: rrspContribution,
      province: profile.province,
      credits: income.credits,
    });
    percentEquivalent = netBreakdown.net > 0 ? savingsTotal / netBreakdown.net : 0;
  }

  const fireNumberToday = calculateFireNumber(retirement.spendingToday, economy.swr || 0.04);
  const coastGrowth = calcRealGrowth(economy.preRetReturn, economy.inflation);
  const yearsToGrow = Math.max(0, profile.fireAge - profile.coastAge);
  const coastNumberToday = calculateCoastFireNumber({
    fireNumber: fireNumberToday,
    yearsToGrow,
    inflationAdjustedGrowthRate: coastGrowth,
  });

  const savingsMode = savings.mode === 'dollar' ? 'Dollar' : 'Percent';
  const dollarAmounts = savingsMode === 'Dollar'
    ? {
        TFSA: contributions.tfsa,
        RRSP: contributions.rrsp,
        RDSP: contributions.rdsp,
        FHSA: contributions.fhsa,
        NonRegistered: contributions.nonRegistered,
      }
    : undefined;

  const projection = runFullProjection({
    currentAge: profile.currentAge,
    coastAge: profile.coastAge,
    fireAge: profile.fireAge,
    lifeExpectancy: profile.lifeExpectancy,
    savings: savingsTotal,
    savingsMode,
    dollarAmounts,
    priority: savings.priority,
    tfsaRoom: accounts.tfsaRoom,
    rrspRoom: accounts.rrspRoom,
    rdspRoom: accounts.rdspRoom,
    fhsaRoom: accounts.fhsaRoom,
    tfsa: accounts.tfsaBalance,
    rrsp: accounts.rrspBalance,
    nonReg: accounts.nonRegBalance,
    rdsp: accounts.rdspBalance,
    fhsa: accounts.fhsaBalance,
    acb: accounts.acb,
    income: income.gross,
    incomeGrowth: income.growth,
    preRetRate: economy.preRetReturn,
    postRetRate: economy.postRetReturn,
    dividendYield: economy.dividendYield,
    retirementSpending: retirement.spendingToday,
    inflation: economy.inflation,
    rrspMatch: income.rrspMatch,
    refundOption: savings.refundOption,
    province: profile.province,
    annualTFSALimit: limits.annualTfsa,
    rrspMaxCap: limits.rrspMax,
    cppAnnual: benefits.cppAnnual,
    cppStartAge: benefits.cppStartAge,
    oasAnnual: benefits.oasAnnual,
    oasStartAge: benefits.oasStartAge,
    gisThreshold: benefits.gisThreshold,
    homeValue: housing.status === 'owner' ? housing.homeValue : 0,
    homeAppr: economy.homeAppreciation,
    mortgageBalance: housing.status === 'owner' ? housing.mortgageBalance : 0,
    annualMortgagePayment: housing.annualMortgagePayment,
  });

  const targetSeries = buildTargetSeries({
    currentAge: profile.currentAge,
    inflation: economy.inflation,
    base: fireNumberToday,
    startAge: profile.currentAge,
    endAge: profile.lifeExpectancy,
  });

  const coastSeries = buildTargetSeries({
    currentAge: profile.coastAge,
    inflation: economy.inflation,
    base: coastNumberToday,
    startAge: profile.coastAge,
    endAge: profile.lifeExpectancy,
  });

  const projectedFireAge = findFirstAge(
    projection.portfolio,
    targetSeries,
    profile.currentAge,
    profile.lifeExpectancy,
  );

  const projectedCoastAge = findFirstAge(
    projection.portfolio,
    coastSeries,
    profile.currentAge,
    profile.fireAge,
  );

  const fireNumberNominalAtFireAge = targetSeries[profile.fireAge];

  const retirementYears = Math.max(0, profile.lifeExpectancy - profile.fireAge);
  const retirementStartBalance = projection.portfolio[profile.fireAge]?.balance ?? 0;

  let simulationResult;
  if (simulation.mode === 'monteCarlo') {
    simulationResult = {
      ...runMonteCarloSimulation({
        portfolio: retirementStartBalance,
        spending: retirement.spendingToday,
        years: retirementYears,
        expectedReturn: economy.postRetReturn,
        returnStdDev: simulation.returnStdDev,
        inflationMean: economy.inflation,
        inflationStdDev: simulation.inflationStdDev,
        paths: simulation.paths,
        seed: simulation.seed,
      }),
      mode: 'monteCarlo',
    };
  } else {
    simulationResult = runFullSimulation({
      portfolio: retirementStartBalance,
      spending: retirement.spendingToday,
      years: retirementYears,
      mode: 'historical',
    });
  }

  const graph = getGraphData({
    portfolio: projection.portfolio,
    homeValues: projection.homeValues,
    mortgage: projection.mortgage,
  });

  const incomeSeries = [];
  for (let age = profile.fireAge; age <= profile.lifeExpectancy; age += 1) {
    const row = projection.portfolio[age];
    if (!row) continue;
    incomeSeries.push({
      age,
      rrsp: row.rrspWithdrawal,
      tfsa: row.tfsaWithdrawal,
      nonReg: row.nonRegWithdrawal,
      cpp: row.cppIncome ?? 0,
      oas: row.oasIncome ?? 0,
      gis: row.gis ?? 0,
      spending: retirement.spendingToday * Math.pow(1 + economy.inflation, age - profile.fireAge),
    });
  }

  const net = netBreakdown ?? { net: 0, netTax: 0, deductions: { cpp: 0, ei: 0, total: 0 }, taxableIncome: 0, refund: 0 };

  return {
    projection,
    graph,
    netEarnings: {
      annual: net.net,
      monthly: net.net / 12,
      semiMonthly: net.net / 24,
      tax: net.netTax,
      taxableIncome: net.taxableIncome,
      cpp: net.deductions.cpp,
      ei: net.deductions.ei,
      deductionsTotal: net.deductions.total,
      rrspContribution,
      refund: net.refund,
    },
    savings: {
      total: savingsTotal,
      percentEquivalent,
      contributions,
      mode: savingsMode,
    },
    fire: {
      fireNumberToday,
      fireNumberAtFireAge: fireNumberNominalAtFireAge,
      coastNumberToday,
      projectedFireAge,
      projectedCoastAge,
    },
    simulation: simulationResult,
    incomeSeries,
  };
}
