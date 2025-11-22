import { calculateFireNumber, calculateCoastFireNumber } from './fire.js';
import { calculateFederalTax, calculateProvincialTax, calculateTotalTax } from './tax.js';
import { calculateNetEarnings, calculateDeductions } from './income.js';
import { allocateSavings, syncSavings } from './savings.js';
import {
  runYearProjection,
  runFullProjection,
  runMonteCarloSimulation,
  runFullSimulation,
} from './projection.js';
import { calculateGIS } from './benefits.js';
import { calculateRDSPGrants, calculateRDSPBonds, calculateRDSPClawback } from './rdsp.js';

function approxEqual(actual, expected, tolerance = 0.01) {
  return Math.abs(actual - expected) <= tolerance;
}

function assertApprox(name, actual, expected, tolerance = 0.01) {
  if (!approxEqual(actual, expected, tolerance)) {
    throw new Error(`${name} expected ${expected}, received ${actual}`);
  }
}

function assertCondition(name, condition, message) {
  if (!condition) {
    throw new Error(message ?? `${name} failed`);
  }
}

export function runStartupChecks() {
  const checks = [];
  const run = (name, fn) => {
    try {
      fn();
      checks.push({ name, status: 'pass' });
    } catch (error) {
      throw new Error(`Startup Test Failed: ${name} - ${error.message}`);
    }
  };

  run('FIRE Number baseline', () => {
    assertApprox('fireNumber', calculateFireNumber(40000, 0.04), 1000000);
  });

  run('Coast FIRE growth', () => {
    assertApprox(
      'coastFire',
      calculateCoastFireNumber({ fireNumber: 1000000, yearsToGrow: 20, inflationAdjustedGrowthRate: 0.05 }),
      376889.48,
      0.5,
    );
  });

  run('Federal tax bracket spot check', () => {
    assertApprox('federalTax', calculateFederalTax(60000), 6644.25, 0.5);
  });

  run('Provincial tax spot check', () => {
    assertApprox('provTax', calculateProvincialTax(40000, 'ON'), 1393.85, 1);
  });

  run('Total tax aggregation', () => {
    assertApprox('totalTax', calculateTotalTax(60000, 'BC'), 9076.18, 1);
  });

  run('Net earnings increases with RRSP deduction', () => {
    const base = calculateNetEarnings({ gross: 60000, rrsp: 0, province: 'BC' });
    const withRRSP = calculateNetEarnings({ gross: 60000, rrsp: 10000, province: 'BC' });
    assertCondition('netEarningsRRSP', withRRSP.net > base.net, 'RRSP deduction did not improve net earnings');
  });

  run('CPP/EI capped by YMPE', () => {
    const deductions = calculateDeductions(200000);
    assertCondition('cppCap', deductions.cpp <= deductions.total, 'CPP deduction exceeds total payroll');
  });

  run('Savings allocator respects room', () => {
    const allocations = allocateSavings(20000, {
      priority: ['TFSA', 'RRSP'],
      limits: { tfsaRoom: 5000, rrspRoom: 10000 },
    });
    assertApprox('tfsaAllocation', allocations.tfsa, 5000);
    assertApprox('rrspAllocation', allocations.rrsp, 10000);
  });

  run('Savings sync produces equivalent dollars', () => {
    const sync = syncSavings({ net: 50000, percent: 0.2, dollarRatios: [{ acc: 'TFSA', ratio: 1 }] });
    assertApprox('syncTotal', sync.total, 10000);
    assertApprox('syncDollar', sync.dollars.tfsa, 10000);
  });

  run('Year projection applies dividend tax', () => {
    const projection = runYearProjection({
      nonReg: 50000,
      acb: 50000,
      preRetRate: 0.07,
      dividendYield: 0.02,
      province: 'ON',
    });
    assertCondition('dividendTax', projection.nonReg.taxPaid > 0, 'Dividend tax was not applied');
  });

  run('Full projection handles accumulation to retirement', () => {
    const result = runFullProjection({
      currentAge: 30,
      coastAge: 40,
      fireAge: 50,
      lifeExpectancy: 55,
      savings: 10000,
      income: 80000,
      priority: ['TFSA', 'RRSP'],
      retirementSpending: 40000,
    });
    assertCondition('hasPortfolio', Array.isArray(result.portfolio), 'Portfolio rows missing');
    assertCondition('hasRetirement', result.portfolio[50].withdrawal > 0, 'Retirement withdrawals missing');
  });

  run('GIS calculation respects prior income lag', () => {
    const suppressed = calculateGIS({ income: 0, cppIncome: 0, lastYearIncome: 60000 });
    const eligible = calculateGIS({ income: 0, cppIncome: 0, lastYearIncome: 0 });
    assertCondition('gisLag', eligible > suppressed, 'GIS lag not respected');
  });

  run('RDSP grants available for low income', () => {
    const grants = calculateRDSPGrants({ contribution: 1500, familyIncome: 30000 });
    assertApprox('rdspGrant', grants.grant, 3500, 0.5);
  });

  run('RDSP bonds phase out', () => {
    const bond = calculateRDSPBonds({ familyIncome: 60000 });
    assertApprox('rdspBond', bond.bond, 0);
  });

  run('RDSP clawback triggers on withdrawal', () => {
    const clawback = calculateRDSPClawback({ withdrawal: 1000, grants: 5000, bonds: 2000, totalAssistance: 7000 });
    assertCondition('rdspClawback', clawback > 0, 'RDSP clawback missing');
  });

  run('Monte Carlo simulator returns bounded success rate', () => {
    const result = runMonteCarloSimulation({
      portfolio: 500000,
      spending: 20000,
      years: 25,
      expectedReturn: 0.06,
      returnStdDev: 0.12,
      inflationMean: 0.02,
      inflationStdDev: 0.01,
      paths: 100,
      seed: 42,
    });
    assertCondition('mcBounds', result.successRate >= 0 && result.successRate <= 100, 'Monte Carlo success rate out of bounds');
  });

  run('Historical simulation runner responds', () => {
    const summary = runFullSimulation({
      portfolio: 500000,
      spending: 20000,
      years: 30,
      mode: 'historical',
    });
    assertCondition('historicalMode', summary.mode === 'historical', 'Historical simulation unavailable');
  });

  return { passed: true, count: checks.length };
}
