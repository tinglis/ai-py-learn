import test from 'node:test';
import assert from 'node:assert/strict';
import {
  runYearProjection,
  runRetirementYear,
  runFullProjection,
  loadHistoricalData,
  runHistoricalSim,
  runFullSimulation,
  getGraphData,
  runMonteCarloSimulation,
} from '../src/lib/projection.js';

const PROVINCE = 'ON';

function approxEqual(actual, expected, tolerance = 0.01) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} !~= ${expected}`);
}

test('TFSA grows with investment returns', () => {
  const year = runYearProjection({ tfsa: 10000, preRetRate: 0.1 });
  approxEqual(year.tfsa.balance, 11000);
});

test('RRSP withdrawal produces tax', () => {
  const year = runYearProjection({ rrsp: 10000 }, { province: PROVINCE, withdrawal: 5000 });
  assert.ok(year.rrsp.taxPaid > 0);
});

test('Non-registered ACB increases with contributions', () => {
  const year = runYearProjection({ nonReg: 10000, acb: 8000, nonRegCont: 2000 });
  approxEqual(year.nonReg.acb, 10000);
});

test('Non-registered dividend taxation applied', () => {
  const year = runYearProjection({
    nonReg: 50000,
    acb: 50000,
    preRetRate: 0.07,
    dividendYield: 0.02,
    province: PROVINCE,
  });
  assert.ok(year.nonReg.taxPaid > 0);
});

test('Dollar-mode savings override priority', () => {
  const year = runYearProjection({
    savingsMode: 'Dollar',
    dollarAmounts: { TFSA: 5000 },
    priority: ['RRSP'],
  });
  assert.equal(year.tfsa.contribution, 5000);
  assert.equal(year.rrsp.contribution, 0);
});

test('Percent-mode savings follow allocation priority', () => {
  const year = runYearProjection({
    savingsMode: 'Percent',
    net: 50000,
    percent: 0.1,
    priority: ['TFSA', 'RRSP'],
    tfsaRoom: 4000,
    rrspRoom: 20000,
  });
  assert.equal(year.tfsa.contribution, 4000);
  assert.ok(year.rrsp.contribution > 0);
});

test('TFSA room increases after contribution and annual limit', () => {
  const year = runYearProjection({ tfsaRoom: 10000, tfsaCont: 5000, annualTFSALimit: 7000 });
  assert.equal(year.tfsaRoom, 12000);
});

test('RRSP room updates with new earned room', () => {
  const year = runYearProjection({
    income: 100000,
    rrspRoom: 20000,
    rrspCont: 10000,
    rrspMaxCap: 32000,
  });
  assert.equal(year.rrspRoom, 28000);
});

test('Retirement withdrawals respect GIS threshold', () => {
  const plan = runRetirementYear({
    spending: 25000,
    cpp: 8000,
    gisThreshold: 20000,
    tfsaBal: 100000,
    rrspBal: 100000,
  });
  assert.ok(plan.rrspWithdrawal <= 12000);
  assert.ok(plan.tfsaWithdrawal > 0);
});

test('Retirement GIS calculation uses prior-year income lag', () => {
  const suppressed = runRetirementYear({
    spending: 20000,
    cpp: 0,
    oas: 0,
    gisThreshold: 20000,
    tfsaBal: 50000,
    rrspBal: 0,
    nonRegBal: 0,
    lastYearIncomeForGIS: 60000,
  });
  const eligible = runRetirementYear({
    spending: 20000,
    cpp: 0,
    oas: 0,
    gisThreshold: 20000,
    tfsaBal: 50000,
    rrspBal: 0,
    nonRegBal: 0,
    lastYearIncomeForGIS: 0,
  });
  assert.equal(suppressed.gis, 0);
  assert.ok(eligible.gis > suppressed.gis);
});

test('Run full projection spans accumulation, coasting, and retirement', () => {
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
  assert.ok(result.portfolio[31].contribution > 0);
  assert.equal(result.portfolio[41].contribution, 0);
  assert.ok(result.portfolio[41].balance > result.portfolio[40].balance);
  assert.ok(result.portfolio[50].withdrawal > 0);
  assert.equal(result.portfolio[51].withdrawal, 40800);
});

test('Full projection tracks GIS amounts with prior-year lag', () => {
  const projection = runFullProjection({
    currentAge: 64,
    coastAge: 64,
    fireAge: 65,
    lifeExpectancy: 67,
    income: 70000,
    savings: 0,
    tfsa: 200000,
    retirementSpending: 15000,
    lastYearIncomeForGIS: 70000,
  });
  const firstRetYear = projection.portfolio[65];
  const secondRetYear = projection.portfolio[66];
  assert.equal(firstRetYear.gis, 0);
  assert.ok(secondRetYear.gis > firstRetYear.gis);
});

test('Employer match increases RRSP contribution totals', () => {
  const result = runFullProjection({
    currentAge: 30,
    coastAge: 35,
    fireAge: 45,
    lifeExpectancy: 46,
    savings: 5000,
    income: 100000,
    rrspMatch: 0.05,
    priority: ['RRSP'],
  });
  assert.ok(result.portfolio[31].rrspCont > 5000);
  assert.ok(result.portfolio[31].rrspBalance > 0);
});

test('Full projection flags failure when spending too high', () => {
  const result = runFullProjection({
    currentAge: 30,
    coastAge: 35,
    fireAge: 40,
    lifeExpectancy: 45,
    retirementSpending: 1000000,
    income: 80000,
    savings: 5000,
  });
  assert.equal(result.result, 'Failure');
});

test('Full projection exposes home values and CPP/OAS income series', () => {
  const result = runFullProjection({
    currentAge: 60,
    coastAge: 60,
    fireAge: 65,
    lifeExpectancy: 67,
    savings: 10000,
    income: 70000,
    retirementSpending: 30000,
    homeValue: 500000,
    homeAppr: 0.02,
    mortgageBalance: 300000,
    annualMortgagePayment: 20000,
    cppAnnual: 12000,
    cppStartAge: 65,
    oasAnnual: 8000,
    oasStartAge: 65,
  });
  assert.ok(result.homeValues[60] > 0);
  assert.ok(result.mortgage[60] > 0);
  assert.ok(result.portfolio[65].cppIncome > 0);
  assert.ok(result.portfolio[65].oasIncome > 0);
});

test('Graph data aggregates net worth correctly', () => {
  const projection = runFullProjection({
    currentAge: 30,
    coastAge: 31,
    fireAge: 32,
    lifeExpectancy: 33,
    savings: 5000,
    income: 70000,
    retirementSpending: 30000,
  });
  const graph = getGraphData({ portfolio: projection.portfolio, homeValues: { 31: 200000 }, mortgage: { 31: 150000 } });
  const expected = projection.portfolio[31].balance + 200000 - 150000;
  assert.equal(graph.netWorth[31], expected);
});

test('Historical data loads with sufficient length', () => {
  const { ok, data } = loadHistoricalData();
  assert.ok(ok);
  assert.ok(data.length > 100);
});

test('Historical simulation identifies success and failure scenarios', () => {
  const success = runHistoricalSim({ portfolio: 1000000, spending: 10000, years: 30, startYear: 1900 });
  const failure = runHistoricalSim({ portfolio: 100000, spending: 100000, years: 30, startYear: 1900 });
  assert.equal(success, 'Success');
  assert.equal(failure, 'Failure');
});

test('Full simulation returns bounded success rate', () => {
  const { successRate, mode } = runFullSimulation({
    portfolio: 1000000,
    spending: 40000,
    years: 30,
    mode: 'historical',
  });
  assert.ok(successRate >= 0 && successRate <= 100);
  assert.equal(mode, 'historical');
});

test('Monte Carlo simulation exposes percentile stats and ruin ages', () => {
  const result = runMonteCarloSimulation({
    portfolio: 500000,
    spending: 20000,
    years: 25,
    expectedReturn: 0.06,
    returnStdDev: 0.12,
    inflationMean: 0.02,
    inflationStdDev: 0.01,
    paths: 200,
    seed: 42,
  });
  assert.ok(result.successRate >= 0 && result.successRate <= 100);
  assert.equal(result.paths, 200);
  assert.ok(result.percentiles.p50 >= 0);
  if (result.ruinAges.length > 0) {
    assert.ok(result.ruinAges.every((age) => age >= 0 && age <= 25));
  }
});

test('Full simulation supports Monte Carlo mode', () => {
  const result = runFullSimulation({
    portfolio: 500000,
    spending: 20000,
    years: 25,
    mode: 'monteCarlo',
    seed: 99,
    paths: 100,
  });
  assert.equal(result.mode, 'monteCarlo');
  assert.ok('percentiles' in result);
});
