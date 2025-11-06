import fs from 'node:fs';
import path from 'node:path';
import { allocateSavings } from './savings.js';
import { calculateTotalTax } from './tax.js';
import { calculateNetEarnings } from './income.js';
import { calculateGIS } from './benefits.js';
import { calculateRDSPClawback } from './rdsp.js';

const DEFAULT_TAX_YEAR = 2024;
const CAPITAL_GAINS_INCLUSION_RATE = 0.5;
const DEFAULT_PROVINCE = 'ON';
const DEFAULT_MONTE_CARLO_PATHS = 1000;

function createSeededRng(seed = Date.now()) {
  let state = Math.floor(Math.abs(seed)) % 2147483647;
  if (state === 0) {
    state = 1;
  }
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function sampleStandardNormal(rng = Math.random) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function computePercentiles(values = [], targets = []) {
  if (!values.length) {
    return targets.reduce((acc, key) => ({ ...acc, [key]: 0 }), {});
  }
  const sorted = [...values].sort((a, b) => a - b);
  const toIndex = (p) => {
    if (sorted.length === 1) return 0;
    const rank = (p / 100) * (sorted.length - 1);
    const lower = Math.floor(rank);
    const upper = Math.ceil(rank);
    if (lower === upper) {
      return sorted[lower];
    }
    const weight = rank - lower;
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  };
  const result = {};
  for (const target of targets) {
    result[`p${target}`] = toIndex(target);
  }
  return result;
}

function roundToCents(value) {
  return Math.round((value + Number.EPSILON) * 100) / 100;
}

function resolveProvince(value) {
  return value ?? DEFAULT_PROVINCE;
}

function resolveNumber(value, fallback = 0) {
  return Number.isFinite(value) ? value : fallback;
}

export function runYearProjection(state = {}, overrides = {}) {
  const params = { ...state, ...overrides };
  const taxYear = params.taxYear ?? DEFAULT_TAX_YEAR;
  const province = resolveProvince(params.province ?? params.prov);
  const inclusionRate = params.capitalGainsInclusionRate ?? CAPITAL_GAINS_INCLUSION_RATE;
  const rate = params.preRetRate ?? params.postRetRate ?? 0;
  const dividendYield = Math.max(0, Math.min(rate, params.dividendYield ?? 0));
  const otherIncome = params.otherIncome ?? 0;

  let tfsaContribution = params.tfsaCont ?? 0;
  let rrspContribution = params.rrspCont ?? 0;
  let rdspContribution = params.rdspCont ?? 0;
  let fhsaContribution = params.fhsaCont ?? 0;
  let nonRegContribution = params.nonRegCont ?? 0;

  if (params.savingsMode === 'Dollar') {
    const amounts = params.dollarAmounts ?? {};
    tfsaContribution = amounts.TFSA ?? amounts.tfsa ?? tfsaContribution;
    rrspContribution = amounts.RRSP ?? amounts.rrsp ?? rrspContribution;
    rdspContribution = amounts.RDSP ?? amounts.rdsp ?? rdspContribution;
    fhsaContribution = amounts.FHSA ?? amounts.fhsa ?? fhsaContribution;
    nonRegContribution =
      amounts.NonRegistered ??
      amounts.nonRegistered ??
      amounts.nonReg ??
      nonRegContribution;
  } else if (params.savingsMode === 'Percent') {
    const net = params.net ?? 0;
    const percent = params.percent ?? 0;
    const total = net * percent;
    const limits = {
      tfsaRoom: params.tfsaRoom ?? Number.POSITIVE_INFINITY,
      rrspRoom: params.rrspRoom ?? Number.POSITIVE_INFINITY,
      rdspRoom: params.rdspRoom ?? Number.POSITIVE_INFINITY,
      fhsaRoom: params.fhsaRoom ?? Number.POSITIVE_INFINITY,
    };
    const allocations = allocateSavings(total, { priority: params.priority ?? [], limits });
    tfsaContribution = allocations.tfsa ?? tfsaContribution;
    rrspContribution = allocations.rrsp ?? rrspContribution;
    rdspContribution = allocations.rdsp ?? rdspContribution;
    fhsaContribution = allocations.fhsa ?? fhsaContribution;
    nonRegContribution =
      (allocations.nonRegistered ?? 0) + (allocations.unallocated ?? 0);
  }

  const income = params.income ?? 0;
  const rrspMatchRate = params.rrspMatch ?? 0;
  let employerMatch = 0;
  if (rrspMatchRate > 0 && income > 0) {
    employerMatch = income * rrspMatchRate;
    const room = params.rrspRoom;
    if (Number.isFinite(room)) {
      const available = Math.max(0, room - rrspContribution);
      employerMatch = Math.min(employerMatch, available);
    }
  }
  const totalRrspContribution = rrspContribution + employerMatch;

  const startingTfsa = params.tfsa ?? 0;
  const startingRrsp = params.rrsp ?? 0;
  const startingNonReg = params.nonReg ?? 0;
  const startingRdsp = params.rdsp ?? 0;
  const startingFhsa = params.fhsa ?? 0;
  const startingAcb = params.acb ?? startingNonReg;

  let tfsaWithdrawal = params.tfsaWithdrawal ?? 0;
  let rrspWithdrawal = params.rrspWithdrawal ?? 0;
  let nonRegWithdrawal = params.nonRegWithdrawal ?? 0;
  let rdspWithdrawal = params.rdspWithdrawal ?? 0;

  const genericWithdrawal = params.withdrawal ?? 0;
  const withdrawFrom = params.withdrawFrom?.toLowerCase();
  if (genericWithdrawal > 0) {
    if (withdrawFrom === 'tfsa') {
      tfsaWithdrawal += genericWithdrawal;
    } else if (withdrawFrom === 'rrsp') {
      rrspWithdrawal += genericWithdrawal;
    } else if (
      withdrawFrom === 'nonregistered' ||
      withdrawFrom === 'non-reg' ||
      withdrawFrom === 'nonreg'
    ) {
      nonRegWithdrawal += genericWithdrawal;
    } else if (startingRrsp > 0) {
      rrspWithdrawal += genericWithdrawal;
    } else if (startingTfsa > 0) {
      tfsaWithdrawal += genericWithdrawal;
    } else {
      nonRegWithdrawal += genericWithdrawal;
    }
  }

  const maxTfsaWithdrawal = startingTfsa + tfsaContribution;
  const maxRrspWithdrawal = startingRrsp + totalRrspContribution;
  const maxNonRegWithdrawal = startingNonReg + nonRegContribution;
  const maxRdspWithdrawal = startingRdsp + rdspContribution;
  if (tfsaWithdrawal > maxTfsaWithdrawal) tfsaWithdrawal = maxTfsaWithdrawal;
  if (rrspWithdrawal > maxRrspWithdrawal) rrspWithdrawal = maxRrspWithdrawal;
  if (nonRegWithdrawal > maxNonRegWithdrawal)
    nonRegWithdrawal = maxNonRegWithdrawal;
  if (rdspWithdrawal > maxRdspWithdrawal) rdspWithdrawal = maxRdspWithdrawal;

  let tfsaBalance = Math.max(0, startingTfsa + tfsaContribution - tfsaWithdrawal);
  let rrspBalance = Math.max(
    0,
    startingRrsp + totalRrspContribution - rrspWithdrawal,
  );
  let nonRegBalance = Math.max(
    0,
    startingNonReg + nonRegContribution - nonRegWithdrawal,
  );
  let rdspBalance = Math.max(
    0,
    startingRdsp + rdspContribution - rdspWithdrawal,
  );
  let fhsaBalance = Math.max(0, startingFhsa + fhsaContribution);

  let rdspClawback = 0;
  if (rdspWithdrawal > 0) {
    rdspClawback = calculateRDSPClawback({
      withdrawal: rdspWithdrawal,
      grants: params.rdspGrantsInLast10Yrs ?? 0,
      bonds: params.rdspBondsInLast10Yrs ?? 0,
      totalAssistance:
        params.rdspAssistance ??
        (params.rdspGrantsInLast10Yrs ?? 0) + (params.rdspBondsInLast10Yrs ?? 0),
    });
    rdspBalance = Math.max(0, rdspBalance - rdspClawback);
  }

  let acb = startingAcb + nonRegContribution;
  let nonRegTaxPaid = 0;
  if (nonRegWithdrawal > 0 && startingNonReg + nonRegContribution > 0) {
    const totalBeforeWithdrawal = startingNonReg + nonRegContribution;
    const withdrawalRatio = Math.min(1, nonRegWithdrawal / totalBeforeWithdrawal);
    const acbReduction = acb * withdrawalRatio;
    acb = Math.max(0, acb - acbReduction);
    const capitalGain = Math.max(0, nonRegWithdrawal - acbReduction);
    const taxableGain = capitalGain * inclusionRate;
    if (taxableGain > 0) {
      nonRegTaxPaid += calculateTotalTax(taxableGain, province, {
        taxYear,
        includePersonalAmount: false,
      });
    }
  }

  const dividendBase = startingNonReg + nonRegContribution;
  const dividends = dividendBase * dividendYield;
  if (dividends > 0) {
    nonRegTaxPaid += calculateTotalTax(dividends, province, {
      taxYear,
      includePersonalAmount: false,
    });
  }
  const capitalGrowthRate = Math.max(0, rate - dividendYield);
  const nonRegGrowthBase = Math.max(0, nonRegBalance);
  const capitalGrowth = nonRegGrowthBase * capitalGrowthRate;
  nonRegBalance = nonRegGrowthBase + dividends + capitalGrowth;

  tfsaBalance *= 1 + rate;
  rrspBalance *= 1 + rate;
  rdspBalance *= 1 + rate;
  fhsaBalance *= 1 + rate;

  const rrspTaxPaid = rrspWithdrawal
    ? calculateTotalTax(rrspWithdrawal, province, {
        taxYear,
        includePersonalAmount: false,
      })
    : 0;

  const totalTaxPaid = roundToCents(nonRegTaxPaid + rrspTaxPaid);

  const annualTfsaLimit = params.annualTFSALimit ?? 0;
  const newTfsaRoom = roundToCents(
    Math.max(0, (params.tfsaRoom ?? 0) - tfsaContribution + annualTfsaLimit),
  );
  const earnedRrspRoom = Math.min(
    params.rrspMaxCap ?? Number.POSITIVE_INFINITY,
    Math.max(0, 0.18 * (income ?? 0)),
  );
  const newRrspRoom = roundToCents(
    Math.max(0, (params.rrspRoom ?? 0) - totalRrspContribution + earnedRrspRoom),
  );

  let taxRefund = 0;
  if (income > 0 && totalRrspContribution > 0) {
    const netResult = calculateNetEarnings({
      gross: income,
      rrsp: totalRrspContribution,
      province,
      taxYear,
    });
    taxRefund = netResult.refund;
  }

  const homeValue = params.homeValue != null
    ? roundToCents(params.homeValue * (1 + (params.homeAppr ?? 0)))
    : undefined;

  return {
    tfsa: {
      balance: roundToCents(tfsaBalance),
      contribution: roundToCents(tfsaContribution),
      withdrawal: roundToCents(tfsaWithdrawal),
    },
    rrsp: {
      balance: roundToCents(rrspBalance),
      contribution: roundToCents(totalRrspContribution),
      withdrawal: roundToCents(rrspWithdrawal),
      employerMatch: roundToCents(employerMatch),
      taxPaid: roundToCents(rrspTaxPaid),
      taxRefund: roundToCents(taxRefund),
    },
    nonReg: {
      balance: roundToCents(nonRegBalance),
      contribution: roundToCents(nonRegContribution),
      withdrawal: roundToCents(nonRegWithdrawal),
      acb: roundToCents(acb),
      taxPaid: roundToCents(nonRegTaxPaid),
    },
    rdsp: {
      balance: roundToCents(rdspBalance),
      contribution: roundToCents(rdspContribution),
      withdrawal: roundToCents(rdspWithdrawal),
      clawback: roundToCents(rdspClawback),
      balanceReduction: roundToCents(rdspWithdrawal + rdspClawback),
    },
    fhsa: {
      balance: roundToCents(fhsaBalance),
      contribution: roundToCents(fhsaContribution),
    },
    totals: {
      contributions: roundToCents(
        tfsaContribution + totalRrspContribution + nonRegContribution + rdspContribution + fhsaContribution,
      ),
      withdrawals: roundToCents(tfsaWithdrawal + rrspWithdrawal + nonRegWithdrawal + rdspWithdrawal),
      taxPaid: roundToCents(totalTaxPaid),
    },
    tfsaRoom: newTfsaRoom,
    rrspRoom: newRrspRoom,
    taxPaid: roundToCents(totalTaxPaid),
    homeValue,
  };
}

export function runRetirementYear({
  spending = 0,
  cpp = 0,
  oas = 0,
  gisThreshold = 0,
  tfsaBal = 0,
  rrspBal = 0,
  nonRegBal = 0,
  province = DEFAULT_PROVINCE,
  taxYear = DEFAULT_TAX_YEAR,
  rdspBalance = 0,
  rdspGrantsInLast10Yrs = 0,
  rdspWithdrawal = 0,
  lastYearIncomeForGIS,
} = {}) {
  const remainingNeed = Math.max(0, spending - (cpp + oas));
  const maxRrspForGIS = Math.max(0, gisThreshold - (cpp + oas));
  const rrspWithdrawal = Math.min(remainingNeed, maxRrspForGIS, rrspBal);
  let residualNeed = Math.max(0, remainingNeed - rrspWithdrawal);
  const nonRegWithdrawal = Math.min(residualNeed, nonRegBal);
  residualNeed = Math.max(0, residualNeed - nonRegWithdrawal);
  const tfsaWithdrawal = Math.min(residualNeed, tfsaBal);
  residualNeed = Math.max(0, residualNeed - tfsaWithdrawal);
  const extraRrsp = Math.min(residualNeed, rrspBal - rrspWithdrawal);
  const totalRrspWithdrawal = rrspWithdrawal + extraRrsp;
  residualNeed = Math.max(0, residualNeed - extraRrsp);
  const calculatedRdspWithdrawal =
    rdspWithdrawal > 0 ? Math.min(rdspWithdrawal, rdspBalance) : 0;
  residualNeed = Math.max(0, residualNeed - calculatedRdspWithdrawal);
  const rdspTopUp = Math.min(residualNeed, rdspBalance - calculatedRdspWithdrawal);
  const totalRdspWithdrawal = calculatedRdspWithdrawal + rdspTopUp;
  residualNeed = Math.max(0, residualNeed - rdspTopUp);

  let rdspClawback = 0;
  if (totalRdspWithdrawal > 0) {
    rdspClawback = calculateRDSPClawback({
      withdrawal: totalRdspWithdrawal,
      grants: rdspGrantsInLast10Yrs,
      bonds: 0,
      totalAssistance: rdspGrantsInLast10Yrs,
    });
  }

  const taxableIncome = totalRrspWithdrawal + nonRegWithdrawal;
  const gis = calculateGIS({
    income: taxableIncome,
    cppIncome: cpp,
    lastYearIncome: lastYearIncomeForGIS,
    taxYear,
  });
  const incomeForGISNextYear = Math.max(0, taxableIncome + cpp);

  return {
    rrspWithdrawal: roundToCents(totalRrspWithdrawal),
    nonRegWithdrawal: roundToCents(nonRegWithdrawal),
    tfsaWithdrawal: roundToCents(tfsaWithdrawal),
    rdspWithdrawal: roundToCents(totalRdspWithdrawal),
    residualNeed: roundToCents(residualNeed),
    gis: roundToCents(gis),
    rdspBalanceReduction: roundToCents(totalRdspWithdrawal + rdspClawback),
    taxableIncome: roundToCents(taxableIncome),
    incomeForGISNextYear: roundToCents(incomeForGISNextYear),
  };
}

export function runFullProjection(options = {}) {
  const {
    currentAge = 30,
    coastAge = currentAge + 10,
    fireAge = currentAge + 25,
    lifeExpectancy = fireAge + 30,
    savings = 0,
    savingsMode = 'Percent',
    dollarAmounts = {},
    priority = ['TFSA', 'RRSP', 'NON-REGISTERED'],
    tfsaRoom: initialTfsaRoom = 0,
    rrspRoom: initialRrspRoom = 0,
    rdspRoom: initialRdspRoom = Number.POSITIVE_INFINITY,
    fhsaRoom: initialFhsaRoom = Number.POSITIVE_INFINITY,
    tfsa: initialTfsa = 0,
    rrsp: initialRrsp = 0,
    nonReg: initialNonReg = 0,
    rdsp: initialRdsp = 0,
    fhsa: initialFhsa = 0,
    acb: initialAcb = initialNonReg,
    income: initialIncome = 0,
    incomeGrowth = 0,
    preRetRate = 0.05,
    postRetRate = 0.04,
    dividendYield = 0.02,
    retirementSpending = 0,
    inflation = 0.02,
    rrspMatch = 0,
    refundOption = 'spend',
    province = DEFAULT_PROVINCE,
    annualTFSALimit = 7000,
    rrspMaxCap = 32000,
    cppAnnual = 0,
    cppStartAge = 65,
    oasAnnual = 0,
    oasStartAge = 65,
    gisThreshold: configuredGisThreshold = 20000,
    homeValue: initialHomeValue = null,
    homeAppr = 0,
    mortgageBalance: initialMortgageBalance = null,
    annualMortgagePayment = 0,
  } = options;

  let tfsaBalance = initialTfsa;
  let rrspBalance = initialRrsp;
  let nonRegBalance = initialNonReg;
  let rdspBalance = initialRdsp;
  let fhsaBalance = initialFhsa;
  let acb = initialAcb;
  let tfsaRoom = initialTfsaRoom;
  let rrspRoom = initialRrspRoom;
  let rdspRoom = initialRdspRoom;
  let fhsaRoom = initialFhsaRoom;
  let income = initialIncome;
  let refundCarry = 0;
  let lastYearIncomeForGIS = Math.max(0, options.lastYearIncomeForGIS ?? income);
  let homeValue =
    initialHomeValue != null ? roundToCents(initialHomeValue) : initialHomeValue;
  let mortgageBalance =
    initialMortgageBalance != null
      ? roundToCents(initialMortgageBalance)
      : initialMortgageBalance;

  const portfolio = [];
  const homeValues = {};
  const mortgageSeries = {};
  let result = 'Success';

  for (let age = currentAge; age <= lifeExpectancy; age += 1) {
    const isAccumulation = age < coastAge;
    const isCoasting = age >= coastAge && age < fireAge;
    const isRetirement = age >= fireAge;

    let tfsaContribution = 0;
    let rrspContribution = 0;
    let nonRegContribution = 0;
    let rdspContribution = 0;
    let fhsaContribution = 0;
    let totalWithdrawal = 0;
    let rrspWithdrawal = 0;
    let tfsaWithdrawal = 0;
    let nonRegWithdrawal = 0;
    let rdspWithdrawal = 0;
    let yearProjection = null;
    let withdrawalPlan = null;
    let employerMatchAmount = 0;
    let gisAmount = 0;

    if (isAccumulation) {
      if (savingsMode.toLowerCase() === 'dollar') {
        const dollarPlan = { ...dollarAmounts };
        if (refundCarry > 0 && refundOption === 'reinvest_next_year') {
          dollarPlan.NonRegistered =
            (dollarPlan.NonRegistered ?? 0) + refundCarry;
          refundCarry = 0;
        }
        const projection = runYearProjection({
          tfsa: tfsaBalance,
          rrsp: rrspBalance,
          nonReg: nonRegBalance,
          rdsp: rdspBalance,
          fhsa: fhsaBalance,
          acb,
          savingsMode: 'Dollar',
          dollarAmounts: dollarPlan,
          tfsaRoom,
          rrspRoom,
          rdspRoom,
          fhsaRoom,
          income,
          preRetRate,
          dividendYield,
          rrspMatch,
          annualTFSALimit,
          rrspMaxCap,
          province,
          taxYear: options.taxYear ?? DEFAULT_TAX_YEAR,
        });

        yearProjection = projection;
        tfsaContribution = projection.tfsa.contribution;
        rrspContribution = projection.rrsp.contribution;
        nonRegContribution = projection.nonReg.contribution;
        rdspContribution = projection.rdsp.contribution;
        fhsaContribution = projection.fhsa.contribution;
        employerMatchAmount = projection.rrsp.employerMatch;
        totalWithdrawal = projection.totals.withdrawals;
        rrspWithdrawal = projection.rrsp.withdrawal;
        tfsaWithdrawal = projection.tfsa.withdrawal;
        nonRegWithdrawal = projection.nonReg.withdrawal;
        rdspWithdrawal = projection.rdsp.withdrawal;

        tfsaBalance = projection.tfsa.balance;
        rrspBalance = projection.rrsp.balance;
        nonRegBalance = projection.nonReg.balance;
        rdspBalance = projection.rdsp.balance;
        fhsaBalance = projection.fhsa.balance;
        acb = projection.nonReg.acb;
        tfsaRoom = projection.tfsaRoom;
        rrspRoom = projection.rrspRoom;

        if (refundOption === 'reinvest_next_year') {
          refundCarry = projection.rrsp.taxRefund;
        }
      } else {
        let totalSavings = savings + refundCarry;
        refundCarry = 0;
        const limits = {
          tfsaRoom,
          rrspRoom,
          rdspRoom,
          fhsaRoom,
        };
        const allocations = allocateSavings(totalSavings, {
          priority,
          limits,
        });
        tfsaContribution = allocations.tfsa ?? 0;
        rrspContribution = allocations.rrsp ?? 0;
        rdspContribution = allocations.rdsp ?? 0;
        fhsaContribution = allocations.fhsa ?? 0;
        nonRegContribution =
          (allocations.nonRegistered ?? 0) + (allocations.unallocated ?? 0);

        const projection = runYearProjection(
          {
            tfsa: tfsaBalance,
            rrsp: rrspBalance,
            nonReg: nonRegBalance,
            rdsp: rdspBalance,
            fhsa: fhsaBalance,
            acb,
            tfsaCont: tfsaContribution,
            rrspCont: rrspContribution,
            rdspCont: rdspContribution,
            fhsaCont: fhsaContribution,
            nonRegCont: nonRegContribution,
            tfsaRoom,
            rrspRoom,
            rdspRoom,
            fhsaRoom,
            income,
            preRetRate,
            dividendYield,
            rrspMatch,
            annualTFSALimit,
            rrspMaxCap,
            province,
            taxYear: options.taxYear ?? DEFAULT_TAX_YEAR,
          },
          {},
        );

        yearProjection = projection;
        tfsaContribution = projection.tfsa.contribution;
        rrspContribution = projection.rrsp.contribution;
        nonRegContribution = projection.nonReg.contribution;
        rdspContribution = projection.rdsp.contribution;
        fhsaContribution = projection.fhsa.contribution;
        employerMatchAmount = projection.rrsp.employerMatch;
        totalWithdrawal = projection.totals.withdrawals;
        rrspWithdrawal = projection.rrsp.withdrawal;
        tfsaWithdrawal = projection.tfsa.withdrawal;
        nonRegWithdrawal = projection.nonReg.withdrawal;
        rdspWithdrawal = projection.rdsp.withdrawal;

        tfsaBalance = projection.tfsa.balance;
        rrspBalance = projection.rrsp.balance;
        nonRegBalance = projection.nonReg.balance;
        rdspBalance = projection.rdsp.balance;
        fhsaBalance = projection.fhsa.balance;
        acb = projection.nonReg.acb;
        tfsaRoom = projection.tfsaRoom;
        rrspRoom = projection.rrspRoom;

        if (refundOption === 'reinvest_next_year') {
          refundCarry = projection.rrsp.taxRefund;
        }
      }
      lastYearIncomeForGIS = Math.max(0, income);
    } else if (isCoasting) {
      const projection = runYearProjection({
        tfsa: tfsaBalance,
        rrsp: rrspBalance,
        nonReg: nonRegBalance,
        rdsp: rdspBalance,
        fhsa: fhsaBalance,
        acb,
        tfsaRoom,
        rrspRoom,
        rdspRoom,
        fhsaRoom,
        preRetRate,
        dividendYield,
        annualTFSALimit,
        rrspMaxCap,
        province,
        taxYear: options.taxYear ?? DEFAULT_TAX_YEAR,
      });
      yearProjection = projection;
      tfsaContribution = projection.tfsa.contribution;
      rrspContribution = projection.rrsp.contribution;
      nonRegContribution = projection.nonReg.contribution;
      rdspContribution = projection.rdsp.contribution;
      fhsaContribution = projection.fhsa.contribution;
      employerMatchAmount = projection.rrsp.employerMatch;
      totalWithdrawal = projection.totals.withdrawals;
      rrspWithdrawal = projection.rrsp.withdrawal;
      tfsaWithdrawal = projection.tfsa.withdrawal;
      nonRegWithdrawal = projection.nonReg.withdrawal;
      rdspWithdrawal = projection.rdsp.withdrawal;
      tfsaBalance = projection.tfsa.balance;
      rrspBalance = projection.rrsp.balance;
      nonRegBalance = projection.nonReg.balance;
      rdspBalance = projection.rdsp.balance;
      fhsaBalance = projection.fhsa.balance;
      acb = projection.nonReg.acb;
      tfsaRoom = projection.tfsaRoom;
      rrspRoom = projection.rrspRoom;
      lastYearIncomeForGIS = Math.max(0, income);
    } else if (isRetirement) {
      const yearsIntoRetirement = age - fireAge;
      const spendingNominal =
        retirementSpending * Math.pow(1 + inflation, yearsIntoRetirement);
      const cppIncome =
        age >= cppStartAge
          ? cppAnnual * Math.pow(1 + inflation, Math.max(0, age - cppStartAge))
          : 0;
      const oasIncome =
        age >= oasStartAge
          ? oasAnnual * Math.pow(1 + inflation, Math.max(0, age - oasStartAge))
          : 0;
      withdrawalPlan = runRetirementYear({
        spending: spendingNominal,
        cpp: cppIncome,
        oas: oasIncome,
        gisThreshold: configuredGisThreshold,
        tfsaBal: tfsaBalance,
        rrspBal: rrspBalance,
        nonRegBal: nonRegBalance,
        rdspBalance,
        rdspGrantsInLast10Yrs: options.rdspGrantsInLast10Yrs ?? 0,
        lastYearIncomeForGIS,
        taxYear: options.taxYear ?? DEFAULT_TAX_YEAR,
        province,
      });
      gisAmount = withdrawalPlan.gis;
      rrspWithdrawal = withdrawalPlan.rrspWithdrawal;
      nonRegWithdrawal = withdrawalPlan.nonRegWithdrawal;
      tfsaWithdrawal = withdrawalPlan.tfsaWithdrawal;
      rdspWithdrawal = withdrawalPlan.rdspWithdrawal;
      totalWithdrawal =
        rrspWithdrawal + nonRegWithdrawal + tfsaWithdrawal + rdspWithdrawal;

      const projection = runYearProjection({
        tfsa: tfsaBalance,
        rrsp: rrspBalance,
        nonReg: nonRegBalance,
        rdsp: rdspBalance,
        fhsa: fhsaBalance,
        acb,
        tfsaWithdrawal,
        rrspWithdrawal,
        nonRegWithdrawal,
        rdspWithdrawal,
        preRetRate: postRetRate,
        dividendYield,
        tfsaRoom,
        rrspRoom,
        annualTFSALimit,
        rrspMaxCap,
        province,
        taxYear: options.taxYear ?? DEFAULT_TAX_YEAR,
      });

      yearProjection = projection;
      tfsaContribution = projection.tfsa.contribution;
      rrspContribution = projection.rrsp.contribution;
      nonRegContribution = projection.nonReg.contribution;
      rdspContribution = projection.rdsp.contribution;
      fhsaContribution = projection.fhsa.contribution;
      employerMatchAmount = projection.rrsp.employerMatch;
      totalWithdrawal = projection.totals.withdrawals;
      rrspWithdrawal = projection.rrsp.withdrawal;
      tfsaWithdrawal = projection.tfsa.withdrawal;
      nonRegWithdrawal = projection.nonReg.withdrawal;
      rdspWithdrawal = projection.rdsp.withdrawal;

      tfsaBalance = projection.tfsa.balance;
      rrspBalance = projection.rrsp.balance;
      nonRegBalance = projection.nonReg.balance;
      rdspBalance = projection.rdsp.balance;
      fhsaBalance = projection.fhsa.balance;
      acb = projection.nonReg.acb;
      tfsaRoom = projection.tfsaRoom;
      rrspRoom = projection.rrspRoom;

      if (tfsaBalance + rrspBalance + nonRegBalance + rdspBalance < 0.01) {
        result = 'Failure';
      }

      lastYearIncomeForGIS = withdrawalPlan.incomeForGISNextYear;
    }

    portfolio[age] = {
      age,
      balance: roundToCents(
        tfsaBalance + rrspBalance + nonRegBalance + rdspBalance + fhsaBalance,
      ),
      tfsaCont: roundToCents(tfsaContribution),
      rrspCont: roundToCents(rrspContribution),
      nonRegCont: roundToCents(nonRegContribution),
      rdspCont: roundToCents(rdspContribution),
      fhsaCont: roundToCents(fhsaContribution),
      contribution: roundToCents(yearProjection ? yearProjection.totals.contributions : tfsaContribution + rrspContribution + nonRegContribution + rdspContribution + fhsaContribution),
      withdrawal: roundToCents(yearProjection ? yearProjection.totals.withdrawals : totalWithdrawal),
      rrspWithdrawal: roundToCents(rrspWithdrawal),
      tfsaWithdrawal: roundToCents(tfsaWithdrawal),
      nonRegWithdrawal: roundToCents(nonRegWithdrawal),
      rdspWithdrawal: roundToCents(rdspWithdrawal),
      employerMatch: roundToCents(employerMatchAmount),
      gis: roundToCents(gisAmount),
      tfsaBalance: roundToCents(tfsaBalance),
      rrspBalance: roundToCents(rrspBalance),
      nonRegBalance: roundToCents(nonRegBalance),
      rdspBalance: roundToCents(rdspBalance),
      fhsaBalance: roundToCents(fhsaBalance),
      cppIncome: roundToCents(
        age >= fireAge
          ? age >= cppStartAge
            ? cppAnnual * Math.pow(1 + inflation, Math.max(0, age - cppStartAge))
            : 0
          : 0,
      ),
      oasIncome: roundToCents(
        age >= fireAge
          ? age >= oasStartAge
            ? oasAnnual * Math.pow(1 + inflation, Math.max(0, age - oasStartAge))
            : 0
          : 0,
      ),
    };

    if (homeValue != null) {
      homeValues[age] = roundToCents(homeValue);
      homeValue = roundToCents(homeValue * (1 + homeAppr));
    }
    if (mortgageBalance != null) {
      mortgageSeries[age] = roundToCents(Math.max(0, mortgageBalance));
      mortgageBalance = roundToCents(
        Math.max(0, mortgageBalance - annualMortgagePayment),
      );
    }

    income *= 1 + incomeGrowth;
  }

  return {
    portfolio,
    result,
    homeValues,
    mortgage: mortgageSeries,
  };
}

export function loadHistoricalData() {
  const file = path.join('src', 'data', 'historical_data.json');
  if (!fs.existsSync(file)) {
    return { ok: false, data: [] };
  }
  const raw = JSON.parse(fs.readFileSync(file, 'utf-8'));
  return { ok: true, data: raw };
}

export function runHistoricalSim({
  portfolio = 0,
  spending = 0,
  years = 30,
  startYear = 1900,
} = {}) {
  const { ok, data } = loadHistoricalData();
  if (!ok) {
    throw new Error('Historical data unavailable');
  }
  const startIndex = data.findIndex((entry) => entry.year === startYear);
  if (startIndex === -1 || startIndex + years >= data.length) {
    return 'Failure';
  }
  let balance = portfolio;
  for (let i = 0; i < years; i += 1) {
    const entry = data[startIndex + i];
    balance *= 1 + entry.return;
    balance -= spending * Math.pow(1 + entry.inflation, i);
    if (balance <= 0) {
      return 'Failure';
    }
  }
  return 'Success';
}

export function runMonteCarloSimulation({
  portfolio = 1000000,
  spending = 40000,
  years = 30,
  expectedReturn = 0.05,
  returnStdDev = 0.1,
  inflationMean = 0.02,
  inflationStdDev = 0.01,
  correlation = 0,
  paths = DEFAULT_MONTE_CARLO_PATHS,
  seed,
} = {}) {
  const rng = seed != null ? createSeededRng(seed) : Math.random;
  const finalBalances = [];
  const ruinAges = [];
  let successCount = 0;
  const corr = Math.max(-0.99, Math.min(0.99, correlation));
  const sqrtTerm = Math.sqrt(Math.max(0, 1 - corr * corr));

  for (let p = 0; p < paths; p += 1) {
    let balance = portfolio;
    let ruined = false;
    let ruinYear = years;
    let spendingFactor = 1;
    for (let year = 0; year < years; year += 1) {
      const zReturn = sampleStandardNormal(rng);
      const zInflBase = sampleStandardNormal(rng);
      const zInfl = corr * zReturn + sqrtTerm * zInflBase;
      const logReturn =
        (expectedReturn - (returnStdDev ** 2) / 2) + returnStdDev * zReturn;
      const annualReturn = Math.exp(logReturn) - 1;
      const inflationRate = Math.max(-0.99, inflationMean + inflationStdDev * zInfl);

      balance *= 1 + annualReturn;
      const nominalSpending = spending * spendingFactor;
      balance -= nominalSpending;
      spendingFactor *= 1 + inflationRate;

      if (balance <= 0) {
        ruined = true;
        ruinYear = year;
        balance = 0;
        break;
      }
    }
    if (!ruined) {
      successCount += 1;
    } else {
      ruinAges.push(ruinYear);
    }
    finalBalances.push(roundToCents(balance));
  }

  const successRate = paths > 0 ? (successCount / paths) * 100 : 0;
  const percentilesRaw = computePercentiles(finalBalances, [5, 25, 50, 75, 95]);
  const percentiles = Object.fromEntries(
    Object.entries(percentilesRaw).map(([key, value]) => [key, roundToCents(value)]),
  );

  return {
    successRate,
    percentiles,
    ruinAges,
    paths,
  };
}

export function runFullSimulation({
  portfolio = 1000000,
  spending = 40000,
  years = 30,
  mode = 'historical',
  expectedReturn,
  returnStdDev,
  inflationMean,
  inflationStdDev,
  correlation,
  paths,
  seed,
} = {}) {
  if (mode === 'historical') {
    const { ok, data } = loadHistoricalData();
    if (!ok) {
      throw new Error('Historical data unavailable');
    }
    let successCount = 0;
    const runs = [];
    for (let i = 0; i + years < data.length; i += 1) {
      const outcome = runHistoricalSim({
        portfolio,
        spending,
        years,
        startYear: data[i].year,
      });
      runs.push(outcome);
      if (outcome === 'Success') {
        successCount += 1;
      }
    }
    const successRate = runs.length > 0 ? (successCount / runs.length) * 100 : 0;
    return {
      successRate,
      runs,
      mode,
    };
  }

  if (mode === 'monteCarlo') {
    return {
      ...runMonteCarloSimulation({
        portfolio,
        spending,
        years,
        expectedReturn,
        returnStdDev,
        inflationMean,
        inflationStdDev,
        correlation,
        paths,
        seed,
      }),
      mode,
    };
  }

  throw new Error(`Unsupported simulation mode: ${mode}`);
}

export function getGraphData({
  portfolio = [],
  homeValues = {},
  mortgage = {},
} = {}) {
  const invPortfolio = [];
  const netWorth = [];
  const homeEquity = [];
  const mortgageSeries = [];
  for (const entry of portfolio) {
    if (!entry) {
      continue;
    }
    const age = entry.age;
    const investments = entry.balance ?? 0;
    const home = homeValues[age] ?? 0;
    const mortgageBalance = mortgage[age] ?? 0;
    invPortfolio[age] = investments;
    homeEquity[age] = home;
    mortgageSeries[age] = mortgageBalance;
    netWorth[age] = roundToCents(investments + home - mortgageBalance);
  }
  return {
    invPortfolio,
    netWorth,
    homeEquity,
    mortgage: mortgageSeries,
  };
}
