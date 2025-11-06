import React, { useMemo, useState } from 'react';
import provincesConfig from './config/tax/2024/provinces.json';
import { buildDashboard } from './ui/calculations.js';
import SetupWizard from './components/SetupWizard.jsx';
import InputPanel from './components/InputPanel.jsx';
import NetEarningsPanel from './components/NetEarningsPanel.jsx';
import ResultsPanel from './components/ResultsPanel.jsx';
import PortfolioChart from './components/PortfolioChart.jsx';
import IncomeSourceChart from './components/IncomeSourceChart.jsx';

const PROVINCES = Object.keys(provincesConfig);

const DEFAULT_STATE = {
  profile: {
    currentAge: 32,
    coastAge: 40,
    fireAge: 55,
    lifeExpectancy: 90,
    province: 'ON',
  },
  economy: {
    inflation: 0.02,
    preRetReturn: 0.07,
    postRetReturn: 0.05,
    swr: 0.04,
    dividendYield: 0.02,
    dividendType: 'eligible',
    homeAppreciation: 0.02,
  },
  savings: {
    mode: 'percent',
    percent: 0.2,
    priority: ['TFSA', 'RRSP', 'NON-REGISTERED'],
    refundOption: 'reinvest_next_year',
    dollar: {
      tfsa: { amount: 500, frequency: 'monthly' },
      rrsp: { amount: 400, frequency: 'monthly' },
      rdsp: { amount: 0, frequency: 'annually' },
      fhsa: { amount: 0, frequency: 'annually' },
      nonReg: { amount: 200, frequency: 'monthly' },
    },
  },
  income: {
    gross: 90000,
    growth: 0.03,
    rrspMatch: 0.05,
    credits: [],
  },
  accounts: {
    tfsaBalance: 45000,
    tfsaRoom: 20000,
    rrspBalance: 65000,
    rrspRoom: 30000,
    rdspBalance: 0,
    rdspRoom: 0,
    fhsaBalance: 0,
    fhsaRoom: 8000,
    nonRegBalance: 25000,
    acb: 20000,
  },
  housing: {
    status: 'owner',
    homeValue: 650000,
    mortgageBalance: 320000,
    annualMortgagePayment: 24000,
    annualPropertyTax: 4500,
    maintenanceRate: 0.01,
    rent: 0,
  },
  benefits: {
    cppAnnual: 12000,
    cppStartAge: 65,
    oasAnnual: 8400,
    oasStartAge: 65,
    gisThreshold: 20000,
  },
  retirement: {
    spendingToday: 45000,
  },
  limits: {
    annualTfsa: 7000,
    rrspMax: 32000,
  },
  simulation: {
    mode: 'historical',
    paths: 1000,
    returnStdDev: 0.12,
    inflationStdDev: 0.01,
    seed: 42,
  },
};

function mergeDeep(base, patch) {
  const result = { ...base };
  for (const [key, value] of Object.entries(patch)) {
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      result[key] = mergeDeep(base[key] ?? {}, value);
    } else {
      result[key] = value;
    }
  }
  return result;
}

export default function App() {
  const [state, setState] = useState(DEFAULT_STATE);
  const [showWizard, setShowWizard] = useState(true);

  const derived = useMemo(() => buildDashboard(state), [state]);

  const handleStateChange = (updater) => {
    setState((prev) => {
      if (typeof updater === 'function') {
        return updater(prev);
      }
      return mergeDeep(prev, updater);
    });
  };

  const handleWizardComplete = (partial) => {
    handleStateChange(partial);
    setShowWizard(false);
  };

  return (
    <>
      {showWizard && (
        <SetupWizard
          state={state}
          provinces={PROVINCES}
          onComplete={handleWizardComplete}
          onSkip={() => setShowWizard(false)}
        />
      )}
      <div className="app-shell">
        <InputPanel
          state={state}
          provinces={PROVINCES}
          derived={derived}
          onChange={handleStateChange}
        />
        <NetEarningsPanel net={derived.netEarnings} savings={derived.savings} />
        <div className="panel">
          <ResultsPanel fire={derived.fire} simulation={derived.simulation} state={state} />
          <div className="chart-container">
            <h3>Portfolio Projection</h3>
            <div className="chart-wrapper">
              <PortfolioChart
                projection={derived.projection}
                graph={derived.graph}
                fireAge={state.profile.fireAge}
              />
            </div>
          </div>
          <div className="chart-container">
            <h3>Retirement Income Sources</h3>
            <div className="chart-wrapper">
              <IncomeSourceChart data={derived.incomeSeries} />
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
