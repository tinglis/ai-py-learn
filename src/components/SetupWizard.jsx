import React, { useState } from 'react';

const STEPS = ['Profile', 'Income', 'Goals'];

export default function SetupWizard({ state, provinces, onComplete, onSkip }) {
  const [step, setStep] = useState(0);
  const [profile, setProfile] = useState({ ...state.profile });
  const [income, setIncome] = useState({ gross: state.income.gross, rrspMatch: state.income.rrspMatch });
  const [savingsPercent, setSavingsPercent] = useState(state.savings.percent);
  const [spending, setSpending] = useState(state.retirement.spendingToday);

  const next = () => setStep((value) => Math.min(value + 1, STEPS.length - 1));
  const prev = () => setStep((value) => Math.max(value - 1, 0));

  const finish = () => {
    onComplete({
      profile,
      income: { ...state.income, gross: Number(income.gross) || 0, rrspMatch: Number(income.rrspMatch) || 0 },
      savings: { ...state.savings, percent: Number(savingsPercent) || 0.15 },
      retirement: { ...state.retirement, spendingToday: Number(spending) || 0 },
    });
  };

  return (
    <div className="wizard-overlay">
      <div className="wizard">
        <h2>Welcome to Maple FIRE</h2>
        <p className="info-text">Configure your baseline assumptions so the dashboard can simulate your path to Financial Independence.</p>
        {step === 0 && (
          <div>
            <div className="field">
              <label>Province</label>
              <select
                value={profile.province}
                onChange={(event) => setProfile((prev) => ({ ...prev, province: event.target.value }))}
              >
                {provinces.map((code) => (
                  <option key={code} value={code}>
                    {code}
                  </option>
                ))}
              </select>
            </div>
            <div className="inline-group">
              <div className="field">
                <label>Current Age</label>
                <input
                  type="number"
                  value={profile.currentAge}
                  onChange={(event) =>
                    setProfile((prev) => ({ ...prev, currentAge: Number(event.target.value) }))
                  }
                />
              </div>
              <div className="field">
                <label>Coast Age</label>
                <input
                  type="number"
                  value={profile.coastAge}
                  onChange={(event) =>
                    setProfile((prev) => ({ ...prev, coastAge: Number(event.target.value) }))
                  }
                />
              </div>
            </div>
            <div className="inline-group">
              <div className="field">
                <label>FIRE Age</label>
                <input
                  type="number"
                  value={profile.fireAge}
                  onChange={(event) =>
                    setProfile((prev) => ({ ...prev, fireAge: Number(event.target.value) }))
                  }
                />
              </div>
              <div className="field">
                <label>Life Expectancy</label>
                <input
                  type="number"
                  value={profile.lifeExpectancy}
                  onChange={(event) =>
                    setProfile((prev) => ({ ...prev, lifeExpectancy: Number(event.target.value) }))
                  }
                />
              </div>
            </div>
          </div>
        )}
        {step === 1 && (
          <div>
            <div className="field">
              <label>Gross Annual Income</label>
              <input
                type="number"
                value={income.gross}
                onChange={(event) => setIncome((prev) => ({ ...prev, gross: event.target.value }))}
              />
            </div>
            <div className="field">
              <label>Employer RRSP Match (%)</label>
              <input
                type="number"
                value={income.rrspMatch}
                step="0.01"
                onChange={(event) => setIncome((prev) => ({ ...prev, rrspMatch: event.target.value }))}
              />
            </div>
            <div className="field">
              <label>Savings Rate (% of net)</label>
              <input
                type="number"
                value={Math.round(savingsPercent * 100)}
                onChange={(event) => setSavingsPercent(Number(event.target.value) / 100)}
              />
              <p className="info-text">You can fine-tune the allocation later on the dashboard.</p>
            </div>
          </div>
        )}
        {step === 2 && (
          <div>
            <div className="field">
              <label>Desired Retirement Spending (Today&apos;s $)</label>
              <input
                type="number"
                value={spending}
                onChange={(event) => setSpending(event.target.value)}
              />
            </div>
            <p className="info-text">
              Maple FIRE will inflate this spending path automatically and calculate the FIRE number, Coast FIRE number, and
              simulation metrics based on your plan.
            </p>
          </div>
        )}
        <div className="wizard-buttons">
          <button className="secondary" type="button" onClick={step === 0 ? onSkip : prev}>
            {step === 0 ? 'Skip' : 'Back'}
          </button>
          {step === STEPS.length - 1 ? (
            <button className="primary" type="button" onClick={finish}>
              Launch Dashboard
            </button>
          ) : (
            <button className="primary" type="button" onClick={next}>
              Next
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
