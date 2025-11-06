import React from 'react';

const FREQUENCIES = [
  { value: 'annually', label: 'Annually' },
  { value: 'monthly', label: 'Monthly' },
  { value: 'semiMonthly', label: 'Semi-Monthly' },
  { value: 'weekly', label: 'Weekly' },
];

function updateAtPath(prev, path, value) {
  const segments = Array.isArray(path) ? path : String(path).split('.');
  const next = { ...prev };
  let cursor = next;
  for (let i = 0; i < segments.length - 1; i += 1) {
    const key = segments[i];
    cursor[key] = { ...cursor[key] };
    cursor = cursor[key];
  }
  cursor[segments[segments.length - 1]] = value;
  return next;
}

function asNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

export default function InputPanel({ state, provinces, derived, onChange }) {
  const { profile, economy, savings, income, accounts, housing, benefits, retirement, limits, simulation } = state;

  const setValue = (path, value) => {
    onChange((prev) => updateAtPath(prev, path, value));
  };

  const toggleSavingsMode = (mode) => {
    if (mode === savings.mode) return;
    if (mode === 'percent') {
      setValue(['savings', 'mode'], 'percent');
    } else {
      setValue(['savings', 'mode'], 'dollar');
      setValue(['savings', 'percent'], derived.savings.percentEquivalent);
    }
  };

  const priorityString = savings.priority.join(', ');

  const handlePriorityChange = (event) => {
    const entries = event.target.value
      .split(',')
      .map((token) => token.trim())
      .filter(Boolean);
    setValue(['savings', 'priority'], entries.length > 0 ? entries : ['TFSA', 'RRSP', 'NON-REGISTERED']);
  };

  return (
    <div className="panel">
      <h2>Plan Inputs</h2>
      <section>
        <h3>Profile</h3>
        <div className="field">
          <label>Province</label>
          <select value={profile.province} onChange={(event) => setValue(['profile', 'province'], event.target.value)}>
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
              onChange={(event) => setValue(['profile', 'currentAge'], asNumber(event.target.value))}
            />
          </div>
          <div className="field">
            <label>Coast Age</label>
            <input
              type="number"
              value={profile.coastAge}
              onChange={(event) => setValue(['profile', 'coastAge'], asNumber(event.target.value))}
            />
          </div>
        </div>
        <div className="inline-group">
          <div className="field">
            <label>FIRE Age</label>
            <input
              type="number"
              value={profile.fireAge}
              onChange={(event) => setValue(['profile', 'fireAge'], asNumber(event.target.value))}
            />
          </div>
          <div className="field">
            <label>Life Expectancy</label>
            <input
              type="number"
              value={profile.lifeExpectancy}
              onChange={(event) => setValue(['profile', 'lifeExpectancy'], asNumber(event.target.value))}
            />
          </div>
        </div>
      </section>

      <section>
        <h3>Economy</h3>
        <div className="inline-group">
          <div className="field">
            <label>Inflation</label>
            <input
              type="number"
              step="0.001"
              value={economy.inflation}
              onChange={(event) => setValue(['economy', 'inflation'], asNumber(event.target.value))}
            />
          </div>
          <div className="field">
            <label>Pre-Ret Return</label>
            <input
              type="number"
              step="0.001"
              value={economy.preRetReturn}
              onChange={(event) => setValue(['economy', 'preRetReturn'], asNumber(event.target.value))}
            />
          </div>
          <div className="field">
            <label>Post-Ret Return</label>
            <input
              type="number"
              step="0.001"
              value={economy.postRetReturn}
              onChange={(event) => setValue(['economy', 'postRetReturn'], asNumber(event.target.value))}
            />
          </div>
        </div>
        <div className="inline-group">
          <div className="field">
            <label>Safe Withdrawal Rate</label>
            <input
              type="number"
              step="0.001"
              value={economy.swr}
              onChange={(event) => setValue(['economy', 'swr'], asNumber(event.target.value))}
            />
          </div>
          <div className="field">
            <label>Dividend Yield</label>
            <input
              type="number"
              step="0.001"
              value={economy.dividendYield}
              onChange={(event) => setValue(['economy', 'dividendYield'], asNumber(event.target.value))}
            />
          </div>
          <div className="field">
            <label>Home Appreciation</label>
            <input
              type="number"
              step="0.001"
              value={economy.homeAppreciation}
              onChange={(event) => setValue(['economy', 'homeAppreciation'], asNumber(event.target.value))}
            />
          </div>
        </div>
      </section>

      <section>
        <h3>Income & Savings</h3>
        <div className="field">
          <label>Gross Income</label>
          <input
            type="number"
            value={income.gross}
            onChange={(event) => setValue(['income', 'gross'], asNumber(event.target.value))}
          />
        </div>
        <div className="field">
          <label>Income Growth</label>
          <input
            type="number"
            step="0.001"
            value={income.growth}
            onChange={(event) => setValue(['income', 'growth'], asNumber(event.target.value))}
          />
        </div>
        <div className="field">
          <label>Employer RRSP Match</label>
          <input
            type="number"
            step="0.001"
            value={income.rrspMatch}
            onChange={(event) => setValue(['income', 'rrspMatch'], asNumber(event.target.value))}
          />
        </div>
        <div className="field">
          <label>Savings Mode</label>
          <div className="toggle-group">
            <button
              type="button"
              className={savings.mode === 'percent' ? 'active' : ''}
              onClick={() => toggleSavingsMode('percent')}
            >
              Percentage
            </button>
            <button
              type="button"
              className={savings.mode === 'dollar' ? 'active' : ''}
              onClick={() => toggleSavingsMode('dollar')}
            >
              Dollar Amounts
            </button>
          </div>
        </div>
        {savings.mode === 'percent' ? (
          <div className="field">
            <label>Savings Rate (% of net)</label>
            <input
              type="range"
              min="0"
              max="70"
              value={Math.round(savings.percent * 100)}
              onChange={(event) => setValue(['savings', 'percent'], Number(event.target.value) / 100)}
            />
            <p className="info-text">{Math.round(savings.percent * 100)}% of net earnings allocated to savings.</p>
          </div>
        ) : (
          <div>
            {['tfsa', 'rrsp', 'rdsp', 'fhsa', 'nonReg'].map((key) => (
              <div key={key} className="inline-group">
                <div className="field">
                  <label>{key.toUpperCase()} Amount</label>
                  <input
                    type="number"
                    value={savings.dollar[key]?.amount ?? 0}
                    onChange={(event) =>
                      setValue(['savings', 'dollar', key], {
                        ...(savings.dollar[key] ?? { frequency: 'monthly' }),
                        amount: asNumber(event.target.value),
                      })
                    }
                  />
                </div>
                <div className="field">
                  <label>Frequency</label>
                  <select
                    value={savings.dollar[key]?.frequency ?? 'monthly'}
                    onChange={(event) =>
                      setValue(['savings', 'dollar', key], {
                        ...(savings.dollar[key] ?? { amount: 0 }),
                        frequency: event.target.value,
                      })
                    }
                  >
                    {FREQUENCIES.map((freq) => (
                      <option key={freq.value} value={freq.value}>
                        {freq.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            ))}
            <p className="info-text">
              Equivalent savings rate: {Math.round(derived.savings.percentEquivalent * 1000) / 10}% of net earnings.
            </p>
          </div>
        )}
        <div className="field">
          <label>Allocation Priority</label>
          <input type="text" value={priorityString} onChange={handlePriorityChange} />
        </div>
        <div className="field">
          <label>RRSP Tax Refund Handling</label>
          <select
            value={savings.refundOption}
            onChange={(event) => setValue(['savings', 'refundOption'], event.target.value)}
          >
            <option value="reinvest_next_year">Re-invest next year</option>
            <option value="spend">Spend immediately</option>
          </select>
        </div>
      </section>

      <section>
        <h3>Accounts</h3>
        <div className="inline-group">
          <div className="field">
            <label>TFSA Balance</label>
            <input
              type="number"
              value={accounts.tfsaBalance}
              onChange={(event) => setValue(['accounts', 'tfsaBalance'], asNumber(event.target.value))}
            />
          </div>
          <div className="field">
            <label>TFSA Room</label>
            <input
              type="number"
              value={accounts.tfsaRoom}
              onChange={(event) => setValue(['accounts', 'tfsaRoom'], asNumber(event.target.value))}
            />
          </div>
        </div>
        <div className="inline-group">
          <div className="field">
            <label>RRSP Balance</label>
            <input
              type="number"
              value={accounts.rrspBalance}
              onChange={(event) => setValue(['accounts', 'rrspBalance'], asNumber(event.target.value))}
            />
          </div>
          <div className="field">
            <label>RRSP Room</label>
            <input
              type="number"
              value={accounts.rrspRoom}
              onChange={(event) => setValue(['accounts', 'rrspRoom'], asNumber(event.target.value))}
            />
          </div>
        </div>
        <div className="inline-group">
          <div className="field">
            <label>Non-Registered Balance</label>
            <input
              type="number"
              value={accounts.nonRegBalance}
              onChange={(event) => setValue(['accounts', 'nonRegBalance'], asNumber(event.target.value))}
            />
          </div>
          <div className="field">
            <label>Non-Registered ACB</label>
            <input
              type="number"
              value={accounts.acb}
              onChange={(event) => setValue(['accounts', 'acb'], asNumber(event.target.value))}
            />
          </div>
        </div>
        <div className="inline-group">
          <div className="field">
            <label>RDSP Balance</label>
            <input
              type="number"
              value={accounts.rdspBalance}
              onChange={(event) => setValue(['accounts', 'rdspBalance'], asNumber(event.target.value))}
            />
          </div>
          <div className="field">
            <label>FHSA Balance</label>
            <input
              type="number"
              value={accounts.fhsaBalance}
              onChange={(event) => setValue(['accounts', 'fhsaBalance'], asNumber(event.target.value))}
            />
          </div>
        </div>
      </section>

      <section>
        <h3>Housing</h3>
        <div className="field">
          <label>Status</label>
          <div className="toggle-group">
            <button
              type="button"
              className={housing.status === 'owner' ? 'active' : ''}
              onClick={() => setValue(['housing', 'status'], 'owner')}
            >
              Owner
            </button>
            <button
              type="button"
              className={housing.status === 'renter' ? 'active' : ''}
              onClick={() => setValue(['housing', 'status'], 'renter')}
            >
              Renter
            </button>
          </div>
        </div>
        {housing.status === 'owner' ? (
          <>
            <div className="field">
              <label>Home Value</label>
              <input
                type="number"
                value={housing.homeValue}
                onChange={(event) => setValue(['housing', 'homeValue'], asNumber(event.target.value))}
              />
            </div>
            <div className="field">
              <label>Mortgage Balance</label>
              <input
                type="number"
                value={housing.mortgageBalance}
                onChange={(event) => setValue(['housing', 'mortgageBalance'], asNumber(event.target.value))}
              />
            </div>
            <div className="field">
              <label>Annual Mortgage Payment</label>
              <input
                type="number"
                value={housing.annualMortgagePayment}
                onChange={(event) => setValue(['housing', 'annualMortgagePayment'], asNumber(event.target.value))}
              />
            </div>
          </>
        ) : (
          <div className="field">
            <label>Monthly Rent</label>
            <input
              type="number"
              value={housing.rent}
              onChange={(event) => setValue(['housing', 'rent'], asNumber(event.target.value))}
            />
          </div>
        )}
      </section>

      <section>
        <h3>Benefits</h3>
        <div className="inline-group">
          <div className="field">
            <label>CPP Annual</label>
            <input
              type="number"
              value={benefits.cppAnnual}
              onChange={(event) => setValue(['benefits', 'cppAnnual'], asNumber(event.target.value))}
            />
          </div>
          <div className="field">
            <label>CPP Start Age</label>
            <input
              type="number"
              value={benefits.cppStartAge}
              onChange={(event) => setValue(['benefits', 'cppStartAge'], asNumber(event.target.value))}
            />
          </div>
        </div>
        <div className="inline-group">
          <div className="field">
            <label>OAS Annual</label>
            <input
              type="number"
              value={benefits.oasAnnual}
              onChange={(event) => setValue(['benefits', 'oasAnnual'], asNumber(event.target.value))}
            />
          </div>
          <div className="field">
            <label>OAS Start Age</label>
            <input
              type="number"
              value={benefits.oasStartAge}
              onChange={(event) => setValue(['benefits', 'oasStartAge'], asNumber(event.target.value))}
            />
          </div>
        </div>
        <div className="field">
          <label>GIS Threshold</label>
          <input
            type="number"
            value={benefits.gisThreshold}
            onChange={(event) => setValue(['benefits', 'gisThreshold'], asNumber(event.target.value))}
          />
        </div>
      </section>

      <section>
        <h3>Retirement & Limits</h3>
        <div className="field">
          <label>Spending (Today&apos;s $)</label>
          <input
            type="number"
            value={retirement.spendingToday}
            onChange={(event) => setValue(['retirement', 'spendingToday'], asNumber(event.target.value))}
          />
        </div>
        <div className="inline-group">
          <div className="field">
            <label>Annual TFSA Limit</label>
            <input
              type="number"
              value={limits.annualTfsa}
              onChange={(event) => setValue(['limits', 'annualTfsa'], asNumber(event.target.value))}
            />
          </div>
          <div className="field">
            <label>RRSP Max Cap</label>
            <input
              type="number"
              value={limits.rrspMax}
              onChange={(event) => setValue(['limits', 'rrspMax'], asNumber(event.target.value))}
            />
          </div>
        </div>
      </section>

      <section>
        <h3>Simulation</h3>
        <div className="field">
          <label>Mode</label>
          <div className="toggle-group">
            <button
              type="button"
              className={simulation.mode === 'historical' ? 'active' : ''}
              onClick={() => setValue(['simulation', 'mode'], 'historical')}
            >
              Historical
            </button>
            <button
              type="button"
              className={simulation.mode === 'monteCarlo' ? 'active' : ''}
              onClick={() => setValue(['simulation', 'mode'], 'monteCarlo')}
            >
              Monte Carlo
            </button>
          </div>
        </div>
        {simulation.mode === 'monteCarlo' && (
          <>
            <div className="field">
              <label>Paths</label>
              <input
                type="number"
                value={simulation.paths}
                onChange={(event) => setValue(['simulation', 'paths'], asNumber(event.target.value))}
              />
            </div>
            <div className="field">
              <label>Return Std Dev</label>
              <input
                type="number"
                step="0.001"
                value={simulation.returnStdDev}
                onChange={(event) => setValue(['simulation', 'returnStdDev'], asNumber(event.target.value))}
              />
            </div>
            <div className="field">
              <label>Inflation Std Dev</label>
              <input
                type="number"
                step="0.001"
                value={simulation.inflationStdDev}
                onChange={(event) => setValue(['simulation', 'inflationStdDev'], asNumber(event.target.value))}
              />
            </div>
            <div className="field">
              <label>Random Seed</label>
              <input
                type="number"
                value={simulation.seed}
                onChange={(event) => setValue(['simulation', 'seed'], asNumber(event.target.value))}
              />
            </div>
          </>
        )}
      </section>
    </div>
  );
}
