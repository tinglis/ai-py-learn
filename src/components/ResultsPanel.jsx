import React from 'react';

const currency = new Intl.NumberFormat('en-CA', { style: 'currency', currency: 'CAD', maximumFractionDigits: 0 });

function formatAge(value) {
  return Number.isFinite(value) ? `${value}` : '—';
}

function formatPercent(value) {
  if (!Number.isFinite(value)) return '—';
  return `${value.toFixed(1)}%`;
}

export default function ResultsPanel({ fire, simulation, state }) {
  const { profile } = state;
  const simulationDetails = simulation.mode === 'monteCarlo'
    ? `Success: ${formatPercent(simulation.successRate)} • Median Ending Balance: ${currency.format(simulation.percentiles?.p50 ?? 0)}`
    : `Historical success rate: ${formatPercent(simulation.successRate)}`;

  return (
    <div>
      <h2>Key Outcomes</h2>
      <div className="summary-grid">
        <div className="summary-card">
          <h4>FIRE Number (Today)</h4>
          <p>{currency.format(fire.fireNumberToday)}</p>
        </div>
        <div className="summary-card">
          <h4>FIRE Number (Age {profile.fireAge})</h4>
          <p>{currency.format(fire.fireNumberAtFireAge)}</p>
        </div>
        <div className="summary-card">
          <h4>Projected FIRE Age</h4>
          <p>{formatAge(fire.projectedFireAge)}</p>
        </div>
        <div className="summary-card">
          <h4>Coast FIRE (Today)</h4>
          <p>{currency.format(fire.coastNumberToday)}</p>
        </div>
        <div className="summary-card">
          <h4>Projected Coast Age</h4>
          <p>{formatAge(fire.projectedCoastAge)}</p>
        </div>
      </div>
      <div style={{ marginTop: '1.5rem' }}>
        <h3>Simulation</h3>
        <p className="info-text">Mode: {simulation.mode === 'monteCarlo' ? 'Monte Carlo (stochastic)' : 'Historical (deterministic)'}</p>
        <p>{simulationDetails}</p>
        {simulation.mode === 'monteCarlo' && simulation.percentiles && (
          <ul className="info-text">
            <li>5th percentile: {currency.format(simulation.percentiles.p5 ?? 0)}</li>
            <li>95th percentile: {currency.format(simulation.percentiles.p95 ?? 0)}</li>
            <li>Ruin paths: {simulation.ruinAges.length}</li>
          </ul>
        )}
      </div>
    </div>
  );
}
