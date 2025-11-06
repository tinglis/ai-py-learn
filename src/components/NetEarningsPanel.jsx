import React from 'react';

const currency = new Intl.NumberFormat('en-CA', { style: 'currency', currency: 'CAD', maximumFractionDigits: 0 });

export default function NetEarningsPanel({ net, savings }) {
  return (
    <div className="panel">
      <h2>Net Earnings Snapshot</h2>
      <div className="summary-grid">
        <div className="summary-card">
          <h4>Annual Net</h4>
          <p>{currency.format(net.annual)}</p>
        </div>
        <div className="summary-card">
          <h4>Monthly</h4>
          <p>{currency.format(net.monthly)}</p>
        </div>
        <div className="summary-card">
          <h4>Semi-Monthly</h4>
          <p>{currency.format(net.semiMonthly)}</p>
        </div>
        <div className="summary-card">
          <h4>RRSP Refund</h4>
          <p>{currency.format(net.refund)}</p>
        </div>
      </div>
      <div style={{ marginTop: '1.5rem' }}>
        <div className="result-row">
          <span>CPP + EI Premiums</span>
          <strong>{currency.format(net.cpp + net.ei)}</strong>
        </div>
        <div className="result-row">
          <span>RRSP Contributions</span>
          <strong>{currency.format(net.rrspContribution)}</strong>
        </div>
        <div className="result-row">
          <span>Taxable Income</span>
          <strong>{currency.format(net.taxableIncome)}</strong>
        </div>
        <div className="result-row">
          <span>Net Tax</span>
          <strong>{currency.format(net.tax)}</strong>
        </div>
      </div>
      <p className="info-text" style={{ marginTop: '1rem' }}>
        RRSP contributions automatically reduce taxable income and boost your net earnings through the resulting refund. Select
        the refund behaviour in the savings panel to reinvest or spend it next year.
      </p>
      <p className="info-text">
        Current savings mode: <strong>{savings.mode}</strong>. Total planned contributions this year: <strong>{currency.format(savings.total)}</strong>.
      </p>
    </div>
  );
}
