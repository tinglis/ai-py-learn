import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts';

const currency = new Intl.NumberFormat('en-CA', { style: 'currency', currency: 'CAD', maximumFractionDigits: 0 });

function buildData(projection, graph) {
  const rows = [];
  for (const entry of projection.portfolio) {
    if (!entry) continue;
    rows.push({
      age: entry.age,
      TFSA: entry.tfsaBalance,
      RRSP: entry.rrspBalance,
      RDSP: entry.rdspBalance,
      FHSA: entry.fhsaBalance,
      NonRegistered: entry.nonRegBalance,
      Total: entry.balance,
      NetWorth: graph.netWorth[entry.age] ?? entry.balance,
    });
  }
  return rows;
}

export default function PortfolioChart({ projection, graph, fireAge }) {
  const data = useMemo(() => buildData(projection, graph), [projection, graph]);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.2)" />
        <XAxis dataKey="age" stroke="#94a3b8" />
        <YAxis stroke="#94a3b8" tickFormatter={(value) => `${Math.round(value / 1000)}k`} />
        <Tooltip
          formatter={(value) => currency.format(value)}
          contentStyle={{ background: '#0f172a', borderRadius: '1rem', border: '1px solid rgba(148,163,184,0.3)' }}
        />
        <Legend />
        <ReferenceLine x={fireAge} stroke="#38bdf8" strokeDasharray="4 4" label={{ position: 'top', value: 'Retirement' }} />
        <Line type="monotone" dataKey="Total" stroke="#38bdf8" strokeWidth={3} dot={false} />
        <Line type="monotone" dataKey="NetWorth" stroke="#22d3ee" strokeWidth={2} dot={false} />
        <Line type="monotone" dataKey="TFSA" stroke="#a855f7" strokeWidth={1.5} dot={false} />
        <Line type="monotone" dataKey="RRSP" stroke="#f97316" strokeWidth={1.5} dot={false} />
        <Line type="monotone" dataKey="NonRegistered" stroke="#facc15" strokeWidth={1.5} dot={false} />
        <Line type="monotone" dataKey="RDSP" stroke="#2dd4bf" strokeWidth={1.5} dot={false} />
        <Line type="monotone" dataKey="FHSA" stroke="#60a5fa" strokeWidth={1.5} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
