import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

const currency = new Intl.NumberFormat('en-CA', { style: 'currency', currency: 'CAD', maximumFractionDigits: 0 });

export default function IncomeSourceChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.2)" />
        <XAxis dataKey="age" stroke="#94a3b8" />
        <YAxis stroke="#94a3b8" tickFormatter={(value) => `${Math.round(value / 1000)}k`} />
        <Tooltip
          formatter={(value) => currency.format(value)}
          contentStyle={{ background: '#0f172a', borderRadius: '1rem', border: '1px solid rgba(148,163,184,0.3)' }}
        />
        <Legend />
        <Bar dataKey="rrsp" stackId="withdrawals" fill="#f97316" name="RRSP" />
        <Bar dataKey="tfsa" stackId="withdrawals" fill="#a855f7" name="TFSA" />
        <Bar dataKey="nonReg" stackId="withdrawals" fill="#facc15" name="Non-Reg" />
        <Bar dataKey="cpp" stackId="income" fill="#38bdf8" name="CPP" />
        <Bar dataKey="oas" stackId="income" fill="#22d3ee" name="OAS" />
        <Bar dataKey="gis" stackId="income" fill="#34d399" name="GIS" />
      </BarChart>
    </ResponsiveContainer>
  );
}
