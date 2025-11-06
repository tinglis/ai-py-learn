# Maple FIRE Core Logic

This repository now ships the foundational client-side engines **and** an interactive React dashboard that brings the Maple FIRE specification to life. All projections, simulations, and visualisations run locally in the browser using the shared JavaScript modules so personal data never leaves the device.

## Running the UI

```bash
npm install
npm run dev
```

The Vite dev server launches at <http://localhost:5173> with a guided onboarding wizard. Adjust any slider or input in the left-hand panel to see the net-earnings breakdown, FIRE numbers, and charts update instantly.

## Available Modules

### FIRE Math (`src/lib/fire.js`)
- `calculateFireNumber(spending, swr)` – Standard FIRE target in nominal dollars.
- `calculateCoastFireNumber({ fireNumber, yearsToGrow, inflationAdjustedGrowthRate })` – Required balance at the start of a coasting phase.

### Tax & Benefit Engines
- `src/lib/tax.js` – Configuration-backed federal/provincial income tax calculations.
- `src/lib/benefits.js` – CPP/EI payroll deductions, OAS/GIS clawbacks, RRIF factors, and GIS helper logic sourced from `src/config/benefits/<taxYear>/core.json`.
- `src/lib/rdsp.js` – RDSP grants, bonds, and 10-year assistance holdback modelling.

### Income & Savings (`src/lib/income.js`, `src/lib/savings.js`)
- Circular net-earnings solver, RRSP refund feedback, and CPP/EI deduction helpers.
- Savings synchronisation utilities for percentage/dollar modes plus allocation with room limits.

### Projection & Simulation (`src/lib/projection.js`)
- Single-year engine that respects account rules (TFSA, RRSP, non-registered, RDSP, FHSA) including dividend taxation, employer matches, and room updates.
- Multi-year projection covering accumulation, coasting, and retirement withdrawal sequencing with GIS-aware strategies, prior-year GIS income tracking, RRSP refund reinvestment, and CPP/OAS income indexation.
- Home equity and mortgage series are produced alongside per-account balances for dashboard visualisations.
- Historical simulation loader operating on `src/data/historical_data.json`, plus helpers to build dashboard graph data.
- Monte Carlo simulator with configurable paths, seeded randomness for testing, percentile outputs, and ruin-age tracking.

### Startup Verification (`src/lib/startupChecks.js`)
- Self-test harness that mirrors the product brief’s acceptance criteria. The React entry point invokes `runStartupChecks()` before rendering and crashes with a descriptive error if any invariant regresses.

## Configuration Data
- Tax brackets: `src/config/tax/<taxYear>/federal.json` and `provinces.json`.
- Benefits & contribution limits: `src/config/benefits/<taxYear>/core.json`.
- Historical market & inflation series: `src/data/historical_data.json`.

All loaders cache parsed JSON so updates to the configuration files are reflected immediately without code changes.

## Tests

The project uses the Node.js test runner. Execute everything with:

```bash
npm test
```

Test coverage currently exercises:

- FIRE and Coast FIRE number helpers.
- Federal and provincial tax spot checks.
- CPP/EI deductions, RRSP refund feedback loop, and net-earnings convergence.
- Savings synchronisation plus allocation room limits.
- Yearly account engine behaviour (growth, withdrawals, dividend tax, room updates).
- Multi-phase projection scenarios, employer match handling, GIS-aware withdrawals with prior-year GIS income lag, CPP/OAS indexing, and net-worth graph aggregation.
- Historical simulation loader/outcomes plus Monte Carlo percentile reporting.
- RDSP grant/bond logic and 10-year assistance clawback triggers.
- Startup guardrails executed via `runStartupChecks()`.

These scenarios mirror the startup verification outline from the Maple FIRE specification and guard against regressions as new UI layers integrate with the engines.
