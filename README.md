# Maple FIRE Planner (Easy Start Guide)

This project is a **browser-based** Canadian FIRE planning tool. Once it is running, a web page opens where you can plug in your
numbers and see the projections update instantly. All calculations happen on your own machine—no accounts, servers, or extra
setup required.

## Quick Start (No Experience Needed)

1. **Install Node.js (only once).**
   * Go to <https://nodejs.org> and download the **LTS** installer.
   * Run the installer and accept the defaults.
2. **Open this project folder** in your terminal or command prompt.
3. **Install the project dependencies** (React, charts, build tools):

   ```bash
   npm install
   ```

4. **Launch the planner:**

   ```bash
   npm run dev
   ```

   * The command prints a local link such as `http://localhost:5173`.
   * Copy that address into your browser (Chrome, Edge, Firefox, Safari, etc.).
   * A short onboarding wizard walks you through the essential inputs.

5. **Stop the planner** any time by returning to the terminal window and pressing `Ctrl + C`.

> 💡 While the dev server is running you can change values in the UI and the charts update automatically. There is no “build” step
> required during normal tinkering.

## What You Will See in the Browser

* **Left panel** – sliders and fields for age, income, savings style (percent vs. dollars), account balances, housing details, and
  government benefit ages.
* **Middle panel** – a simple take-home pay breakdown that spells out gross income → CPP/EI → RRSP deduction → taxes → net
  earnings.
* **Right panel** – headline results: FIRE number, Coast FI number, projected FIRE/Coast ages, success rate, and summary notes.
* **Charts** –
  * Portfolio growth (per account, total portfolio, and net worth).
  * Stacked retirement income sources (RRSP, TFSA, non-registered, CPP, OAS, GIS, RDSP).

Scroll to the bottom for simulation options. You can run **Historical** or **Monte Carlo** simulations by selecting a mode and
pressing **Run Simulation**. Results appear beside the button along with a percentile chart so you can compare outcomes.

## Useful Commands

| Task | Command |
| --- | --- |
| Install dependencies | `npm install` |
| Start the dev server (opens the browser app) | `npm run dev` |
| Run automated tests | `npm test` |
| Create a production build | `npm run build` |
| Preview the production build | `npm run preview` |

The tests run a set of self-checks that cover tax calculations, savings allocation logic, projection phases, RDSP rules, and both
simulation engines. The app also runs a “startup check” automatically—if a core rule ever breaks you will see a clear error before
the UI loads.

## Folder Tour (For When You Want to Dive Deeper)

* `src/lib/` – calculation engines (tax, benefits, savings, projections, simulations, RDSP helpers).
* `src/ui/` – small helper functions used by the React components.
* `src/components/` – the dashboard panels, charts, and onboarding wizard.
* `src/config/` – JSON files for tax brackets, benefit amounts, and contribution limits.
* `src/data/historical_data.json` – blended historical return/inflation series used for the simulations.
* `tests/` – Node.js test suites that mirror the acceptance checks from the Maple FIRE spec.

That’s it! Start the dev server, open the provided link in your browser, and you are ready to experiment with your FIRE plan.
