# Web dashboard

This frontend is a React + Vite dashboard for interacting with the decentralized ad network prototype.

## Main capabilities

- connect a browser wallet
- view backend and ML health
- create campaigns
- browse active campaign inventory
- trigger click validation from the publisher workflow
- inspect recent fraud decisions, metrics, and settlements
- switch between advertiser, publisher, and security-focused views

## Tech stack

- React
- Vite
- ethers.js

## Configuration

Optional environment variable:

- `VITE_API_BASE_URL` – backend base URL, defaults to `http://localhost:3001`

## Scripts

- `npm run dev`
- `npm run build`
- `npm run lint`
- `npm run preview`

## Runtime expectations

The dashboard expects the backend to provide:

- `/campaigns`
- `/metrics`
- `/clicks`
- `/ml/metadata`
- `/health`
- `/simulate-click`

For wallet features, the browser should expose `window.ethereum` (for example via MetaMask).

## Views in the current UI

- **Publisher** – browse campaigns and validate clicks
- **Advertiser** – create campaigns and inspect budgets/payouts
- **Security** – review fraud logs, latest model decision, and benchmark metadata

## Validation status

Confirmed:

- IDE diagnostics on `src/App.jsx`, `src/App.css`, and `src/index.css` are clean.

Blocked in this workspace:

- `npm run build`
- `npm run lint`

Reason:

- `web/node_modules` is not currently present, and
- the local npm shell path fails with `execvpe(/bin/bash) failed`, which prevents normal script execution.

## Notes

- The dashboard is already wired to the new backend/ML endpoints.
- Settlement display supports both simulation records and real Sepolia receipts.
- If you install dependencies in a healthy shell, the next recommended checks are `npm run build` and `npm run lint`.