# Decentralized Ad Network

This repository implements a working end-to-end prototype for a decentralized advertising system with:

- a **TalkingData-backed fraud-screening model**,
- an **Express settlement and campaign backend**,
- a **React dashboard** for advertisers, publishers, and security monitoring,
- and a **Solidity payout contract** aligned with the backend settlement ABI.

## Architecture

1. **`ml-api/`** trains or loads a persisted fraud-detection bundle from the TalkingData ad-click dataset.
2. **`backend/`** stores campaign state, derives click features, calls the ML API, and records simulated or on-chain settlements.
3. **`web/`** provides a Vite/React dashboard for campaign creation, click validation, wallet connection, and metrics.
4. **`contracts/`** contains `PublisherSettlement.sol`, the owner-controlled payout vault used for Sepolia settlement.

## Repository layout

- `backend/` – Express API, deterministic CID metadata, settlement ledger, click feature builder
- `ml-api/` – Flask inference/training service, persisted artifacts, evaluation plots
- `web/` – React dashboard connected to backend APIs
- `contracts/` – Solidity settlement contract source
- `talkingdata-adtracking-fraud-detection.zip` – local dataset archive used by the ML pipeline

## Current workflow

1. Advertiser creates a campaign through `POST /campaigns` or the dashboard.
2. Backend stores campaign metadata and assigns a deterministic content ID style value.
3. Publisher click events are converted into dataset-aligned features in `backend/click-utils.js`.
4. Backend calls `ml-api /predict` to classify the click.
5. Valid clicks trigger either:
   - **simulation mode** when no chain credentials are configured, or
   - **Sepolia settlement mode** through `releasePayment(address publisher, uint256 amountWei)`.

## Quick start

### 1. ML API

From `ml-api/`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The ML API runs on `http://localhost:5000`.

### 2. Backend

From `backend/`:

```bash
npm install
npm start
```

The backend runs on `http://localhost:3001`.

Optional backend environment variables:

- `PORT` – backend port, default `3001`
- `ML_API_URL` – default `http://localhost:5000`
- `RPC_URL` or `INFURA_API_KEY` – Sepolia RPC
- `PRIVATE_KEY` – settlement signer
- `CONTRACT_ADDRESS` – deployed `PublisherSettlement` contract

### 3. Frontend

From `web/`:

```bash
npm install
npm run dev
```

Optional frontend environment variable:

- `VITE_API_BASE_URL` – backend base URL, default `http://localhost:3001`

## Wallet and Sepolia setup (MetaMask)

1. Install MetaMask for your browser: https://metamask.io
2. Create a new wallet (or import an existing one). Secure your seed phrase offline.
3. Enable test networks in MetaMask: Settings → Advanced → “Show test networks”.
4. Switch to the Sepolia test network (Chain ID 11155111; hex 0xaa36a7).
5. Get test ETH for your wallet address from a faucet:
   - https://www.alchemy.com/faucets/ethereum-sepolia
   - https://www.infura.io/faucet/sepolia
   - https://sepoliafaucet.com/

Tip: The dashboard will ask MetaMask to switch to Sepolia automatically when you click “Connect wallet”.

## On‑chain settlement setup (deploy, fund, authorize)

You can run entirely in simulation mode without blockchain credentials. To enable real Sepolia settlement:

1) Set environment variables (backend):

Create `backend/.env` with either an RPC URL or an Infura key, plus your signer and contract address (filled after deploy):

```env
RPC_URL=https://sepolia.infura.io/v3/<YOUR_INFURA_KEY>
PRIVATE_KEY=0x<YOUR_SEPOLIA_SIGNER_PRIVATE_KEY>
CONTRACT_ADDRESS=0x<SET_AFTER_DEPLOY>
# Optional: ML API URL override
# ML_API_URL=http://localhost:5000
```

2) Deploy the settlement contract (script):

From `backend/` (Node 18+):

```bash
npm install
node deploy-publisher.js
```

The script compiles `contracts/PublisherSettlement.sol` (Solidity 0.8.20) with OpenZeppelin and deploys it to Sepolia using your `PRIVATE_KEY`. Copy the printed address into `CONTRACT_ADDRESS`.

3) Fund the contract vault:

Send a small amount of Sepolia ETH to the contract address using MetaMask, or run the helper:

```bash
node fund-publisher.js   # uses FUND_AMOUNT_ETH (default 0.01)
```

4) Authorize the backend signer (only if different from deployer):

The deployer is an oracle by default. If your backend uses a different `PRIVATE_KEY`, authorize it as an oracle by calling `setOracle(<BACKEND_ADDRESS>, true)` as the contract owner using Remix or Etherscan “Write Contract”.

## Run the full stack with on‑chain settlement

1. Start the ML API (port 5000): see Quick start → ML API
2. Start the backend (port 3001): `npm start` inside `backend/`
3. Start the frontend (Vite dev server): `npm run dev` inside `web/`
4. Open the dashboard, click “Connect wallet” (MetaMask will switch to Sepolia), then use “Validate & Settle”.

Notes:

- When blockchain is configured, validated clicks settle to the connected wallet on-chain.
- When blockchain is not configured, the backend remains in simulation mode and records off‑chain settlements.

## Troubleshooting

- “Wallet not connected” or “missing-wallet”: click “Connect wallet” in the UI.
- “Simulation only” mode in the Blockchain panel: set `RPC_URL`/`INFURA_API_KEY`, `PRIVATE_KEY`, and `CONTRACT_ADDRESS` in `backend/.env`, then restart the backend.
- “vault-exhausted”: the contract vault is out of funds—send Sepolia ETH to the contract (or run `node fund-publisher.js`).
- “already-settled”: idempotency guard rejected a duplicate event/click.
- RPC rate‑limited: wait and retry or use a different provider.
- Gas price too high: try again later; settlement is queued and serialized by the backend.

## Verified checks

The following were executed successfully during the refactor:

- `python -m unittest test_pipeline.py` in `ml-api/`
- `node --test click-utils.test.js` in `backend/`
- `node --check server.js` in `backend/`
- IDE diagnostics on `web/src/App.jsx`, `web/src/App.css`, and `web/src/index.css`
- IDE diagnostics on `backend/debug-contract.js` and `contracts/PublisherSettlement.sol`

## Known limitations and honest status

- The dashboard source is updated and diagnostics-clean, but **frontend build/lint were not completed** in this workspace because:
  - `web/node_modules` is currently absent, and
  - the local shell/npm environment fails with `execvpe(/bin/bash) failed`.
- Real Sepolia settlement requires a funded wallet, a funded deployed contract, and valid RPC credentials.
- The project currently uses a **deterministic CID-style hash** for ad asset metadata rather than a live IPFS pinning pipeline.
- The ML labels are **weakly supervised proxies** derived from TalkingData attribution outcomes plus anomaly heuristics, which is appropriate for this dataset but not equivalent to a manually labeled fraud corpus.

## Results produced by the ML service

The ML pipeline persists:

- a model bundle,
- model metadata,
- ROC curve,
- precision-recall curve,
- and a confusion matrix SVG

under `ml-api/artifacts/`.

## Contract alignment

The backend and debug utility both expect this ABI surface:

- `releasePayment(address publisher, uint256 amountWei)`
- `getBalance()`
- `owner()`

That interface is implemented in `contracts/PublisherSettlement.sol`.
# Final-project
