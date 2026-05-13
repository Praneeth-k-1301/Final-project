# Backend service

The backend is an Express API that coordinates campaigns, click validation, settlement recording, and ML integration.

## Responsibilities

- persist campaigns, click logs, and settlements in `data/store.json`
- derive dataset-aligned click features in `click-utils.js`
- call the Flask ML API for fraud screening
- execute or simulate settlement depending on chain configuration
- expose dashboard-facing metrics and historical records

## Scripts

- `npm start` – run `server.js`
- `npm run dev` – run `server.js` in watch mode
- `npm run debug` – execute `debug-contract.js`
- `npm test` – run `node --test`

## Environment variables

- `PORT` – defaults to `3001`
- `ML_API_URL` – defaults to `http://localhost:5000`
- `RPC_URL` – preferred Sepolia RPC URL
- `INFURA_API_KEY` – fallback if `RPC_URL` is not provided
- `PRIVATE_KEY` – backend signer for real settlement
- `CONTRACT_ADDRESS` – deployed `PublisherSettlement` address

If `RPC_URL`/`PRIVATE_KEY`/`CONTRACT_ADDRESS` are missing, settlement automatically falls back to simulation mode.

## API endpoints

### Health and metadata

- `GET /health`
- `GET /metrics`
- `GET /ml/metadata`

### Campaign inventory

- `GET /campaigns`
- `GET /get-ads` (legacy alias)
- `POST /campaigns`
- `POST /upload-ad` (legacy alias)

Required campaign fields:

- `title`
- `image`
- `targetUrl`

Optional fields used by the dashboard and ML feature builder include:

- `budgetEth`
- `payoutEth`
- `appCode`
- `osCode`
- `channelCode`
- `advertiserAddress`

### Click validation

- `GET /clicks`
- `POST /clicks/validate`
- `POST /simulate-click` (dashboard-friendly alias)

Expected click payload fields include:

- `campaignId` or `adId`
- `walletAddress`
- `visitorId`
- `clickTime`
- `deviceType`
- `userAgent`
- `platform`

## Storage model

`data/store.json` contains three top-level collections:

- `campaigns`
- `clickLogs`
- `settlements`

The backend seeds a starter campaign if the store file does not yet exist.

## Settlement behavior

- **Simulation mode**: records payout intent and latency without broadcasting a transaction
- **On-chain mode**: calls `releasePayment(walletAddress, amountWei)` and records tx hash, gas used, and explorer URL when successful

## Contract debug utility

`debug-contract.js` validates:

- RPC connectivity
- wallet balance
- deployed bytecode presence
- contract balance
- `getBalance()` and `owner()` compatibility
- gas estimation for `releasePayment(address,uint256)`

Run it with:

```bash
npm run debug
```

## Validation status

Completed successfully:

- `node --test click-utils.test.js`
- `node --check server.js`
- IDE diagnostics on `debug-contract.js`

Not completed locally:

- full backend runtime smoke against a live Sepolia contract, because credentials/funding were not provided in this workspace