# 🔧 Troubleshooting: Smart Contract Call Failed

## Your Error:

```
Smart contract error: missing revert data (action="estimateGas", data=null, 
reason=null, transaction={ ... }, code=CALL_EXCEPTION)
```

## What This Means:

The "missing revert data" or "CALL_EXCEPTION" error occurs when:

1. ❌ **Contract doesn't exist** at that address on Sepolia testnet
2. ❌ **Contract has no ETH** to pay users
3. ❌ **Contract function doesn't match** the expected ABI
4. ❌ **Your wallet isn't the owner** (if contract has owner restrictions)
5. ❌ **Contract bytecode has issues** or is not properly deployed

---

## 🔍 Step 1: Run the Debug Tool

We've created a debugging script to diagnose the issue:

```bash
cd backend
npm run debug
```

This will check:
- ✅ If contract exists at the address
- ✅ Contract balance
- ✅ Your wallet balance
- ✅ If functions are callable
- ✅ Gas estimation

---

## 🔍 Step 2: Check Contract on Etherscan

Your contract address: `0xf589A9b16e22bD0200Edd1b738401eB6b0046B9C`

Visit: [https://sepolia.etherscan.io/address/0xf589A9b16e22bD0200Edd1b738401eB6b0046B9C](https://sepolia.etherscan.io/address/0xf589A9b16e22bD0200Edd1b738401eB6b0046B9C)

### What to check:

1. **Does the page show "Contract"?**
   - ✅ YES → Contract is deployed, go to Step 3
   - ❌ NO → Contract not deployed, see Solution A below

2. **What is the balance?**
   - ✅ > 0.1 ETH → Good, go to Step 3
   - ❌ 0 ETH → See Solution B below

3. **Can you see transactions?**
   - Check if there's a "Contract Creation" transaction
   - Check if contract has received any ETH

---

## 💡 Solution A: Contract Not Deployed

Your `.env` has this address: `0xf589A9b16e22bD0200Edd1b738401eB6b0046B9C`

**If contract doesn't exist, you need to deploy it:**

### Deploy with Remix IDE:

1. Go to [https://remix.ethereum.org/](https://remix.ethereum.org/)

2. Create new file `AdPayment.sol`:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AdPayment {
    address public owner;
    
    event PaymentReleased(address indexed advertiser, uint256 amount);
    
    constructor() {
        owner = msg.sender;
    }
    
    function releasePayment(address advertiser, uint256 amount) public returns (bool) {
        require(msg.sender == owner, "Only owner can release payment");
        require(address(this).balance >= amount, "Insufficient balance");
        require(advertiser != address(0), "Invalid address");
        
        payable(advertiser).transfer(amount);
        emit PaymentReleased(advertiser, amount);
        return true;
    }
    
    receive() external payable {}
    
    function getBalance() public view returns (uint256) {
        return address(this).balance;
    }
    
    function withdraw() public {
        require(msg.sender == owner, "Only owner");
        payable(owner).transfer(address(this).balance);
    }
}
```

3. **Compile** (Solidity Compiler tab, version 0.8.0+)

4. **Deploy** (Deploy & Run tab):
   - Environment: **Injected Provider - MetaMask**
   - Network in MetaMask: **Sepolia Test Network**
   - Click **Deploy**
   - Confirm transaction in MetaMask

5. **Copy the new contract address** from Remix

6. **Update `.env`**:
   ```env
   CONTRACT_ADDRESS=0x_your_new_contract_address_here
   ```

7. **Restart backend server**

---

## 💡 Solution B: Contract Has No Balance

If contract exists but has 0 ETH balance, you need to fund it:

### Fund Contract:

1. **In MetaMask:**
   - Click "Send"
   - Paste contract address: `0xf589A9b16e22bD0200Edd1b738401eB6b0046B9C`
   - Amount: **0.1 ETH** (enough for ~100 payments of 0.001 ETH)
   - Confirm transaction

2. **Or use Remix:**
   - Open your deployed contract in Remix
   - In "VALUE" field (top), enter: `0.1` and select `Ether`
   - Click red **"receive"** button
   - Confirm in MetaMask

3. **Verify balance:**
   ```bash
   npm run debug
   ```

---

## 💡 Solution C: Contract Function Mismatch

If contract exists and has balance, but function doesn't match:

### Check in Remix:

1. Connect to existing contract:
   - Deploy tab → "At Address"
   - Paste: `0xf589A9b16e22bD0200Edd1b738401eB6b0046B9C`
   - Click "At Address"

2. Check if `releasePayment` function appears

3. Try calling it manually with:
   - `advertiser`: Your wallet address
   - `amount`: `1000000000000000` (0.001 ETH in Wei)

If it fails, your contract code is different than expected.

---

## 💡 Solution D: You're Not the Owner

Some contracts restrict who can call functions:

### Check ownership:

```bash
npm run debug
```

Look for the "owner()" output. If your wallet isn't the owner:

1. **Option 1:** Deploy a new contract where YOU are the owner
2. **Option 2:** Use the original deployer's private key (if you have access)
3. **Option 3:** Modify contract to allow anyone to call `releasePayment`

---

## 🔄 Quick Fix Workflow:

```bash
# 1. Check what's wrong
cd backend
npm run debug

# 2. Based on output, either:
#    - Deploy new contract (if none exists)
#    - Fund contract (if balance is 0)
#    - Update CONTRACT_ADDRESS in .env

# 3. Restart backend
npm run dev

# 4. Test in frontend
# Open http://localhost:5173 and click "Simulate Click"
```

---

## ✅ Expected Success Output:

When everything works, you should see:

```
Calling smart contract releasePayment...
Contract Address: 0x...
Advertiser Address: 0x...
Payment Amount: 0.001 ETH
Contract Balance: 0.1 ETH
Transaction hash: 0x...
Waiting for confirmation...
Transaction confirmed in block: 12345678
```

---

## 📞 Still Stuck?

1. **Run debug tool**: `npm run debug`
2. **Share the output** - it will tell you exactly what's wrong
3. **Check Etherscan** for your contract address
4. **Verify MetaMask** is on Sepolia testnet
5. **Check wallet** has Sepolia ETH for gas

---

## 🔗 Useful Links:

- **Sepolia Etherscan**: https://sepolia.etherscan.io/
- **Remix IDE**: https://remix.ethereum.org/
- **Sepolia Faucet**: https://sepoliafaucet.com/
- **Your Contract**: https://sepolia.etherscan.io/address/0xf589A9b16e22bD0200Edd1b738401eB6b0046B9C
