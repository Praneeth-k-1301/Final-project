import { ethers } from 'ethers'
import dotenv from 'dotenv'

dotenv.config()

const CONTRACT_ABI = [
  'function releasePayment(address publisher, uint256 amountWei) public returns (bool)',
  'function getBalance() public view returns (uint256)',
  'function owner() public view returns (address)',
]

function rpcUrl() {
  if (process.env.RPC_URL) return process.env.RPC_URL
  if (process.env.INFURA_API_KEY) return `https://sepolia.infura.io/v3/${process.env.INFURA_API_KEY}`
  return null
}

async function checkContract() {
  console.log(`\n${'='.repeat(60)}`)
  console.log('🔍 Smart Contract Debugging Tool')
  console.log(`${'='.repeat(60)}\n`)

  const currentRpcUrl = rpcUrl()
  const PRIVATE_KEY = process.env.PRIVATE_KEY
  const CONTRACT_ADDRESS = process.env.CONTRACT_ADDRESS

  if (!currentRpcUrl || !PRIVATE_KEY || !CONTRACT_ADDRESS) {
    console.error('❌ Missing environment variables in .env file!')
    console.log('Required: RPC_URL (or INFURA_API_KEY), PRIVATE_KEY, CONTRACT_ADDRESS\n')
    process.exit(1)
  }

  console.log('📋 Configuration:')
  console.log(`   Contract Address: ${CONTRACT_ADDRESS}`)
  console.log('   Network: Sepolia Testnet')
  console.log(`   RPC Source: ${process.env.RPC_URL ? 'RPC_URL' : 'INFURA_API_KEY'}\n`)

  try {
    const provider = new ethers.JsonRpcProvider(currentRpcUrl)
    const wallet = new ethers.Wallet(PRIVATE_KEY, provider)
    const contract = new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, wallet)

    console.log('👛 Wallet Address:', wallet.address)
    const walletBalance = await provider.getBalance(wallet.address)
    console.log(`   Wallet Balance: ${ethers.formatEther(walletBalance)} ETH`)
    if (walletBalance === 0n) {
      console.log('   ⚠️  WARNING: Wallet has no ETH! Get Sepolia ETH from a faucet before sending transactions.')
    }
    console.log('')

    console.log('🔎 Checking Contract...')
    const code = await provider.getCode(CONTRACT_ADDRESS)
    if (code === '0x') {
      console.error('❌ ERROR: No contract found at this address!')
      console.log('\n📝 Suggested fixes:')
      console.log('   1. Deploy contracts/PublisherSettlement.sol to Sepolia')
      console.log('   2. Update CONTRACT_ADDRESS in your .env file')
      console.log(`   3. Verify deployment at https://sepolia.etherscan.io/address/${CONTRACT_ADDRESS}\n`)
      process.exit(1)
    }

    console.log('✅ Contract exists at this address')
    console.log(`   Bytecode size: ${(code.length - 2) / 2} bytes\n`)

    const contractBalance = await provider.getBalance(CONTRACT_ADDRESS)
    console.log('💰 Contract Balance:', ethers.formatEther(contractBalance), 'ETH')
    if (contractBalance === 0n) {
      console.log('   ⚠️  WARNING: Contract has no ETH. Valid clicks will fail if the backend tries real settlement.')
    } else if (contractBalance < ethers.parseEther('0.001')) {
      console.log('   ⚠️  WARNING: Contract balance is low and can only cover a few payouts.')
    } else {
      console.log('   ✅ Contract has sufficient balance for smoke testing')
    }
    console.log('')

    console.log('📄 Contract Functions:')
    console.log('   ✅ releasePayment(address publisher, uint256 amountWei)')

    try {
      const balance = await contract.getBalance()
      console.log('   ✅ getBalance() - Returns:', ethers.formatEther(balance), 'ETH')
    } catch {
      console.log('   ⚠️  getBalance() - View call failed or function missing')
    }

    try {
      const owner = await contract.owner()
      console.log('   ✅ owner() - Returns:', owner)
      if (owner.toLowerCase() !== wallet.address.toLowerCase()) {
        console.log('   ⚠️  WARNING: Your wallet is not the contract owner and may not be able to release funds.')
      }
    } catch {
      console.log('   ⚠️  owner() - View call failed or function missing')
    }

    console.log('')
    console.log('🎯 Test Transaction:')
    console.log('   Attempting to estimate gas for releasePayment...')
    const testAddress = wallet.address
    const testAmount = ethers.parseEther('0.001')
    const gasEstimate = await contract.releasePayment.estimateGas(testAddress, testAmount)
    console.log(`   ✅ Gas estimate: ${gasEstimate.toString()} units`)
    console.log(`   💵 Estimated cost at 20 gwei: ~${ethers.formatEther(gasEstimate * 20000000000n)} ETH`)
    console.log('\n✅ All checks passed. Contract appears compatible with the backend ABI.\n')
  } catch (error) {
    console.error('❌ ERROR:', error.message)
    console.log('\nPossible causes:')
    console.log('  - The deployed contract ABI does not match the backend expectation')
    console.log('  - The owner signer is incorrect')
    console.log('  - RPC credentials are wrong or rate-limited')
    console.log('  - The contract is underfunded or not deployed\n')
  }
}

checkContract()
