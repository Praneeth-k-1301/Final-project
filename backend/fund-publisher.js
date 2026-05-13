import { ethers } from 'ethers'
import dotenv from 'dotenv'

dotenv.config()

function rpcUrl() {
  if (process.env.RPC_URL) return process.env.RPC_URL
  if (process.env.INFURA_API_KEY) return `https://sepolia.infura.io/v3/${process.env.INFURA_API_KEY}`
  return null
}

async function main() {
  const currentRpcUrl = rpcUrl()
  const PRIVATE_KEY = process.env.PRIVATE_KEY
  const CONTRACT_ADDRESS = process.env.CONTRACT_ADDRESS
  const FUND_AMOUNT_ETH = process.env.FUND_AMOUNT_ETH || '0.01'

  if (!currentRpcUrl || !PRIVATE_KEY || !CONTRACT_ADDRESS) {
    console.error('❌ Missing RPC_URL/INFURA_API_KEY, PRIVATE_KEY, or CONTRACT_ADDRESS in environment')
    process.exit(1)
  }

  const provider = new ethers.JsonRpcProvider(currentRpcUrl)
  const wallet = new ethers.Wallet(PRIVATE_KEY, provider)

  const fundAmount = ethers.parseEther(FUND_AMOUNT_ETH)
  const walletBalance = await provider.getBalance(wallet.address)
  const contractBalanceBefore = await provider.getBalance(CONTRACT_ADDRESS)

  console.log('\n🚰 Funding PublisherSettlement contract...')
  console.log('RPC URL:', currentRpcUrl)
  console.log('Deployer wallet:', wallet.address)
  console.log('Wallet balance:', ethers.formatEther(walletBalance), 'ETH')
  console.log('Contract address:', CONTRACT_ADDRESS)
  console.log('Contract balance (before):', ethers.formatEther(contractBalanceBefore), 'ETH')
  console.log('Requested fund amount:', FUND_AMOUNT_ETH, 'ETH')

  if (walletBalance <= fundAmount) {
    console.error('❌ Wallet balance too low to fund contract with this amount')
    process.exit(1)
  }

  const tx = await wallet.sendTransaction({ to: CONTRACT_ADDRESS, value: fundAmount })
  console.log('\n📦 Funding tx hash:', tx.hash)
  const receipt = await tx.wait()
  console.log('   Block:', receipt.blockNumber)
  console.log('   Gas used:', receipt.gasUsed.toString())

  const contractBalanceAfter = await provider.getBalance(CONTRACT_ADDRESS)
  console.log('\n✅ Funding complete')
  console.log('Contract balance (after):', ethers.formatEther(contractBalanceAfter), 'ETH')
}

main().catch((error) => {
  console.error('❌ Funding failed:', error)
  process.exit(1)
})
