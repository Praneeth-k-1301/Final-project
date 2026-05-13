import { ethers } from 'ethers'
import dotenv from 'dotenv'
import fs from 'fs'
import path from 'path'
import solc from 'solc'
import { fileURLToPath } from 'url'

dotenv.config()

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

function rpcUrl() {
  if (process.env.RPC_URL) return process.env.RPC_URL
  if (process.env.INFURA_API_KEY) return `https://sepolia.infura.io/v3/${process.env.INFURA_API_KEY}`
  return null
}

function compilePublisherSettlement() {
  const sourcePath = path.resolve(__dirname, '../contracts/PublisherSettlement.sol')
  const source = fs.readFileSync(sourcePath, 'utf8')

  /** @type {import('solc').CompilerInput} */
  const input = {
    language: 'Solidity',
    sources: {
      'PublisherSettlement.sol': { content: source },
    },
    settings: {
      optimizer: { enabled: true, runs: 200 },
      outputSelection: {
        '*': {
          '*': ['abi', 'evm.bytecode', 'evm.deployedBytecode'],
        },
      },
    },
  }

  function findImports(importPath) {
    const localPath = path.resolve(__dirname, '../contracts', importPath)
    if (fs.existsSync(localPath)) {
      return { contents: fs.readFileSync(localPath, 'utf8') }
    }
    const nodePath = path.resolve(__dirname, 'node_modules', importPath)
    if (fs.existsSync(nodePath)) {
      return { contents: fs.readFileSync(nodePath, 'utf8') }
    }
    return { error: `File not found: ${importPath}` }
  }

  const output = JSON.parse(solc.compile(JSON.stringify(input), { import: findImports }))

  if (output.errors && output.errors.length > 0) {
    const errors = output.errors.filter((e) => e.severity === 'error')
    if (errors.length > 0) {
      console.error('❌ Solidity compilation failed:')
      for (const err of errors) console.error(err.formattedMessage || err.message)
      process.exit(1)
    }
  }

  const contract = output.contracts['PublisherSettlement.sol'].PublisherSettlement
  if (!contract) {
    console.error('❌ Compiled contract not found in output')
    process.exit(1)
  }

  const abi = contract.abi
  const bytecode = `0x${contract.evm.bytecode.object}`
  return { abi, bytecode }
}

async function main() {
  const currentRpcUrl = rpcUrl()
  const PRIVATE_KEY = process.env.PRIVATE_KEY

  if (!currentRpcUrl || !PRIVATE_KEY) {
    console.error('❌ Missing RPC_URL/INFURA_API_KEY or PRIVATE_KEY in environment')
    process.exit(1)
  }

  console.log('\n🚀 Deploying PublisherSettlement to Sepolia...')
  console.log('RPC URL:', currentRpcUrl)

  const { abi, bytecode } = compilePublisherSettlement()

  const provider = new ethers.JsonRpcProvider(currentRpcUrl)
  const wallet = new ethers.Wallet(PRIVATE_KEY, provider)

  const walletBalance = await provider.getBalance(wallet.address)
  console.log('👛 Deployer wallet:', wallet.address)
  console.log('   Balance:', ethers.formatEther(walletBalance), 'ETH')

  if (walletBalance === 0n) {
    console.error('❌ Wallet has 0 ETH on Sepolia. Please fund it before deploying.')
    process.exit(1)
  }

  const factory = new ethers.ContractFactory(abi, bytecode, wallet)

  console.log('\n📦 Sending deployment transaction...')
  const contract = await factory.deploy()
  const deployTx = contract.deploymentTransaction()
  console.log('   Tx hash:', deployTx.hash)

  console.log('⏳ Waiting for deployment confirmation...')
  const receipt = await deployTx.wait()
  const address = await contract.getAddress()
  console.log('\n✅ Contract deployed!')
  console.log('   Address:', address)
  console.log('   Block:', receipt.blockNumber)
  console.log('   Gas used:', receipt.gasUsed.toString())

  console.log('\nYou can now set CONTRACT_ADDRESS to this value in your backend/ML env.')
}

main().catch((error) => {
  console.error('❌ Deployment failed:', error)
  process.exit(1)
})
