const { expect } = require('chai')
const { ethers } = require('hardhat')

describe('PublisherSettlement', function () {
  it('registers a campaign and settles a click with replay protection and events', async function () {
    const [owner, oracle, publisher] = await ethers.getSigners()

    const PublisherSettlement = await ethers.getContractFactory('PublisherSettlement')
    const settlement = await PublisherSettlement.deploy({ value: ethers.parseEther('1') })
    await settlement.waitForDeployment()

    // Owner configures an additional oracle address (backend signer)
    await expect(settlement.setOracle(oracle.address, true))
      .to.emit(settlement, 'OracleAuthorized')
      .withArgs(oracle.address, true)

    const campaignId = ethers.id('cmp-hardhat-demo')
    const budgetWei = ethers.parseEther('0.1')
    const payoutPerClickWei = ethers.parseEther('0.001')

    await settlement.registerCampaign(
      campaignId,
      owner.address,
      publisher.address,
      'ipfs://demo-campaign',
      budgetWei,
      payoutPerClickWei,
    )

    const stored = await settlement.campaigns(campaignId)
    expect(stored.advertiser).to.equal(owner.address)
    expect(stored.payoutReceiver).to.equal(publisher.address)
    expect(stored.active).to.equal(true)

    const eventId = ethers.id('click-1')
    const payoutAmount = payoutPerClickWei

    // Before settlement, click must not be marked as settled
    expect(await settlement.isSettled(eventId)).to.equal(false)

    const oracleSettlement = settlement.connect(oracle)

    await expect(oracleSettlement.releasePaymentForEvent(eventId, publisher.address, payoutAmount))
      .to.emit(settlement, 'ClickValidated')
      .withArgs(eventId, publisher.address, payoutAmount)
      .and.to.emit(settlement, 'PaymentReleased')

    // After settlement, event is marked as processed and publisher has a pending balance
    expect(await settlement.isSettled(eventId)).to.equal(true)
    const pending = await settlement.pendingWithdrawals(publisher.address)
    expect(pending).to.equal(payoutAmount)

    // Second settlement attempt for the same eventId must revert with custom error
    await expect(
      oracleSettlement.releasePaymentForEvent(eventId, publisher.address, payoutAmount),
    ).to.be.revertedWithCustomError(settlement, 'AlreadySettled').withArgs(eventId)

    // Publisher can withdraw their funds (pull-payment + nonReentrant)
    await settlement.connect(publisher).withdrawPublisher()

    // After withdrawing, pending balance must be zero
    const pendingAfter = await settlement.pendingWithdrawals(publisher.address)
    expect(pendingAfter).to.equal(0n)
  })
})
