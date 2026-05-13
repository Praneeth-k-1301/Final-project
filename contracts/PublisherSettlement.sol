// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/// @title PublisherSettlement
/// @notice Settlement vault for validated ad-click payouts with oracle access control,
///         replay protection, pausing, pull-payments, basic on-chain campaign registry,
///         and reentrancy safety.
contract PublisherSettlement is ReentrancyGuard {
	// --------------------------------- Ownership & roles ---------------------------------
	address public owner;
	mapping(address => bool) public oracles;

	// --------------------------------- Campaign registry ---------------------------------
	/// @notice Minimal on-chain campaign metadata used for auditing and attribution.
	struct Campaign {
		address advertiser; // campaign owner / buyer of traffic
		address payoutReceiver; // default publisher wallet or aggregator
		string ipfsCid; // IPFS CID for creative / metadata bundle
		uint256 budgetWei; // declared budget for the campaign (informational)
		uint256 payoutPerClickWei; // nominal payout per validated click (informational)
		uint256 spentWei; // amount logically spent on this campaign (informational)
		bool active; // whether new traffic should be accepted for this campaign
	}

	/// @dev Campaigns are keyed by an off-chain generated ID (e.g., keccak256 of a
	///      human-readable string such as "cmp-seeded-demo"). The backend is free to
	///      choose any convention as long as it passes the same ID here.
	mapping(bytes32 => Campaign) public campaigns;

	// --------------------------------- Replay protection ---------------------------------
	/// @notice Tracks whether an event/click ID has been processed (valid or fraud).
	mapping(bytes32 => bool) public settledEvents;

	// --------------------------------- Pull-payment ledger ---------------------------------
	/// @notice Pending publisher withdrawals (credited but not yet withdrawn).
	mapping(address => uint256) public pendingWithdrawals;
	uint256 public totalPendingWithdrawals;

	// --------------------------------- Pause state ---------------------------------
	bool public paused;

	// --------------------------------- Events ---------------------------------
	event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
	event DepositReceived(address indexed sender, uint256 amountWei, uint256 newBalanceWei);
	/// @notice Emitted when the oracle/backend declares an off-chain click as valid
	///         and requests settlement for the associated publisher.
	event ClickValidated(bytes32 indexed eventId, address indexed publisher, uint256 amountWei);
	/// @notice Emitted when a payout has been credited to a publisher's pull-payment
	///         balance. This is the on-chain record of payment settlement.
	event PaymentReleased(bytes32 indexed eventId, address indexed publisher, uint256 amountWei, uint256 remainingBalanceWei);
	event Withdrawal(address indexed recipient, uint256 amountWei, uint256 remainingBalanceWei);
	event OracleAuthorized(address indexed oracle, bool enabled);
	event PauseChanged(bool paused);
	event FraudDetected(bytes32 indexed eventId, address indexed publisher);

	// --------------------------------- Errors ---------------------------------
	error NotOwner();
	error NotOracle();
	error ZeroAddress();
	error ZeroEventId();
	error ZeroAmount();
	error AlreadySettled(bytes32 eventId);
	error InsufficientBalance(uint256 requested, uint256 available);
	error TransferFailed();
	error Paused();
	error NotPaused();
	error NothingToWithdraw();

	// --------------------------------- Modifiers ---------------------------------
	modifier onlyOwner() {
		if (msg.sender != owner) revert NotOwner();
		_;
	}

	modifier onlyOracle() {
		if (!oracles[msg.sender]) revert NotOracle();
		_;
	}

	modifier whenNotPaused() {
		if (paused) revert Paused();
		_;
	}

	modifier whenPaused() {
		if (!paused) revert NotPaused();
		_;
	}

	// --------------------------------- Constructor & funding ---------------------------------
	constructor() payable {
		owner = msg.sender;
		// Deployer is the initial oracle/back-end signer.
		oracles[msg.sender] = true;
		emit OwnershipTransferred(address(0), msg.sender);
		emit OracleAuthorized(msg.sender, true);
		if (msg.value > 0) {
			emit DepositReceived(msg.sender, msg.value, address(this).balance);
		}
	}

	receive() external payable {
		// Allow funding even while paused.
		if (msg.value > 0) {
			emit DepositReceived(msg.sender, msg.value, address(this).balance);
		}
	}

	// --------------------------------- Views ---------------------------------
	function getBalance() public view returns (uint256) {
		return address(this).balance;
	}

	function isSettled(bytes32 eventId) public view returns (bool) {
		return settledEvents[eventId];
	}

	// --------------------------------- Role management ---------------------------------
	/// @notice Configure an oracle/backend address that is allowed to trigger settlement.
	function setOracle(address oracle, bool enabled) external onlyOwner {
		if (oracle == address(0)) revert ZeroAddress();
		oracles[oracle] = enabled;
		emit OracleAuthorized(oracle, enabled);
	}

	// --------------------------------- Campaign management ---------------------------------
	/// @notice Register a new campaign with basic metadata. This does not move funds;
	///         advertisers transfer ETH to this contract separately to fund payouts.
	/// @param campaignId Off-chain generated ID (e.g., keccak256 of a human-readable string).
	/// @param advertiser Address that owns the campaign.
	/// @param payoutReceiver Default wallet to receive payouts (can be an aggregator).
	/// @param ipfsCid IPFS CID referencing the ad creative and configuration.
	/// @param budgetWei Informational budget for the campaign (not strictly enforced on-chain).
	/// @param payoutPerClickWei Nominal payout per validated click.
	function registerCampaign(
		bytes32 campaignId,
		address advertiser,
		address payoutReceiver,
		string calldata ipfsCid,
		uint256 budgetWei,
		uint256 payoutPerClickWei
	) external onlyOwner whenNotPaused {
		if (campaignId == bytes32(0)) revert ZeroEventId();
		if (advertiser == address(0) || payoutReceiver == address(0)) revert ZeroAddress();
		if (payoutPerClickWei == 0) revert ZeroAmount();
		Campaign storage campaign = campaigns[campaignId];
		// Disallow overwriting an existing campaign; use updateCampaign for that.
		if (campaign.advertiser != address(0)) revert AlreadySettled(campaignId); // reuse error as generic "already exists"
		campaign.advertiser = advertiser;
		campaign.payoutReceiver = payoutReceiver;
		campaign.ipfsCid = ipfsCid;
		campaign.budgetWei = budgetWei;
		campaign.payoutPerClickWei = payoutPerClickWei;
		campaign.active = true;
		// spentWei starts at 0
	}

	/// @notice Update mutable campaign fields such as metadata and declared budget.
	function updateCampaign(
		bytes32 campaignId,
		string calldata ipfsCid,
		uint256 budgetWei,
		uint256 payoutPerClickWei
	) external onlyOwner whenNotPaused {
		Campaign storage campaign = campaigns[campaignId];
		if (campaign.advertiser == address(0)) revert AlreadySettled(campaignId); // treat missing as error
		if (payoutPerClickWei == 0) revert ZeroAmount();
		campaign.ipfsCid = ipfsCid;
		campaign.budgetWei = budgetWei;
		campaign.payoutPerClickWei = payoutPerClickWei;
	}

	/// @notice Activate or deactivate a campaign.
	function setCampaignStatus(bytes32 campaignId, bool active_) external onlyOwner {
		Campaign storage campaign = campaigns[campaignId];
		if (campaign.advertiser == address(0)) revert AlreadySettled(campaignId);
		campaign.active = active_;
	}

	// --------------------------------- Pausing ---------------------------------
	/// @notice Pause settlement and publisher withdrawals in an emergency.
	function pause() external onlyOwner whenNotPaused {
		paused = true;
		emit PauseChanged(true);
	}

	/// @notice Unpause normal operations.
	function unpause() external onlyOwner whenPaused {
		paused = false;
		emit PauseChanged(false);
	}

	// --------------------------------- Settlement (checks-effects) ---------------------------------
	/// @notice Legacy settlement without an explicit event/click ID.
	/// @dev Credits a pull-payment balance instead of sending ETH directly.
	function releasePayment(address publisher, uint256 amountWei)
		public
		onlyOracle
		whenNotPaused
		returns (bool)
	{
		return _creditPayment(bytes32(0), publisher, amountWei, false);
	}

	/// @notice Settlement using a unique event/click ID with replay protection.
	function releasePaymentForEvent(bytes32 eventId, address publisher, uint256 amountWei)
		public
		onlyOracle
		whenNotPaused
		returns (bool)
	{
		if (eventId == bytes32(0)) revert ZeroEventId();
		if (settledEvents[eventId]) revert AlreadySettled(eventId);
			// Emit an explicit validation event so off-chain indexers can distinguish
			// between validated clicks and settlements even when using pull-payments.
			emit ClickValidated(eventId, publisher, amountWei);
		return _creditPayment(eventId, publisher, amountWei, true);
	}

	/// @dev Internal helper that performs checks and credits a pull-payment balance.
	function _creditPayment(bytes32 eventId, address publisher, uint256 amountWei, bool trackEvent) internal returns (bool) {
		if (publisher == address(0)) revert ZeroAddress();
		if (amountWei == 0) revert ZeroAmount();

		uint256 available = address(this).balance;
		// Ensure total reserved (including this payout) does not exceed contract balance.
		if (available < totalPendingWithdrawals + amountWei) {
			uint256 freeBalance = available - totalPendingWithdrawals;
			revert InsufficientBalance(amountWei, freeBalance);
		}

		if (trackEvent) {
			settledEvents[eventId] = true;
		}

		pendingWithdrawals[publisher] += amountWei;
		totalPendingWithdrawals += amountWei;

		emit PaymentReleased(eventId, publisher, amountWei, address(this).balance);
		return true;
	}

	/// @notice Mark a click/event as fraudulent; prevents later settlement.
	function reportFraud(bytes32 eventId, address publisher)
		external
		onlyOracle
		whenNotPaused
	{
		if (eventId == bytes32(0)) revert ZeroEventId();
		if (settledEvents[eventId]) revert AlreadySettled(eventId);
		settledEvents[eventId] = true;
		emit FraudDetected(eventId, publisher);
	}

	// --------------------------------- Pull-payment withdrawals ---------------------------------
	/// @notice Publisher withdraws their accumulated payouts.
	function withdrawPublisher() external nonReentrant whenNotPaused returns (bool) {
		uint256 amount = pendingWithdrawals[msg.sender];
		if (amount == 0) revert NothingToWithdraw();

		// Effects
		pendingWithdrawals[msg.sender] = 0;
		totalPendingWithdrawals -= amount;

		// Interactions
		(bool success, ) = payable(msg.sender).call{value: amount}("");
		if (!success) revert TransferFailed();

		emit Withdrawal(msg.sender, amount, address(this).balance);
		return true;
	}

	/// @notice Owner can withdraw unreserved funds (not already owed to publishers).
	function withdraw(address payable recipient, uint256 amountWei)
		external
		onlyOwner
		nonReentrant
		returns (bool)
	{
		if (recipient == address(0)) revert ZeroAddress();
		if (amountWei == 0) revert ZeroAmount();

		uint256 available = address(this).balance;
		uint256 freeBalance = available - totalPendingWithdrawals;
		if (freeBalance < amountWei) revert InsufficientBalance(amountWei, freeBalance);

		(bool success, ) = recipient.call{value: amountWei}("");
		if (!success) revert TransferFailed();

		emit Withdrawal(recipient, amountWei, address(this).balance);
		return true;
	}

	// --------------------------------- Ownership ---------------------------------
	function transferOwnership(address newOwner) external onlyOwner {
		if (newOwner == address(0)) revert ZeroAddress();
		address previousOwner = owner;
		owner = newOwner;
		emit OwnershipTransferred(previousOwner, newOwner);
	}
}