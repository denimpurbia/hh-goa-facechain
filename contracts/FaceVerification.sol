// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title FaceVerification
 * @dev Stores and verifies tamper-evident SHA-256 fingerprints of face verification metadata.
 * Biometric embeddings and raw images are NEVER stored on-chain.
 */
contract FaceVerification {
    // Mapping from cryptographic metadata SHA-256 hash (bytes32) to verification status
    mapping(bytes32 => bool) private verifiedRecords;
    
    // Mapping from hash to timestamp of registration
    mapping(bytes32 => uint256) private recordTimestamps;
    
    // Mapping from hash to recorder address
    mapping(bytes32 => address) private recordRecorders;

    // Emitted when a new metadata fingerprint is recorded on-chain
    event RecordStored(
        bytes32 indexed dataHash,
        uint256 timestamp,
        address indexed recorder
    );

    /**
     * @notice Store a cryptographic fingerprint of verified face search metadata
     * @param dataHash 32-byte SHA-256 hash of the canonical metadata
     */
    function storeRecord(bytes32 dataHash) external {
        require(dataHash != bytes32(0), "Invalid data hash");
        require(!verifiedRecords[dataHash], "Record already exists on chain");
        
        verifiedRecords[dataHash] = true;
        recordTimestamps[dataHash] = block.timestamp;
        recordRecorders[dataHash] = msg.sender;

        emit RecordStored(dataHash, block.timestamp, msg.sender);
    }

    /**
     * @notice Check whether a given metadata hash exists and is verified on-chain
     * @param dataHash 32-byte SHA-256 hash to verify
     * @return isVerified True if the record exists on the blockchain
     */
    function verifyRecord(bytes32 dataHash) external view returns (bool isVerified) {
        return verifiedRecords[dataHash];
    }

    /**
     * @notice Retrieve details for a recorded hash
     * @param dataHash 32-byte SHA-256 hash
     * @return exists True if recorded
     * @return timestamp Block timestamp when recorded
     * @return recorder Address that recorded the fingerprint
     */
    function getRecordDetails(bytes32 dataHash) external view returns (
        bool exists,
        uint256 timestamp,
        address recorder
    ) {
        return (
            verifiedRecords[dataHash],
            recordTimestamps[dataHash],
            recordRecorders[dataHash]
        );
    }
}
