# Developer Essentials

## Quick Reference

<CustomDocCardContainer>
    <CustomDocCard
        icon={<Monad />}
        link="/developer-essentials/network-information"
        title="Network Information"
        description="Links and canonical deployments"
    />
        <CustomDocCard
        icon={<Monad />}
        link="/tooling-and-infra"
        title="Tooling & Infra"
        description="Third-party infra supporting Monad Testnet"
    />
    <CustomDocCard
        icon={<Monad />}
        link="/reference"
        title="RPC Reference"
        description="JSON-RPC API"
    />
</CustomDocCardContainer>

## Optimizing for Monad

<CustomDocCardContainer>
    <CustomDocCard
        icon={<Monad />}
        link="/developer-essentials/best-practices"
        title="Best Practices"
        description="Recommendations for building high-performance apps"
    />
    <CustomDocCard
        icon={<Monad />}
        link="/developer-essentials/gas-on-monad"
        title="Gas on Monad"
    />
    <CustomDocCard
        icon={<Monad />}
        link="/developer-essentials/differences"
        title="Differences between Monad & Ethereum"
    />
</CustomDocCardContainer>

## Monad's Architecture in Depth

<CustomDocCardContainer>
    <CustomDocCard
        icon={<Monad />}
        link="/introduction/monad-for-developers"
        title="Monad for Devs"
        description="One-page summary of Monad architecture"
    />
    <CustomDocCard
        icon={<Monad />}
        link="/monad-arch/consensus"
        title="Consensus"
        description="MonadBFT, async execution, mempool, and other consensus details"
    />
    <CustomDocCard
        icon={<Monad />}
        link="/monad-arch/execution"
        title="Execution"
        description="Optimistic parallel execution, MonadDB, and other execution details"
    />
</CustomDocCardContainer>

## Changelog

<CustomDocCardContainer>
    <CustomDocCard
        icon={<Monad />}
        link="/developer-essentials/changelog"
        title="Testnet Changelog"
    />
</CustomDocCardContainer>


## Get Support

<CustomDocCardContainer>
    <CustomDocCard
        icon={<Discord />}
        link="https://discord.gg/monaddev"
        title="Discord"
        description="Join other Monad developers on Discord"
    />
    <CustomDocCard
        icon={<Telegram />}
        link="https://t.me/+06Kv7meSPn80M2U8"
        title="Telegram"
        description="Join other Monad developers on Telegram"
    />
</CustomDocCardContainer>

---

# Best Practices for Building High Performance Apps

URL: https://docs.monad.xyz/developer-essentials/best-practices.md

# Best Practices for Building High Performance Apps

## Configure web hosting to keep costs under control

- Vercel and Railway provide convenient serverless platforms for hosting your application, abstracting away the logistics of web hosting relative to using a cloud provider directly. You may end up paying a premium for the convenience, especially at higher volumes.
- AWS and other cloud providers offer more flexibility and commodity pricing.
- Before choosing any service, check pricing and be aware that many providers offer loss-leader pricing on lower volumes, but then charge higher rates once you hit a certain threshold.
  - For example, suppose there is a $20 plan that includes 1 TB per month of data transfer, with $0.20 per GB beyond that. Be aware that the second TB (and onward) will cost $200. If the next tier up says "contact us", don't assume the next tier up will be charging $20 per TB.
  - If you are building a high-traffic app and you aren't careful about serving static files more cheaply, it will be easy to exceed the loss-leader tier and pay much more than you expect.
- For production deployments on AWS, consider:
  - Amazon S3 + CloudFront for static file hosting and CDN
  - AWS Lambda for serverless functions
  - Amazon ECS or EKS for containerized applications
  - Amazon RDS for database needs
  - This setup typically provides granular cost control and scalability for high-traffic applications.

## Avoid unnecessary RPC calls to methods with static responses

- `eth_chainId` always returns `10143`
- `eth_gasPrice` always returns `52 * 10^9`
- `eth_maxPriorityFeePerGas` always returns `2 * 10^9`

## Use a hardcoded value instead of `eth_estimateGas` call if gas usage is static

Many on-chain actions have a fixed gas cost. The simplest example is that a transfer of native tokens always costs 21,000 gas, but there are many others. This makes it unnecessary to call `eth_estimateGas` for each transaction.

Use a hardcoded value instead, as suggested [here](docs/developer-essentials/gas-on-monad.md#set-the-gas-limit-explicitly-if-it-is-constant). Eliminating an `eth_estimateGas` call substantially speeds up the user workflow in the wallet, and avoids a potential bad behavior in some wallets when `eth_estimateGas` reverts (discussed in the linked page).


## Use an indexer instead of repeatedly calling `eth_getLogs` to listen for your events

Below is a quickstart guide for the most popular data indexing solutions. Please view the [indexer docs](/tooling-and-infra/indexers/) for more details.


### Using Allium
:::note
See also: [**Allium**](docs/tooling-and-infra/indexers/common-data.md#allium)

You'll need an Allium account, which you can request [here](https://www.allium.so/contact).
:::

- Allium Explorer
    - Blockchain analytics platform that provides SQL-based access to historical blockchain data (blocks, transactions, logs, traces, and contracts).
    - You can create Explorer APIs through the [GUI](https://app.allium.so/explorer/api) to query and analyze historical blockchain data. When creating a Query for an API [here](https://app.allium.so/explorer/queries) (using the `New` button), select `Monad Testnet` from the chain list.
    - Relevant docs:
        - [Explorer Documentation](https://docs.allium.so/data-products-analytics/allium-explorer)
        - [Explorer API Tutorial](https://docs.allium.so/data-products-analytics/allium-explorer/explorer-api/explorer-api-user-tutorial)
- Allium Datastreams
    - Provides real-time blockchain data streams (including blocks, transactions, logs, traces, contracts, and balance snapshots) through Kafka, Pub/Sub, and Amazon SNS.
    - [GUI](https://app.allium.so/developer/streams/new) to create new streams for onchain data. When creating a stream, select the relevant `Monad Testnet` topics from the `Select topics` dropdown.
    - Relevant docs:
        - [Datastreams Documentation](https://docs.allium.so/data-products-real-time/allium-datastreams)
        - [Getting Started with Google Pub/Sub](https://docs.allium.so/data-products-real-time/allium-datastreams/kafka-pubsub/getting-started-with-google-pub-sub)
- Allium Developers
    - Enables fetching wallet transaction activity and tracking balances (native, ERC20, ERC721, ERC1155).
    - For the request's body, use `monad_testnet` as the `chain` parameter.
    - Relevant docs:
        - [API Key Setup Guide](https://docs.allium.so/data-products-real-time/allium-developer/wallet-apis-1#getting-started)
        - [Wallet APIs Documentation](https://docs.allium.so/data-products-real-time/allium-developer/wallet-apis)


### Using Envio HyperIndex
:::note
See also: [**Envio**](docs/tooling-and-infra/indexers/indexing-frameworks.md#envio) and [**Guide: How to use Envio HyperIndex to build a token transfer notification bot**](guides/indexers/tg-bot-using-envio.md)
:::
- Follow the [quick start](https://docs.envio.dev/docs/HyperIndex/contract-import) to create an indexer. In the `config.yaml` file, use network ID `10143` to select Monad Testnet.
- Example configuration
    - Sample `config.yaml` file
        ```yaml
        name: your-indexers-name
        networks:
        - id: 10143  # Monad Testnet
          # Optional custom RPC configuration - only add if default indexing has issues
          # rpc_config:
          #   url: YOUR_RPC_URL_HERE  # Replace with your RPC URL (e.g., from Alchemy)
          #   interval_ceiling: 50     # Maximum number of blocks to fetch in a single request
          #   acceleration_additive: 10  # Speed up factor for block fetching
          #   initial_block_interval: 10  # Initial block fetch interval size
          start_block: 0  # Replace with the block you want to start indexing from
          contracts:
          - name: YourContract  # Replace with your contract name
            address:
            - 0x0000000000000000000000000000000000000000  # Replace with your contract address
            # Add more addresses if needed for multiple deployments of the same contract
            handler: src/EventHandlers.ts
            events:
            # Replace with your event signatures
            # Format: EventName(paramType paramName, paramType2 paramName2, ...)
            # Example: Transfer(address from, address to, uint256 amount)
            # Example: OrderCreated(uint40 orderId, address owner, uint96 size, uint32 price, bool isBuy)
            - event: EventOne(paramType1 paramName1, paramType2 paramName2)
            # Add more events as needed
        ```
        
    - Sample `EventHandlers.ts`
        ```tsx
        import {
          YourContract,
          YourContract_EventOne,
        } from "generated";
        
        // Handler for EventOne
        // Replace parameter types and names based on your event definition
        YourContract.EventOne.handler(async ({ event, context }) => {
          // Create a unique ID for this event instance
          const entity: YourContract_EventOne = {
            id: `${event.chainId}_${event.block.number}_${event.logIndex}`,
            // Replace these with your actual event parameters
            paramName1: event.params.paramName1,
            paramName2: event.params.paramName2,
            // Add any additional fields you want to store
          };
        
          // Store the event in the database
          context.YourContract_EventOne.set(entity);
        })
        
        // Add more event handlers as needed
        ```
        
- Important: The `rpc_config` section under a network (check `config.yaml` sample) is optional and should only be configured if you experience issues with the default Envio setup. This configuration allows you to:
    - Use your own RPC endpoint
    - Configure block fetching parameters for better performance
- Relevant docs:
    - [Overview](https://docs.envio.dev/docs/HyperIndex/overview)


### Using GhostGraph
:::note
See also: [**Ghost**](docs/tooling-and-infra/indexers/indexing-frameworks.md#ghost)
:::
- Relevant docs:
    - [Getting Started](https://docs.tryghost.xyz/category/-getting-started)
    - [Setting up a GhostGraph Indexer on Monad Testnet](/guides/indexers/ghost#setting-up-ghostgraph-indexing)


### Using Goldsky
:::note
See also: [**Goldsky**](docs/tooling-and-infra/indexers/common-data.md#goldsky)
:::

- Goldsky Subgraphs
    - To deploy a Goldsky subgraph follow [this guide](https://docs.goldsky.com/subgraphs/deploying-subgraphs#from-source-code).
    - As the network identifier please use `monad-testnet`. For subgraph configuration examples, refer to [The Graph Protocol section](#using-the-graphs-subgraph) below.
    - For information about querying Goldsky subgraphs, see the [GraphQL API documentation](https://docs.goldsky.com/subgraphs/graphql-endpoints).
- Goldsky Mirror
    - Enables direct streaming of on-chain data to your database.
    - For the chain name in the `dataset_name` field when creating a `source` for a pipeline, use `monad_testnet` (check below example)
    - Example `pipeline.yaml` config file
        ```yaml
        name: monad-testnet-erc20-transfers
        apiVersion: 3
        sources:
          monad_testnet_erc20_transfers:
            dataset_name: monad_testnet.erc20_transfers
            filter: address = '0x0' # Add erc20 contract address. Multiple addresses can be added with 'OR' operator: address = '0x0' OR address = '0x1'
            version: 1.2.0
            type: dataset
            start_at: earliest
        
        # Data transformation logic (optional)
        transforms:
          select_relevant_fields:
            sql: |
              SELECT
                  id,
                  address,
                  event_signature,
                  event_params,
                  raw_log.block_number as block_number,
                  raw_log.block_hash as block_hash,
                  raw_log.transaction_hash as transaction_hash
              FROM
                  ethereum_decoded_logs
            primary_key: id
        
        # Sink configuration to specify where data goes eg. DB
        sinks:
          postgres:
            type: postgres
            table: erc20_transfers
            schema: goldsky
            secret_name: A_POSTGRESQL_SECRET
            from: select_relevant_fields
        ```
    - Relevant docs:
        - [Getting Started with Mirror](https://docs.goldsky.com/mirror/create-a-pipeline#goldsky-cli)
        - [Data Streaming Guides](https://docs.goldsky.com/mirror/guides/)


### Using QuickNode Streams
:::note
See also: [**QuickNode Streams**](docs/tooling-and-infra/indexers/common-data.md#quicknode)
:::

- On your QuickNode Dashboard, select `Streams` > `Create Stream`. In the create stream UI, select Monad Testnet under Network. Alternatively, you can use the [Streams REST API](https://www.quicknode.com/docs/streams/rest-api/getting-started) to create and manage streams—use `monad-testnet` as the network identifier.
- You can consume a Stream by choosing a destination during stream creation. Supported destinations include Webhooks, S3 buckets, and PostgreSQL databases. Learn more [here](https://www.quicknode.com/docs/streams/destinations).
- Relevant docs:
    - [Getting Started](https://www.quicknode.com/docs/streams/getting-started)


### Using The Graph's Subgraph
:::note
See also: [**The Graph**](docs/tooling-and-infra/indexers/indexing-frameworks.md#the-graph)
:::
- Network ID to be used for Monad Testnet: `monad-testnet`
- Example configuration
    - Sample `subgraph.yaml` file
        ```yaml
        specVersion: 1.2.0
        indexerHints:
          prune: auto
        schema:
          file: ./schema.graphql
        dataSources:
          - kind: ethereum
            name: YourContractName # Replace with your contract name
            network: monad-testnet # Monad testnet configuration
            source:
              address: "0x0000000000000000000000000000000000000000" # Replace with your contract address
              abi: YourContractABI # Replace with your contract ABI name
              startBlock: 0 # Replace with the block where your contract was deployed/where you want to index from
            mapping:
              kind: ethereum/events
              apiVersion: 0.0.9
              language: wasm/assemblyscript
              entities:
                # List your entities here - these should match those defined in schema.graphql
                # - Entity1
                # - Entity2
              abis:
                - name: YourContractABI # Should match the ABI name specified above
                  file: ./abis/YourContract.json # Path to your contract ABI JSON file
              eventHandlers:
                # Add your event handlers here, for example:
                # - event: EventName(param1Type, param2Type, ...)
                #   handler: handleEventName
              file: ./src/mapping.ts # Path to your event handler implementations
        ```
        
    - Sample `mappings.ts` file
        ```tsx
        import {
          // Import your contract events here
          // Format: EventName as EventNameEvent
          EventOne as EventOneEvent,
          // Add more events as needed
        } from "../generated/YourContractName/YourContractABI" // Replace with your contract name, abi name you supplied in subgraph.yaml
        
        import {
          // Import your schema entities here
          // These should match the entities defined in schema.graphql
          EventOne,
          // Add more entities as needed
        } from "../generated/schema"
        
        /**
          * Handler for EventOne
          * Update the function parameters and body according to your event structure
          */
        export function handleEventOne(event: EventOneEvent): void {
          // Create a unique ID for this entity
          let entity = new EventOne(
            event.transaction.hash.concatI32(event.logIndex.toI32())
          )
          
          // Map event parameters to entity fields
          // entity.paramName = event.params.paramName
          
          // Example:
          // entity.sender = event.params.sender
          // entity.amount = event.params.amount
        
          // Add metadata fields
          entity.blockNumber = event.block.number
          entity.blockTimestamp = event.block.timestamp
          entity.transactionHash = event.transaction.hash
        
          // Save the entity to the store
          entity.save()
        }
        
        /**
          * Add more event handlers as needed
          * Format:
          * 
          * export function handleEventName(event: EventNameEvent): void {
          *   let entity = new EventName(
          *     event.transaction.hash.concatI32(event.logIndex.toI32())
          *   )
          *   
          *   // Map parameters
          *   entity.param1 = event.params.param1
          *   entity.param2 = event.params.param2
          *   
          *   // Add metadata
          *   entity.blockNumber = event.block.number
          *   entity.blockTimestamp = event.block.timestamp
          *   entity.transactionHash = event.transaction.hash
          *   
          *   entity.save()
          * }
          */
        ```
        
    - Sample `schema.graphql` file
        ```graphql
        # Define your entities here
        # These should match the entities listed in your subgraph.yaml
        
        # Example entity for a generic event
        type EventOne @entity(immutable: true) {
          id: Bytes!
          
          # Add fields that correspond to your event parameters
          # Examples with common parameter types:
          # paramId: BigInt!              # uint256, uint64, etc.
          # paramAddress: Bytes!          # address
          # paramFlag: Boolean!           # bool
          # paramAmount: BigInt!          # uint96, etc.
          # paramPrice: BigInt!           # uint32, etc.
          # paramArray: [BigInt!]!        # uint[] array
          # paramString: String!          # string
          
          # Standard metadata fields
          blockNumber: BigInt!
          blockTimestamp: BigInt!
          transactionHash: Bytes!
        }
        
        # Add more entity types as needed for different events
        # Example based on Transfer event:
        # type Transfer @entity(immutable: true) {
        #   id: Bytes!
        #   from: Bytes!                  # address
        #   to: Bytes!                    # address
        #   tokenId: BigInt!              # uint256
        #   blockNumber: BigInt!
        #   blockTimestamp: BigInt!
        #   transactionHash: Bytes!
        # }
        
        # Example based on Approval event:
        # type Approval @entity(immutable: true) {
        #   id: Bytes!
        #   owner: Bytes!                 # address
        #   approved: Bytes!              # address
        #   tokenId: BigInt!              # uint256
        #   blockNumber: BigInt!
        #   blockTimestamp: BigInt!
        #   transactionHash: Bytes!
        # }
- Relevant docs:
    - [Quickstart](https://thegraph.com/docs/en/subgraphs/quick-start/)

### Using thirdweb's Insight API
:::note
See also: [**thirdweb**](docs/tooling-and-infra/indexers/common-data.md#thirdweb)
:::

- REST API offering a wide range of on-chain data, including events, blocks, transactions, token data (such as transfer transactions, balances, and token prices), contract details, and more.
- Use chain ID `10143` for Monad Testnet when constructing request URLs.
    - Example: `https://insight.thirdweb.com/v1/transactions?chain=10143`
- Relevant docs:
    - [Get started](https://portal.thirdweb.com/insight/get-started)
    - [API playground](https://playground.thirdweb.com/insight)


## Manage nonces locally if sending multiple transactions in quick succession

:::note
This only applies if you are setting nonces manually. If you are delegating this to the wallet, no need to worry about this.
:::

- `eth_getTransactionCount` only updates after a transaction is finalized. If you have multiple transactions from the same wallet in short succession, you should implement local nonce tracking. 


## Submit multiple transactions concurrently

If you are submitting a series of transactions, instead submitting sequentially, implement concurrent transaction submission for improved efficiency.

Before:

```jsx
for (let i = 0; i < TIMES; i++) {
  const tx_hash = await WALLET_CLIENT.sendTransaction({
    account: ACCOUNT,
    to: ACCOUNT_1,
    value: parseEther('0.1'),
    gasLimit: BigInt(21000),
    baseFeePerGas: BigInt(50000000000),
    chain: CHAIN,
    nonce: nonce + Number(i),
  })
}
```

After:

```jsx
const transactionsPromises = Array(BATCH_SIZE)
  .fill(null)
  .map(async (_, i) => {
    return await WALLET_CLIENT.sendTransaction({
      to: ACCOUNT_1,
      value: parseEther('0.1'),
      gasLimit: BigInt(21000),
      baseFeePerGas: BigInt(50000000000),
      chain: CHAIN,
      nonce: nonce + Number(i),
    })
  })
const hashes = await Promise.all(transactionsPromises)
```
---

# Testnet Changelog

## v0.9.0 [2025-03-14]

Notable user-facing changes:
* Max contract size increased to 128kb from 24kb (enabled 3/14/25 19:00 GMT)
  * [example 123 kb contract](https://testnet.monadexplorer.com/address/0x0E820425e07E4a992C661E4B970e13002d6e52B9?tab=Contract)

Notable internal changes:
* Improvements to RPC performance for `eth_call`
* Fixed a bug in `debug_traceTransaction`. Previously, within a transaction, only the first 100 calls were being traced
* Dataplane v2 - simpler and more efficient implementation; small performance improvement in broadcast time
* Statesync improvements to mitigate negative performance effects on upstream validator nodes


## v0.8.1 [2025-02-14]

Notable user-facing changes:
* Block time reduced to 500 ms from 1s (enabled 2/14/25 19:00 GMT)
* Block gas limit reduced to 150M from 300M (to keep gas limit consistent) (enabled 2/14/25 19:00 GMT)
* Transactions are charged based on gas limit, not gas consumed (enabled 2/14/25 19:00 GMT)

Notable internal changes:
* UX improvements for transaction status. RPC nodes track status of transactions submitted to them in order to provide updates to users.
---

# Differences between Monad and Ethereum

## Transactions

1. Transactions are charged based on gas limit rather than gas usage, i.e. total tokens deducted from the sender's balance is `value + gas_price * gas_limit`. As discussed in [Gas in Monad](docs/developer-essentials/gas-on-monad.md), this is a DOS-prevention measure for asynchronous execution. This may be revised before mainnet.

2. Transaction type 3 (EIP-4844 type aka blob transactions) are not supported. This is temporary.

3. Max contract size is 128kb (up from 24kb in Ethereum).

4. There is no global mempool. For efficiency, transactions are forwarded to the next few leaders as described in [Local Mempool](/monad-arch/consensus/local-mempool).


## RPC

See: [RPC Differences](/reference/rpc-differences)

---

# Faucet

## Monad Faucet

## Phantom Faucet

---

# Gas on Monad

Gas pricing is the same on Monad as on Ethereum.

There is one difference: the gas charged for a transaction is the **gas limit**, not the gas used. This behavior is on Monad Testnet and is being actively evaluated for mainnet.


## Quick Hits

| Feature | Detail |
| --- | --- |
| **Opcode pricing** | All opcodes cost the same amount of gas as on Ethereum, see [reference](https://www.evm.codes/).<br/> For example, `ADD` costs 3 gas. |
| **EIP-1559** | Monad is EIP-1559-compatible; base fee and priority fee work as on Ethereum. See [this section](#gas-price-detail) for details. |
| **Base fee** | Hard-coded to 50 monad gwei (i.e. `50 * 10^-9 MON`) per unit of gas on testnet.<br/> This will become dynamic in the future. |
| **Transaction ordering** | Default Monad client behavior is that transactions are ordered according to a Priority Gas Auction (descending total gas price).
| **Gas charged** | The gas charged for a transaction is the **gas limit**. That is: total tokens deducted from the sender's balance is `value + gas_price * gas_limit`. This is a DOS-prevention measure for asynchronous execution, see [discussion](#why-charge-gas-limit-rather-than-gas-used) below. |


## Definitions

A common point of confusion among users is the distinction between **gas** of a transaction (units of work) and the **gas price** of a transaction (price in native tokens per unit of work). 

| Feature | Definition |
| --- | --- |
| **Gas** | A unit of work. Gas measures the amount of work the network has to do to process something.<br/><br/> Since the network has multiple kinds of resources (network bandwidth, CPU, SSD bandwidth, and state growth), gas is inherently a projection from many dimensions into a single one. |
| **Gas price (price_per_gas)** | The **price** (in native tokens) paid **to process one unit** of gas. |
| **Gas limit** | The maximum **number of units of gas** that a transaction is allowed to consume.<br/><br/> When a user signs a transaction, the transaction specifies a gas limit and a price limit (tokens-per-gas limit). Since `tokens_spent = tokens_per_gas * gas`, the user is signing off on a maximum amount of native tokens they are willing to spend on the transaction. |

### Gas price detail

[This](https://www.blocknative.com/blog/eip-1559-fees) article provides a good explanation of EIP-1559 gas pricing. A summary follows.

Gas price is expressed using three parameters: `base_price_per_gas`, `priority_price_per_gas`, and `max_price_per_gas`. 
* `base_price_per_gas` is a network-determined parameter. Every transaction in the same block will have the same `base_price_per_gas`
* Users specify `priority_price_per_gas` and `max_price_per_gas` when signing a transaction
* When a transaction is included, the `price_per_gas` paid is the minimum of `max_price_per_gas` and `base_price_per_gas + priority_price_per_gas`
* Since everyone in the same block will pay the same `base_price_per_gas`, the `priority_price_per_gas` is a way for users to pay more to prioritize their transactions.
* Since users don't determine `base_price_per_gas`, the `max_price_per_gas` is a safeguard that limits the amount they may end up paying. Of course, if that value is set too low, the transaction will not end up being chosen for inclusion.

On Monad Testnet, these dynamics still apply, but `base_price_per_gas` is set to a static value of 50 monad gwei.

This means that `price_per_gas` is the minimum of `max_price_per_gas` and `50 * 10^9 wei + priority_price_per_gas`; users can set either parameter to achieve the same effect.

## Why charge gas limit rather than gas used?

Asynchronous execution means that leaders build blocks (and validators vote on block validity) prior to executing the transactions. 

If the protocol charged `gas_used`, a user could submit a transaction with a large `gas_limit` that actually consumes very little gas. This transaction would take up a lot of space toward the block gas limit but wouldn't pay very much for taking up that space, opening up a DOS vector.

The pricing formula is being actively discussed and may change before mainnet.


## Recommendations for developers

### Set the gas limit explicitly if it is constant

Many on-chain actions have a fixed gas cost. The simplest example is that a transfer of native tokens always costs 21,000 gas, but there are many others.

For actions where the gas cost of the transaction is known ahead of time, it is recommended to set it directly prior to handing the transaction off to the wallet. This offers several benefits:
* It reduces latency and gives users a better experience, since the wallet doesn't have to call `eth_estimateGas` and wait for the RPC to respond.
* It retains greater control over the user experience, avoiding cases where the wallet sets a high gas limit in a corner case as described in the warning below.

:::warning
Some wallets, including MetaMask, are known to have the following behavior: when `eth_estimateGas` is called and the contract call reverts, they set the gas limit for this transaction to a very high value.

This is the wallet's way of giving up on setting the gas limit and accepting whatever gas usage is at execution time. However, it doesn't make sense on Monad Testnet where the gas limit is charged.

Contract call reversion happens whenever the user is trying to do something impossible. For example, a user might be trying to mint an NFT that has minted out.

If the gas limit is known ahead of time, setting it explicitly is best practice, since it ensures the wallet won't handle this case unexpectedly.
:::
---

# Network Information

## Monad Testnet

| Name               | Value                                                                  |
| ------------------ | ---------------------------------------------------------------------- |
| Network Name       | Monad Testnet                                                          |
| Chain ID           | 10143                                                                  |
| Currency Symbol    | MON                                                                    |
| RPC Endpoint ([reference](/reference/json-rpc)) | [https://testnet-rpc.monad.xyz](https://testnet-rpc.monad.xyz)         |
| Block Explorer     | [https://testnet.monadexplorer.com](https://testnet.monadexplorer.com) |

### Helpful Links

| Name                  | URL                                                                |
| --------------------- | ------------------------------------------------------------------ |
| Testnet Hub & Faucet  | [https://testnet.monad.xyz](https://testnet.monad.xyz)             |
| Ecosystem Directory   | [https://www.monad.xyz/ecosystem](https://www.monad.xyz/ecosystem) |
| Network Visualization | [https://gmonads.com](https://gmonads.com)                         |

### Canonical Contracts on Testnet

| Name                 | Address                                    |
| -------------------- | ------------------------------------------ 
| CreateX      | [0xba5Ed099633D3B313e4D5F7bdc1305d3c28ba5Ed](https://testnet.monadexplorer.com/address/0xba5Ed099633D3B313e4D5F7bdc1305d3c28ba5Ed) |
| Foundry Deterministic Deployer | [0x4e59b44847b379578588920ca78fbf26c0b4956c](https://testnet.monadexplorer.com/address/0x4e59b44847b379578588920cA78FbF26c0B4956C) |
| EntryPoint v0.6      | [0x5FF137D4b0FDCD49DcA30c7CF57E578a026d2789](https://testnet.monadexplorer.com/address/0x5FF137D4b0FDCD49DcA30c7CF57E578a026d2789) | |
| EntryPoint v0.7      | [0x0000000071727De22E5E9d8BAf0edAc6f37da032](https://testnet.monadexplorer.com/address/0x0000000071727De22E5E9d8BAf0edAc6f37da032) |
| Multicall3   | [0xcA11bde05977b3631167028862bE2a173976CA11](https://testnet.monadexplorer.com/address/0xcA11bde05977b3631167028862bE2a173976CA11) |
| Permit2              | [0x000000000022d473030f116ddee9f6b43ac78ba3](https://testnet.monadexplorer.com/address/0x000000000022d473030f116ddee9f6b43ac78ba3) |
| SafeSingletonFactory | [0x914d7Fec6aaC8cd542e72Bca78B30650d45643d7](https://testnet.monadexplorer.com/address/0x914d7Fec6aaC8cd542e72Bca78B30650d45643d7) |
| UniswapV2Factory  | [0x733e88f248b742db6c14c0b1713af5ad7fdd59d0](https://testnet.monadexplorer.com/address/0x733e88f248b742db6c14c0b1713af5ad7fdd59d0) |
| UniswapV3Factory  | [0x961235a9020b05c44df1026d956d1f4d78014276](https://testnet.monadexplorer.com/address/0x961235a9020b05c44df1026d956d1f4d78014276) |
| UniswapV2Router02 | [0xfb8e1c3b833f9e67a71c859a132cf783b645e436](https://testnet.monadexplorer.com/address/0xfb8e1c3b833f9e67a71c859a132cf783b645e436) |
| Uniswap UniversalRouter   | [0x3ae6d8a282d67893e17aa70ebffb33ee5aa65893](https://testnet.monadexplorer.com/address/0x3ae6d8a282d67893e17aa70ebffb33ee5aa65893) |
| WrappedMonad | [0x760AfE86e5de5fa0Ee542fc7B7B713e1c5425701](https://testnet.monadexplorer.com/address/0x760AfE86e5de5fa0Ee542fc7B7B713e1c5425701) |

See also:
* [Uniswap deployments](https://github.com/Uniswap/contracts/blob/bf676eed3dc31b18c70aba61dcc6b3c6e4d0028f/deployments/10143.md)

### Testnet Tokens (partial list)

| Name                 | Address                                    |
| -------------------- | ------------------------------------------ 
| USDC (testnet) | [0xf817257fed379853cDe0fa4F97AB987181B1E5Ea](https://testnet.monadexplorer.com/address/0xf817257fed379853cDe0fa4F97AB987181B1E5Ea) |
| USDT (testnet) | [0x88b8E2161DEDC77EF4ab7585569D2415a1C1055D](https://testnet.monadexplorer.com/address/0x88b8E2161DEDC77EF4ab7585569D2415a1C1055D) |
| WBTC (testnet) | [0xcf5a6076cfa32686c0Df13aBaDa2b40dec133F1d](https://testnet.monadexplorer.com/address/0xcf5a6076cfa32686c0Df13aBaDa2b40dec133F1d) |
| WETH (testnet) | [0xB5a30b0FDc5EA94A52fDc42e3E9760Cb8449Fb37](https://testnet.monadexplorer.com/address/0xB5a30b0FDc5EA94A52fDc42e3E9760Cb8449Fb37) |
| WSOL (testnet) | [0x5387C85A4965769f6B0Df430638a1388493486F1](https://testnet.monadexplorer.com/address/0x5387C85A4965769f6B0Df430638a1388493486F1) |

### Supported Infrastructure on Testnet

See the [Tooling and Infrastructure](tooling-and-infra/README.md) page for a list of providers supporting testnet.
---

# Guides

Start building smart contracts and applications on Monad with our quickstart guides.

## Get a wallet

<CustomDocCardContainer>
    <CustomDocCard
        icon={<Phantom />}
        link="https://phantom.com/download"
        title="Phantom"
    />
    <CustomDocCard
        icon={<Metamask />}
        link="https://metamask.io/download/"
        title="Metamask"
    />
</CustomDocCardContainer>

## Deploy Smart Contract

<CustomDocCardContainer>
    <CustomDocCard
        icon={<Foundry />}
        link="/guides/deploy-smart-contract/foundry"
        title="Foundry"
        description="Deploy a smart contract on Monad using Foundry"
    />
    <CustomDocCard
        icon={<Hardhat />}
        link="/guides/deploy-smart-contract/hardhat"
        title="Hardhat"
        description="Deploy a smart contract on Monad using Hardhat"
    />
    <CustomDocCard
        icon={<Remix />}
        link="/guides/deploy-smart-contract/remix"
        title="Remix"
        description="Deploy a smart contract on Monad using Remix"
    />
</CustomDocCardContainer>

## Verify Smart Contract

<CustomDocCardContainer>
    <CustomDocCard
        icon={<Foundry />}
        link="/guides/verify-smart-contract/foundry"
        title="Foundry"
        description="Verify a smart contract on Monad using Foundry"
    />
    <CustomDocCard
        icon={<Hardhat />}
        link="/guides/verify-smart-contract/hardhat"
        title="Hardhat"
        description="Verify a smart contract on Monad using Hardhat"
    />
</CustomDocCardContainer>

## Indexing

<CustomDocCardContainer>
    <CustomDocCard
        icon={<Monad />}
        link="/guides/indexers/ghost"
        title="GhostGraph"
        description="Index transfers with GhostGraph"
    />
    <CustomDocCard
        icon={<Monad />}
        link="/guides/indexers/tg-bot-using-envio"
        title="Envio"
        description="Index transfers for a telegram bot using Envio"
    />
    <CustomDocCard
        icon={<Monad />}
        link="/guides/indexers/quicknode-streams"
        title="QuickNode Streams"
        description="Index transfers using QuickNode Streams"
    />
</CustomDocCardContainer>


{/* TODO: Add Developer Tools & Infra */}
{/* ## Developer Tools & Infra */}

## Connectivity

<CustomDocCardContainer>
    <CustomDocCard
        icon={<Monad />}
        link="/guides/reown-guide"
        title="Reown AppKit"
        description="Connect a wallet to your app with Reown AppKit"
    />
    <CustomDocCard
        icon={<Monad />}
        link="/guides/blinks-guide"
        title="Blinks"
        description="Build a donation blink"
    />
</CustomDocCardContainer>

## AI

<CustomDocCardContainer>
    <CustomDocCard
        icon={<Monad />}
        link="/guides/monad-mcp"
        title="MCP Server"
        description="Build an MCP server to interact with Monad Testnet"
    />
</CustomDocCardContainer>
---

# Add Monad to Your Wallet
---

# Add Monad Testnet to MetaMask

Follow this quick guide to add Monad Testnet to your MetaMask wallet.

1. Open MetaMask

![open metamask](/img/guides/add-monad-to-wallet/metamask/1.png)

2. Click on the "Add a Custom Network" button.

![add custom network](/img/guides/add-monad-to-wallet/metamask/2.png)

3. Add the Network name, add a RPC URL by clicking the "Add RPC URL" button.

![add network name](/img/guides/add-monad-to-wallet/metamask/3.png)

4. Enter the RPC details.

```
https://testnet-rpc.monad.xyz
```

![add rpc details](/img/guides/add-monad-to-wallet/metamask/4.png)

5. Enter the Chain ID, Currency Symbol, add the Block Explorer URL by clicking the "Add a block explorer URL" button.

![add chain id explorer](/img/guides/add-monad-to-wallet/metamask/5.png)

6. Enter the Block Explorer URL

```
https://testnet.monadexplorer.com
```

![add block explorer](/img/guides/add-monad-to-wallet/metamask/6.png)

7. Finally you can click on the "Save" button.

![save](/img/guides/add-monad-to-wallet/metamask/7.png)

8. You should see a success message.

![monad testnet sucess message](/img/guides/add-monad-to-wallet/metamask/8.png)

9. You can now see and use the Monad Testnet network in your MetaMask wallet.

![monad testnet](/img/guides/add-monad-to-wallet/metamask/9.png)

10. You should be able to see your Monad Testnet assets.

![monad testnet assets](/img/guides/add-monad-to-wallet/metamask/10.png)


The next step is to get MON tokens from the Monad Testnet Faucet.
---

# How to build a donation blink

URL: https://docs.monad.xyz/guides/blinks-guide.md

In this guide, you will learn how to build a [Blink](https://www.dialect.to/) that allows people to donate MON with a single click.

## Prerequisites

- Code Editor of your choice ([Cursor](https://www.cursor.com/) or [Visual Studio Code](https://code.visualstudio.com/) recommended).
- [Node](https://nodejs.org/en/download) 18.x.x or above.
- Basic TypeScript knowledge.
- Testnet MON ([Faucet](https://testnet.monad.xyz)).

## Initial setup

### Initialize the project

```bash
npx create-next-app@14 blink-starter-monad && cd blink-starter-monad
```

**When prompted, configure your project with these settings:**

- ✓ Ok to proceed? → Yes
- ✓ Would you like to use TypeScript? → Yes
- ✓ Would you like to use ESLint? → Yes
- ✓ Would you like to use Tailwind CSS? → Yes
- ✓ Would you like your code inside a `src/` directory? → Yes
- ✓ Would you like to use App Router? → Yes
- ✓ Would you like to customize the import alias (`@/*` by default)? → No

### Install dependencies

```bash
npm install @solana/actions wagmi viem@2.x
```

### Start development server

The development server is used to start a local test environment that runs on your computer. It is perfect to test and develop your blink, before you ship it to production.

```bash
npm run dev
```

## Building the Blink

Now that we have our basic setup finished, it is time to start building the blink.

### Create an endpoint

To write a blink provider, you have to create an endpoint. Thanks to NextJS, this all works pretty straightforward. All you have to do is to create the following folder structure:

```
src/
└── app/
    └── api/
            └── actions/
                └── donate-mon/
                    └── route.ts
```

### Create actions.json

Create a route in `app` folder for the `actions.json` file which will be hosted in the root directory of our application. This file is needed to tell other applications which blink providers are available on your website. **Think of it as a sitemap for blinks.** 

You can read more about the [actions.json](https://docs.dialect.to/documentation/actions/specification/actions.json) in the official [Dialect documentation](https://docs.dialect.to/documentation/actions/specification/actions.json).


```
src/
└── app/
    └── actions.json/
        └── route.ts
```

```js
// src/app/actions.json/route.ts

export const GET = async () => {
  const payload: ActionsJson = {
    rules: [
      // map all root level routes to an action
      {
        pathPattern: "/*",
        apiPath: "/api/actions/*",
      },
      // idempotent rule as the fallback
      {
        pathPattern: "/api/actions/**",
        apiPath: "/api/actions/**",
      },
    ],
  };

  return Response.json(payload, {
    headers: ACTIONS_CORS_HEADERS,
  });
};

// DO NOT FORGET TO INCLUDE THE `OPTIONS` HTTP METHOD
// THIS WILL ENSURE CORS WORKS FOR BLINKS
export const OPTIONS = GET;
```

### Add an image for the blink

Every blink has an image that is rendered on top. If you have your image already hosted somewhere, you can skip this step but if you haven't you can just create a `public` folder in your `NextJS` project and paste an image there.

In our example we will paste a file called `donate-mon.png` into this public folder. You can right-click and save the image below.

![donate-mon](/img/guides/blinks-guide/donate-mon.png)

![image](/img/guides/blinks-guide/1.png)

### OPTIONS endpoint and headers

This enables CORS for cross-origin requests and standard headers for the API endpoints. This is standard configuration you do for every Blink.

```js
// src/app/api/actions/donate-mon/route.ts

// CAIP-2 format for Monad
const blockchain = `eip155:10143`;

// Create headers with CAIP blockchain ID
const headers = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers":
  "Content-Type, x-blockchain-ids, x-action-version",
  "Content-Type": "application/json",
  "x-blockchain-ids": blockchain,
  "x-action-version": "2.0",
};

// OPTIONS endpoint is required for CORS preflight requests
// Your Blink won't render if you don't add this
export const OPTIONS = async () => {
  return new Response(null, { headers });
};
```

### GET endpoint

`GET` returns the Blink metadata and UI configuration. 

It describes:

- How the Action appears in Blink clients
- What parameters users need to provide
- How the Action should be executed

```js
// src/app/api/actions/donate-mon/route.ts

// GET endpoint returns the Blink metadata (JSON) and UI configuration
export const GET = async (req: Request) => {
  // This JSON is used to render the Blink UI
  const response: ActionGetResponse = {
    type: "action",
    icon: `${new URL("/donate-mon.png", req.url).toString()}`,
    label: "1 MON",
    title: "Donate MON",
    description:
      "This Blink demonstrates how to donate MON on the Monad blockchain. It is a part of the official Blink Starter Guides by Dialect Labs.  \n\nLearn how to build this Blink: https://dialect.to/docs/guides/donate-mon",
    // Links is used if you have multiple actions or if you need more than one params
    links: {
      actions: [
        {
          // Defines this as a blockchain transaction
          type: "transaction",
          label: "0.01 MON",
          // This is the endpoint for the POST request
          href: `/api/actions/donate-mon?amount=0.01`,
        },
        {
          type: "transaction",
          label: "0.05 MON",
          href: `/api/actions/donate-mon?amount=0.05`,
        },
        {
          type: "transaction",
          label: "0.1 MON",
          href: `/api/actions/donate-mon?amount=0.1`,
        },
        {
          // Example for a custom input field
          type: "transaction",
          href: `/api/actions/donate-mon?amount={amount}`,
          label: "Donate",
          parameters: [
            {
              name: "amount",
              label: "Enter a custom MON amount",
              type: "number",
            },
          ],
        },
      ],
    },
  };

  // Return the response with proper headers
  return new Response(JSON.stringify(response), {
    status: 200,
    headers,
  });
};
```

### Testing the Blink

Visit [dial.to](https://dial.to) and type in the link to your blink to see if it works. If your server runs on localhost:3000 the url should be like this: `http://localhost:3000/api/actions/donate-mon`

:::info
[dial.to](https://dial.to) currently supports only GET previews for EVM. To test your POST endpoint, we need to build a Blink Client.
:::

![testing blink](/img/guides/blinks-guide/2.png)


### POST endpoint

`POST` handles the actual MON transfer transaction.

#### POST request to the endpoint

Create the post request structure and add the necessary imports as well as the `donationWallet` on top of the file.

```js
//src/app/api/actions/donate-mon/route.ts

// Update the imports
// Wallet address that will receive the donations
const donationWallet = `<RECEIVER_ADDRESS>`;


// POST endpoint handles the actual transaction creation
export const POST = async (req: Request) => {
  try {
  
  // Code that goes here is in the next step
  
  } catch (error) {
    // Log and return an error response
    console.error("Error processing request:", error);
    return new Response(JSON.stringify({ error: "Internal server error" }), {
      status: 500,
      headers,
    });
  }
};
```

#### Extract data from request

The request contains the URL and the account (PublicKey) from the payer.

```js
// src/app/api/actions/donate-mon/route.ts

// POST endpoint handles the actual transaction creation
export const POST = async (req: Request) => {
  try {
    // Step 1
    // Extract amount from URL
    const url = new URL(req.url);
    const amount = url.searchParams.get("amount");

    if (!amount) {
        throw new Error("Amount is required");
    }

  } catch (error) {
    // Error handling
  }
}
```

#### Create the transaction

Create a new transaction with all the necessary data and add it below in the `POST` request.

```js
// src/app/api/actions/donate-mon/route.ts

// POST endpoint handles the actual transaction creation
export const POST = async (req: Request) => {
  try {

    // ... previous code from step
    
    // Build the transaction
    const transaction = {
        to: donationWallet,
        value: parseEther(amount).toString(),
        chainId: 10143,
    };

    const transactionJson = serialize(transaction);
  
  } catch (error) {
    // Error handling
  }
}
```

#### Return the transaction in response.

Create `ActionPostResponse` and return it to the client.

```ts
// src/app/api/actions/donate-mon/route.ts

export const POST = async (req: Request) => {
  try {
    // ... previous code from step 1 and 2
    
    // Build ActionPostResponse
    const response: ActionPostResponse = {
        type: "transaction",
        transaction: transactionJson,
        message: "Donate MON",
    };

    // Return the response with proper headers
    return new Response(JSON.stringify(response), {
        status: 200,
        headers,
    });

  } catch (error) {
    // Error handling
  }
}
```

### Full code in `route.ts`

```ts
// CAIP-2 format for Monad
const blockchain = `eip155:10143`;

// Wallet address that will receive the donations
const donationWallet = `<RECEIVER_ADDRESS>`;

// Create headers with CAIP blockchain ID
const headers = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers":
  "Content-Type, x-blockchain-ids, x-action-version",
  "Content-Type": "application/json",
  "x-blockchain-ids": blockchain,
  "x-action-version": "2.0",
};

// OPTIONS endpoint is required for CORS preflight requests
// Your Blink won't render if you don't add this
export const OPTIONS = async () => {
  return new Response(null, { headers });
};

// GET endpoint returns the Blink metadata (JSON) and UI configuration
export const GET = async (req: Request) => {
  // This JSON is used to render the Blink UI
  const response: ActionGetResponse = {
    type: "action",
    icon: `${new URL("/donate-mon.png", req.url).toString()}`,
    label: "1 MON",
    title: "Donate MON",
    description:
      "This Blink demonstrates how to donate MON on the Monad blockchain. It is a part of the official Blink Starter Guides by Dialect Labs.  \n\nLearn how to build this Blink: https://dialect.to/docs/guides/donate-mon",
    // Links is used if you have multiple actions or if you need more than one params
    links: {
      actions: [
        {
          // Defines this as a blockchain transaction
          type: "transaction",
          label: "0.01 MON",
          // This is the endpoint for the POST request
          href: `/api/actions/donate-mon?amount=0.01`,
        },
        {
          type: "transaction",
          label: "0.05 MON",
          href: `/api/actions/donate-mon?amount=0.05`,
        },
        {
          type: "transaction",
          label: "0.1 MON",
          href: `/api/actions/donate-mon?amount=0.1`,
        },
        {
          // Example for a custom input field
          type: "transaction",
          href: `/api/actions/donate-mon?amount={amount}`,
          label: "Donate",
          parameters: [
            {
              name: "amount",
              label: "Enter a custom MON amount",
              type: "number",
            },
          ],
        },
      ],
    },
  };

  // Return the response with proper headers
  return new Response(JSON.stringify(response), {
    status: 200,
    headers,
  });
};

// POST endpoint handles the actual transaction creation
export const POST = async (req: Request) => {
    try {
      // Extract amount from URL
      const url = new URL(req.url);
      const amount = url.searchParams.get("amount");

      if (!amount) {
          throw new Error("Amount is required");
      }

      // Build the transaction
      const transaction = {
          to: donationWallet,
          value: parseEther(amount).toString(),
          chainId: 10143,
      };

      const transactionJson = serialize(transaction);

      // Build ActionPostResponse
      const response: ActionPostResponse = {
          type: "transaction",
          transaction: transactionJson,
          message: "Donate MON",
      };

      // Return the response with proper headers
      return new Response(JSON.stringify(response), {
          status: 200,
          headers,
      });
    } catch (error) {
      // Log and return an error response
      console.error("Error processing request:", error);
      return new Response(JSON.stringify({ error: "Internal server error" }), {
        status: 500,
        headers,
      });
  }
};
```

At this point the Blink is ready, but we need a Blink client since [dial.to](https://dial.to) does not support EVM wallets.

## Implementing the Blink client

In this step you will learn to implement the blink client, which is the visual representation of a blink.

### Install dependencies

```bash
npm install connectkit @tanstack/react-query @dialectlabs/blinks
```

### Implement the provider

The provider is necessary to trigger wallet actions in the blink.

### Create config for `WagmiProvider`

This file is used to set the proper configurations for the `WagmiProvider` in the next step.

```ts
// src/config.ts

export const config = createConfig({
  chains: [monadTestnet],
  transports: {
    [monadTestnet.id]: http(),
  },
});
```

### Create the wallet connection context providers

Create the provider that we can use to wrap around our app. Don't forget to use the `“use client”;` at the top of the file if you are in a NextJS project.

:::info
In this project, we are using [ConnectKit](https://docs.family.co/connectkit) but you can use other alternatives as well (Eg: [RainbowKit](https://www.rainbowkit.com/))
:::

```tsx
// src/provider.tsx

"use client";

const queryClient = new QueryClient();

export const Providers = ({ children }: PropsWithChildren) => {
  return (
    <WagmiProvider config={config}>
      <QueryClientProvider client={queryClient}>
        <ConnectKitProvider>{children}</ConnectKitProvider>
      </QueryClientProvider>
    </WagmiProvider>
  );
};
```

### Wrap the app with context provider

If you want your provider to be accessible throughout your app, it is recommended to wrap it around the `children` element in your `layout.tsx`.

```tsx
// src/app/layout.tsx

// additional import
// other code in the file ...

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
      className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
          <Providers>{children}</Providers>
      </body>
    </html>
  );
}
```

### Using the `Blink` component

Now that we have everything wrapped, we can start with the implementation of the blink renderer.
To do so open the `page.tsx` file in your `/src/app` folder.

```tsx
// src/app/page.tsx

"use client";

export default function Home() {
  // Actions registry interval
  useActionsRegistryInterval();

  // ConnectKit modal
  const { setOpen } = useModal();

  // Wagmi adapter, used to connect to the wallet
  const { adapter } = useEvmWagmiAdapter({
    onConnectWalletRequest: async () => {
      setOpen(true);
    },
  });

  // Action we want to execute in the Blink
  const { blink, isLoading } = useBlink({
    url: "evm-action:http://localhost:3000/api/actions/donate-mon",
  });

  return (
    <main className="flex flex-col items-center justify-center">
      <ConnectKitButton />
      <div className="w-1/2 lg:px-4 lg:p-8">
        {isLoading || !blink ? (
          <span>Loading</span>
        ) : (
          // Blink component, used to execute the action
          <Blink blink={blink} adapter={adapter} securityLevel="all" />
        )}
      </div>
    </main>
  );
}
```

### Make a transaction

That's it. To test it, visit [localhost:3000](http://localhost:3000) and click on a button or enter a custom amount that you want to donate.

![blink client](/img/guides/blinks-guide/3.png)

## Conclusion

In this tutorial, you learned how you can create a blink that sends MON to another wallet from scratch using a `NextJS` project. Besides the basic project setup there were two important things that we built.

The first thing was the blink provider. This provider works as an API for the blink and handles how the blink is rendered in the fronend (`GET` request) and executes the blockchain transaction (`POST` request).

The second implementation was the blink client. This client serves as the visual representation of the blink and is what the user sees and uses to interact with the blink provider.

These are two separate parts, which means you can build a blink without worrying about the client implementation and you can implement clients for existing blinks without the need to build your own blink.
---

# Deploy a Contract

<CustomDocCardContainer>
    <CustomDocCard
        icon={<Foundry />}
        link="/guides/deploy-smart-contract/foundry"
        title="Foundry"
        description="Deploy a smart contract on Monad using Foundry"
    />
    <CustomDocCard
        icon={<Hardhat />}
        link="/guides/deploy-smart-contract/hardhat"
        title="Hardhat"
        description="Deploy a smart contract on Monad using Hardhat"
    />
    <CustomDocCard
        icon={<Remix />}
        link="/guides/deploy-smart-contract/remix"
        title="Remix"
        description="Deploy a smart contract on Monad using Remix"
    />
</CustomDocCardContainer>


---

# Deploy a smart contract on Monad using Foundry

URL: https://docs.monad.xyz/guides/deploy-smart-contract/foundry

[Foundry](https://book.getfoundry.sh/) is a blazing fast, portable and modular toolkit for Ethereum application development written in Rust.

## Requirements

Before you begin, you need to install the following tools:

-   [Rust](https://www.rust-lang.org/)

## 1. Installing `foundryup`

Foundryup is the official installer for the Foundry toolchain.

```sh
curl -L https://foundry.paradigm.xyz | bash
```

This will install Foundryup. Simply follow the on-screen instructions, and the `foundryup` command will become available in your CLI.

## 2. Installing `forge`, `cast`, `anvil` and `chisel` binaries

```sh
foundryup
```

:::note
If you're on Windows, you'll need to use WSL, since Foundry currently doesn't work natively on Windows. Please follow [this link](https://learn.microsoft.com/en-us/windows/wsl/install) to learn more about WSL.
:::

## 3. Create a new foundry project

:::tip
You can use `foundry-monad` template to create a new project.

_[Foundry-Monad](https://github.com/monad-developers/foundry-monad) is a Foundry template with Monad configuration._
:::

The below command uses `foundry-monad` to create a new foundry project:

```sh
forge init --template monad-developers/foundry-monad [project_name]
```

Alternatively, you can create a foundry project using the command below:

```sh
forge init [project_name]
```

## 4. Modify Foundry configuration

Update the `foundry.toml` file to add Monad Testnet configuration.

```toml
[profile.default]
src = "src"
out = "out"
libs = ["lib"]

# Monad Testnet Configuration
eth-rpc-url="https://testnet-rpc.monad.xyz"
chain_id = 10143
```

## 5. Write a smart contract

You can write your smart contracts under the `src` folder. There is already a `Counter` contract in the project located at `src/Counter.sol`.

```solidity
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

contract Counter {
    uint256 public number;

    function setNumber(uint256 newNumber) public {
        number = newNumber;
    }

    function increment() public {
        number++;
    }
}
```

## 6. Compile the smart contract

```sh
forge compile
```

Compilation process output can be found in the newly created `out` directory, which includes contract ABI and bytecode.

## 7. Deploy the smart contract

:::note
For deploying contracts, we recommend using keystores instead of private keys.
:::

### Get testnet funds

Deploying smart contracts requires testnet funds. Claim testnet funds via a [faucet](https://testnet.monad.xyz).

### Deploy smart contract

<Tabs>
    <TabItem value="with-keystore" label="Using a Keystore (Recommended)" default>
        Using a keystore is much safer than using a private key because keystore encrypts the private key and can later be referenced in any commands that require a private key.

        Create a new keystore by importing a newly generated private key with the command below.

        ```sh
        cast wallet import monad-deployer --private-key $(cast wallet new | grep 'Private key:' | awk '{print $3}')
        ```

        Here is what the command above does, step by step:

        -   Generates a new private key
        -   Imports the private key into a keystore file named `monad-deployer`
        -   Prints the address of the newly created wallet to the console

        After creating the keystore, you can read its address using:

        ```sh
        cast wallet address --account monad-deployer
        ```

        Provide a password to encrypt the keystore file when prompted and do not forget it.

        Run the below command to deploy your smart contracts

        ```sh
        forge create src/Counter.sol:Counter --account monad-deployer --broadcast
        ```

    </TabItem>

    <TabItem value="with-private-key" label="Using a Private Key (Not Recommended)">
        Use the below command to deploy a smart contract by directly pasting the private key in the terminal.

        :::warning
        Using a private key is not recommended. You should not be copying and pasting private keys into your terminal. Please use a keystore instead.
        :::

        ```sh
        forge create --private-key <your_private_key> src/Counter.sol:Counter --broadcast
        ```
    </TabItem>

</Tabs>

On successful deployment of the smart contract, the output should be similar to the following:

```sh
[⠊] Compiling...
Deployer: 0xB1aB62fdFC104512F594fCa0EF6ddd93FcEAF67b
Deployed to: 0x67329e4dc233512f06c16cF362EC3D44Cdc800e0
Transaction hash: 0xa0a40c299170c9077d321a93ec20c71e91b8aff54dd9fa33f08d6b61f8953ee0
```

### Next Steps

Check out [how to verify the deployed smart contract on Monad Explorer](/guides/verify-smart-contract/foundry.mdx).

---

# Deploy a smart contract on Monad using Hardhat

URL: https://docs.monad.xyz/guides/deploy-smart-contract/hardhat

[Hardhat](https://hardhat.org/docs) is a comprehensive development environment consisting of different components for editing, compiling, debugging, and deploying your smart contracts.

## Requirements

Before you begin, you need to install the following dependencies:

- Node.js v18.0.0 or later

:::note
If you are on Windows, we strongly recommend using [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/about) when following this guide.
:::

## 1. Create a new Hardhat project

:::tip
You can use the `hardhat-monad` template to create a new project with Monad configuration already set up.

_[hardhat-monad](https://github.com/monad-developers/hardhat-monad) is a Hardhat template with Monad configuration._
:::

Clone the repository to your machine using the command below:

```sh
git clone https://github.com/monad-developers/hardhat-monad.git
```

## 2. Install dependencies

```sh
npm install
```

## 3. Create an .env file

```sh
cp .env.example .env
```

Edit the `.env` file with your private key.

:::warning
Protect your private key carefully. Never commit it to version control, share it in public repositories, or expose it in client-side code. Your private key provides full access to your funds.
:::

## 4. Deploy the smart contract

### Deploying to the local hardhat node

Run hardhat node by running:

```bash
npx hardhat node
```

To deploy the example contract to the local hardhat node, run the following command in a separate terminal:

```bash
npx hardhat ignition deploy ignition/modules/Lock.ts
```

### Deploying to Monad Testnet

The following command will deploy the example contract to the Monad Testnet using [Hardhat Ignition](https://hardhat.org/ignition/docs/getting-started#overview):

```bash
npx hardhat ignition deploy ignition/modules/Lock.ts --network monadTestnet
```

To redeploy the same code to a different address, use the command below:

```bash
npx hardhat ignition deploy ignition/modules/Lock.ts --network monadTestnet --reset
```

You can customize deployment parameters:

```bash
npx hardhat ignition deploy ignition/modules/Lock.ts --network monadTestnet --parameters '{"unlockTime": 1893456000, "lockedAmount": "1000000000000000"}'
```

## Next Steps

Check out [how to verify the deployed smart contract on Monad Explorer](/guides/verify-smart-contract/hardhat).

---

# Deploy a smart contract on Monad using Remix

URL: https://docs.monad.xyz/guides/deploy-smart-contract/remix

[Remix IDE](https://remix.ethereum.org/) is a browser-based IDE that can be used for the entire journey of smart contract development by users at every knowledge level. It requires no setup, fosters a fast development cycle, and has a rich set of plugins with intuitive GUIs.

In this guide you will learn how to deploy and interact with a simple Greeting smart contract on Monad Testnet using [Remix IDE](https://remix.ethereum.org/).

## Requirements

- You need to have the Monad Testnet network added to your wallet.


## Deploying the smart contract

Head over to [Remix IDE](https://remix.ethereum.org/) in your browser. Click 'Start Coding' to create a new project template.

![remix-ide](/img/guides/deploy-smart-contract/remix/1.png)

Make sure the 'contracts' folder is selected, then create a new file using the "Create new file" button on top left corner.

![create-file](/img/guides/deploy-smart-contract/remix/2.png)

Name the new file "Gmonad.sol" and add the following code to it

```sol
// SPDX-License-Identifier: MIT

// Make sure the compiler version is below 0.8.24 since Cancun compiler is not supported just yet
pragma solidity >=0.8.0 <=0.8.24;

contract Gmonad { 
    string public greeting;

    constructor(string memory _greeting) {
        greeting = _greeting;
    }

    function setGreeting(string calldata _greeting) external {
        greeting = _greeting;
    }
}
```

**Note:** You may see a red squiggly line underneath the `pragma solidity...` line; this is because the default compiler version is outside of the range specified in the contract. We'll fix that in the next step.

![code](/img/guides/deploy-smart-contract/remix/3.png)

Let's compile the smart contract. Navigate to the compiler view by clicking the "Solidity compiler" tab on the far left. Then select the right compiler version (0.8.24).

![compiler](/img/guides/deploy-smart-contract/remix/4.png)

Once you have the right compiler version selected, click on the "Compile Gmonad.sol" button. If succesful, you will see a green check mark on the "Solidity compiler" tab icon.

![compile](/img/guides/deploy-smart-contract/remix/5.png)

Now we can deploy the smart contract! Navigate to the deploy view using the "Deploy & run transactions" tab on the far left.

![deploy](/img/guides/deploy-smart-contract/remix/6.png)

Using the "Environment" dropdown, select "Injected Provider" to connect to your wallet.

The screenshot below says "Injected Provider - Metamask"; in case you are using some wallet other than Metamask you may see an appropriate option.

![environment](/img/guides/deploy-smart-contract/remix/7.png)

Your wallet should pop up asking for permission to connect to Remix, click "Connect".

![connect](/img/guides/deploy-smart-contract/remix/8.png)

Once connected you should be able to see your address with your balance in the "Account" dropdown.

Make sure you also see the correct chain id under the "Environment" dropdown.

Now let's deploy the contract. `Gmonad.sol` requires a greeting message to be passed to the constructor before it can be deployed; choose the greeting message of your choice (in this example it is "gmonad").

Now you can deploy the smart contract by clicking the "Deploy" button.

![deploy](/img/guides/deploy-smart-contract/remix/9.png)

You should see a wallet popup asking for confirmation to deploy the smart contract. Click "Confirm".

![confirm](/img/guides/deploy-smart-contract/remix/10.png)

Once the transaction is confirmed you will see the smart contract address in the "Deployed Contracts" section on the bottom left.

![deployed](/img/guides/deploy-smart-contract/remix/11.png)

## Interacting with the smart contract

You can expand the smart contract to see the functions available.

There you will find a `greeting` button which can be used to read the current greeting message stored in the smart contract.

Click the "greeting" button to call the `greeting()` method (which outputs the current greeting message). You'll need to click the expand arrow in the terminal output to see the decoded output.

:::info
This "greeting" button is a getter function which is automatically created for the _public_ `greeting` state variable in the smart contract.
:::

![expand](/img/guides/deploy-smart-contract/remix/12.png)

You can change the greeting message by using the `setGreeting` function.

In this example, we will change the greeting message to "gmonad molandak".

Once again, click the "transact" button to initiate the transaction.

You should see a wallet popup asking for confirmation to change the greeting message. Click "Confirm".

![transact](/img/guides/deploy-smart-contract/remix/13.png)

Once the transaction is confirmed you can view the updated greeting message using the `greeting` button.

![updated](/img/guides/deploy-smart-contract/remix/14.png)

Congratulations! You have successfully deployed and interacted with a smart contract on Monad  Testnet using Remix IDE.


{/* TODO: Add next step to verify the contract using the explorer */}
{/* ## Next steps

Check out how to verify your smart contract on Monad using Explorer */}

---

# EVM Resources
---

# EVM Behavior

## EVM Behavioral Specification

* [Notes on the EVM](https://github.com/CoinCulture/evm-tools/blob/master/analysis/guide.md): straightforward technical specification of the EVM plus some behavioral examples
* [EVM: From Solidity to bytecode, memory and storage](https://www.youtube.com/watch?v=RxL\_1AfV7N4): a 90-minute talk from Peter Robinson and David Hyland-Wood
* [EVM illustrated](https://takenobu-hs.github.io/downloads/ethereum\_evm\_illustrated.pdf): an excellent set of diagrams for confirming your mental model
* [EVM Deep Dives: The Path to Shadowy Super-Coder](https://noxx.substack.com/p/evm-deep-dives-the-path-to-shadowy)

## Opcode Reference

[evm.codes](https://www.evm.codes/): opcode reference (including gas costs) and an interactive sandbox for stepping through bytecode execution

## Solidity Storage Layout

The EVM allows smart contracts to store data in 32-byte words ("storage slots"), however the details of how complex datastructures such as lists or mappings is left as an implementation detail to the higher-level language.  Solidity has a specific way of assigning variables to storage slots, described below:

* [Official docs on storage layout](https://docs.soliditylang.org/en/latest/internals/layout\_in\_storage.html)
* [Storage patterns in Solidity](https://programtheblockchain.com/posts/2018/03/09/understanding-ethereum-smart-contract-storage/)
---

# Other Languages
---

# Huff

[Huff](https://docs.huff.sh/) is most closely described as EVM assembly. Unlike Yul, Huff does not provide control flow constructs or abstract away the inner working of the program stack. Only the most upmost performance sensitive applications take advantage of Huff, however it is a great educational tool to learn how the EVM interprets instructions its lowest level.

* [Huff resources](https://docs.huff.sh/resources/overview/) provides additional resources
---

# Vyper

[Vyper](https://www.quicknode.com/guides/ethereum-development/smart-contracts/how-to-write-an-ethereum-smart-contract-using-vyper) is a popular programming language for the EVM that is logically similar to Solidity and syntactically similar with Python.

The [Vyper documentation](https://docs.vyperlang.org/en/stable/index.html) covers installing the Vyper language, language syntax, coding examples, compilation.

A typical EVM developer looking for a Python-like experience is encouraged to use Vyper as the programming language and [ApeWorx](https://docs.apeworx.io/ape/stable/userguides/quickstart.html), which leverages the Python language, as the testing and deployment framework. ApeWorx also allows for the use of typical Python libraries in analysis of testing results such as Pandas.

Vyper and ApeWorx can be used with [Jupyter](https://jupyter.org/), which offers an interactive environment using a web browser.  A quick setup guide for working with Vyper and Jupyter for smart contract development for the EVM can be found [here](https://medium.com/deepyr/interacting-with-ethereum-using-web3-py-and-jupyter-notebooks-e4207afa0085).

## Resources

* [Vyper by Example](https://vyper-by-example.org/)
* [Snekmate](https://github.com/pcaversaccio/snekmate): a Vyper library of gas-optimized smart contract building blocks
* [Curve contracts](https://github.com/curvefi/curve-contract): the most prominent example usage of Vyper

---

# Yul

[Yul](https://docs.soliditylang.org/en/latest/yul.html) is a intermediate language for Solidity that can generally be thought of as inline assembly for the EVM. It is not quite pure assembly, providing control flow constructs and abstracting away the inner working of the stack while still exposing the raw memory backend to developers. Yul is targeted at developers needing exposure to the EVM's raw memory backend to build high performance gas optimized EVM code.
---

# Solidity Resources

Monad is fully EVM bytecode-compatible, with all supported opcodes and precompiles as of the [Cancun fork](https://www.evm.codes/?fork=cancun). Monad also preserves the standard Ethereum JSON-RPC interfaces.

As such, most development resources for Ethereum Mainnet apply to development on Monad.

This page suggests a **minimal** set of resources for getting started with building a decentralized app for Ethereum. Child pages provide additional detail or options.&#x20;

As [Solidity](https://docs.soliditylang.org/) is the most popular language for Ethereum smart contracts, the resources on this page focus on Solidity; alternatively see resources on [Vyper](/docs/guides/evm-resources/other-languages/vyper.md) or [Huff](/docs/guides/evm-resources/other-languages/huff.md). Note that since smart contracts are composable, contracts originally written in one language can still make calls to contracts in another language.

## **IDEs**

-   [Remix](https://remix.ethereum.org/#lang=en&optimize=false&runs=200&evmVersion=null) is an interactive Solidity IDE. It is the easiest and fastest way to get started coding and compiling Solidity smart contracts without the need for additional tool installations.
-   [VSCode](https://code.visualstudio.com/) + [Solidity extension](https://marketplace.visualstudio.com/items?itemName=NomicFoundation.hardhat-solidity)

## **Basic Solidity**

-   [CryptoZombies](https://cryptozombies.io/en/course) is a great end-to-end introduction to building dApps on the EVM. It provides resources and lessons for anyone from someone who has never coded before, to experienced developers in other disciplines looking to explore blockchain development.
-   [Solidity by Example](https://solidity-by-example.org/) introduces concepts progressively through simple examples; best for developers who already have basic experience with other languages.
-   [Blockchain Basics course by Cyfrin Updraft](https://updraft.cyfrin.io/courses/blockchain-basics) teaches the fundamentals of blockchain, DeFi, and smart contracts.
-   [Solidity Smart Contract Development by Cyfrin Updraft](https://updraft.cyfrin.io/courses/solidity) will teach you how to become a smart contract developer. Learn to build with projects and get hands-on experience.
-   [Ethereum Developer Degree by LearnWeb3](https://learnweb3.io/degrees/ethereum-developer-degree/) is the a good course to go from no background knowledge in web3 to being able to build multiple applications and understanding several key protocols, frameworks, and concepts in the web3 space.

## **Intermediate Solidity**

-   [The Solidity Language](https://docs.soliditylang.org/en/v0.8.21/introduction-to-smart-contracts.html) official documentation is an end-to-end description of Smart Contracts and blockchain basics centered on EVM environments. In addition to Solidity Language documentation, it covers the basics of compiling your code for deployment on an EVM as well as the basic components relevant to deploying a Smart Contract on an EVM.
-   [Solidity Patterns](https://github.com/fravoll/solidity-patterns) repository provides a library of code templates and explanation of their usage.&#x20;
-   The [Uniswap V2](https://github.com/Uniswap/v2-core) contract is a professional yet easy to digest smart contract that provides a great overview of an in-production Solidity dApp. A guided walkthrough of the contract can be found [here](https://ethereum.org/en/developers/tutorials/uniswap-v2-annotated-code/).
-   [Cookbook.dev](https://www.cookbook.dev/search?q=cookbook&categories=Contracts&sort=popular&filter=&page=1) provides a set of interactive example template contracts with live editing, one-click deploy, and an AI chat integration to help with code questions.&#x20;
-   [OpenZeppelin](https://www.openzeppelin.com/contracts) provides customizable template contract library for common Ethereum token deployments such as ERC20, ERC712, and ERC1155. Note, they are not gas optimized.
-   [Rareskills Blog](https://www.rareskills.io/category/solidity) has some great in-depth articles on various concepts in Solidity.
-   [Foundry Fundamentals course by Cyfrin Updraft](https://updraft.cyfrin.io/courses/foundry) is a comprehensive web3 development course designed to teach you about Foundry the industry-standard framework to build, deploy, and test your smart contracts.
-   [Smart Contract Programmer YT channel](https://www.youtube.com/@smartcontractprogrammer) has a plenty of in-depth videos about various Solidity concepts like ABI encoding, EVM memory, and many more.

## **Advanced Solidity**

-   The [Solmate repository](https://github.com/transmissions11/solmate) and [Solady repository](https://github.com/Vectorized/solady/tree/main) provide gas-optimized contracts utilizing Solidity or Yul.
-   [Yul](https://docs.soliditylang.org/en/latest/yul.html) is a intermediate language for Solidity that can generally be thought of as inline assembly for the EVM. It is not quite pure assembly, providing control flow constructs and abstracting away the inner working of the stack while still exposing the raw memory backend to developers. Yul is targeted at developers needing exposure to the EVM's raw memory backend to build high performance gas optimized EVM code.&#x20;
-   [Huff](https://docs.huff.sh/get-started/overview/) is most closely described as EVM assembly. Unlike Yul, Huff does not provide control flow constructs or abstract away the inner working of the program stack. Only the most upmost performance sensitive applications take advantage of Huff, however it is a great educational tool to learn how the EVM interprets instructions its lowest level.
-   [Advanced Foundry course by Cyfrin Updraft](https://updraft.cyfrin.io/courses/advanced-foundry) teaches you about Foundry, how to develop a DeFi protocol and a stablecoin, how to develop a DAO, advanced smart contract development, advanced smart contracts testing and fuzzing and manual verification.
-   [Smart Contract Security course by Cyfrin Updraft](https://updraft.cyfrin.io/courses/security) will teach you everything you need to know to get started auditing and writing secure protocols.
-   [Assembly and Formal Verification course by Cyfrin Updraft](https://updraft.cyfrin.io/courses/formal-verification) teaches you about Assembly, writing smart contracts using Huff and Yul, Ethereum Virtual Machine OPCodes, Formal verification testing, Smart contract invariant testing and tools like Halmos, Certora, Kontrol.
-   [Smart Contract DevOps course by Cyfrin Updraft](https://updraft.cyfrin.io/courses/wallets) teaches about access control best practices when working with wallets, post-deployment security, smart contract and web3 devOps and live protocols maintenance and monitoring.
-   [Secureum YT Channel](https://www.youtube.com/@SecureumVideos/videos) has plenty videos about Solidity from Solidity Basics to all the way to advanced concepts like Fuzzing and Solidity auditing.

## Tutorials

-   [Ethernaut](https://ethernaut.openzeppelin.com/): learn by solving puzzles
-   [Damn Vulnerable DeFi](https://www.damnvulnerabledefi.xyz): DVD is a series of smart contract challenges which consists of vulnerable contracts and you are supposed to be able to hack it. These challenges are a good way to practice and apply the Solidity skills you have acquired.

## Best practices/patterns

-   [DeFi developer roadmap](https://github.com/OffcierCia/DeFi-Developer-Road-Map)
-   [RareSkills Book of Gas Optimization](https://www.rareskills.io/post/gas-optimization)

## Testing

-   [Echidna](https://github.com/crytic/echidna): fuzz testing
-   [Slither](https://github.com/crytic/slither): static analysis for vulnerability detection
-   [solidity-coverage](https://github.com/sc-forks/solidity-coverage/tree/master): code coverage for Solidity testing

## Smart contract archives

-   [Smart contract sanctuary](https://github.com/tintinweb/smart-contract-sanctuary) - contracts verified on Etherscan
-   [EVM function signature database](https://www.4byte.directory/)
---

# Use an Indexer

<CustomDocCardContainer>
    <CustomDocCard
        icon={<Monad />}
        link="/guides/indexers/ghost"
        title="GhostGraph"
        description="Index transfers with GhostGraph"
    />
    <CustomDocCard
        icon={<Monad />}
        link="/guides/indexers/tg-bot-using-envio"
        title="Envio"
        description="Index transfers for a telegram bot using Envio"
    />
    <CustomDocCard
        icon={<Monad />}
        link="/guides/indexers/quicknode-streams"
        title="QuickNode Streams"
        description="Index transfers using QuickNode Streams"
    />
</CustomDocCardContainer>

---

# How to index token transfers with GhostGraph

URL: https://docs.monad.xyz/guides/indexers/ghost.md

## Introduction

In this guide, you will create an ERC20 token on Monad Testnet and index its transfers with [GhostGraph](https://docs.tryghost.xyz/). You'll learn how to:

- Deploy a basic ERC20 token contract
- Test the contract locally
- Deploy to Monad Testnet
- Set up event tracking with GhostGraph

## Prerequisites

Before starting, ensure you have:

- Node.js installed (v16 or later)
- Git installed
- [Foundry](https://github.com/foundry-rs/foundry) installed
- Some MONAD testnet tokens (for gas fees)
- Basic knowledge of Solidity and ERC20 tokens
  

## Project Setup

First, clone the starter repository:

```sh
git clone https://github.com/chrischang/cat-token-tutorial.git
cd cat-token-tutorial
```

## CatToken Contract Implementation

The `src/CatToken.sol` contract implements a basic ERC20 token with a fixed supply. Here's the code:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract CatToken is ERC20 {
    /**
     * @dev Constructor that gives msg.sender all existing tokens.
     * Initial supply is 1 billion tokens.
     */
    constructor() ERC20("CatToken", "CAT") {
        // Mint initial supply of 1 billion tokens to deployer
        // This will emit a Transfer event that GhostGraph   can index
        _mint(msg.sender, 1_000_000_000 * 10 ** decimals());
    }
}
```

This implementation:

- Creates a token with name "CatToken" and symbol "CAT"
- Mints 1 billion tokens to the deployer's address
- Uses OpenZeppelin's battle-tested ERC20 implementation

## Testing the Contract

Navigate to the test file `test/CatToken.t.sol`:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract CatTokenTest is Test {
    CatToken public token;
    address public owner;
    address public user;

    function setUp() public {
        owner = address(this);
        user = address(0x1);

        token = new CatToken();
    }

    function testInitialSupply() public view {
        assertEq(token.totalSupply(), 1_000_000_000 * 10**18);
        assertEq(token.balanceOf(owner), 1_000_000_000 * 10**18);
    }

    function testTransfer() public {
        uint256 amount = 1_000_000 * 10**18;
        token.transfer(user, amount);
        assertEq(token.balanceOf(user), amount);
        assertEq(token.balanceOf(owner), 999_000_000 * 10**18);
    }
}
```

Run the tests:

```sh
forge test -vv
```

## Deployment Setup

### 1. Create a `.env` file:

```sh
cp .env.example .env
```

### 2. Add your credentials to `.env` file:

```sh
PRIVATE_KEY=your_private_key_here
MONAD_TESTNET_RPC=https://testnet-rpc.monad.xyz
```

### 3. Create deployment script `script/DeployCatToken.s.sol`:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract DeployCatToken is Script {
    function run() external {
        // Retrieve private key from environment
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");

        vm.startBroadcast(deployerPrivateKey);
        CatToken token = new CatToken();
        vm.stopBroadcast();

        // Log the token address - this will be needed for GhostGraph indexing and submit transactions
        console.log("CatToken deployed to:", address(token));
    }
}
```

## Deploying CatToken on Monad Testnet

### 1. Load environment variables:

```sh
source .env
```

### 2. Deploy the contract:

```sh
forge script script/DeployCatToken.s.sol \
--rpc-url $MONAD_TESTNET_RPC \
--broadcast
```

Save the deployed contract address for the next steps.

Remember to add `TOKEN_ADDRESS` into your `.env` file

You should now have

```sh
PRIVATE_KEY=your_private_key_here
MONAD_TESTNET_RPC=https://testnet-rpc.monad.xyz
TOKEN_ADDRESS=0x...
```

## Verify Smart Contract

### 1. Load environment variables:

```sh
source .env
```

### 2. Verify the contract:

```sh
forge verify-contract \
  --rpc-url $MONAD_TESTNET_RPC \
  --verifier sourcify \
  --verifier-url 'https://sourcify-api-monad.blockvision.org' \
  $TOKEN_ADDRESS \
  src/CatToken.sol:CatToken
```

After verification, you should see the contract verified on the [MonadExplorer](https://testnet.monadexplorer.com). You should see a checkmark and the banner stating the contract source code verified.

![Verified Contract](/img/guides/indexers/ghost/verified-contract.png)


## Script for Token Transfers Transactions Onchain

We perform some token transfer transactions onchain to trigger the `Transfer` event that GhostGraph will index.

View the transfer script `script/TransferCatTokens.s.sol`:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract TransferCatTokens is Script {
    function run() external {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        address token = vm.envAddress("TOKEN_ADDRESS");

        vm.startBroadcast(deployerPrivateKey);

        // Send tokens to test addresses
        CatToken(token).transfer(address(0x1), 1000 * 10**18);
        CatToken(token).transfer(address(0x2), 2000 * 10**18);
        CatToken(token).transfer(address(0x3), 3000 * 10**18);

        vm.stopBroadcast();
    }
}
```

Run the below command to execute transfers:

```sh
forge script script/TransferCatTokens.s.sol \
--rpc-url $MONAD_TESTNET_RPC \
--broadcast
```

You have now deployed your ERC-20 contract and submitted transactions on the Monad testnet. Let’s track these onchain events with GhostGraph.

## Setting Up GhostGraph Indexing

1. Visit [GhostGraph](https://tryghost.xyz/) and click sign up for an account

2. Create a new GhostGraph

![create_ghost_graph](/img/guides/indexers/ghost/create_ghost_graph.png)

3. Copy and paste this into `events.sol` file. We are interested in tracking token flow. Let’s insert this event here. To learn more about events: https://docs.tryghost.xyz/ghostgraph/getting-started/define-events

```solidity
interface Events {
    event Transfer(address indexed from, address indexed to, uint256 value);
}
```

4. Copy and paste this into `schema.sol` file. In this case, we are creating a few struct which we will use to save entity into the Ghost database. To learn more about schema: https://docs.tryghost.xyz/ghostgraph/getting-started/define-schema

```solidity
struct Global {
    string id;
    uint256 totalHolders;
}

struct User {
    address id;
    uint256 balance;
}

struct Transfer {
    string id;
    address from;
    address to;
    uint256 amount;

    uint64 block;
    address emitter;
    uint32 logIndex;
    bytes32 transactionHash;
    uint32 txIndex;
    uint32 timestamp;
}
```

5. Click on `generate code` button to generate `indexer.sol` file along with some other readonly files. This file will be where the logic and transformations resides.

6. Copy and paste this into `indexer.sol` be sure to insert your token address to the `CAT_TESTNET_TOKEN_CONTRACT_ADDRESS` variable.

```solidity
// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

contract MyIndex is GhostGraph {
    using StringHelpers for EventDetails;
    using StringHelpers for uint256;
    using StringHelpers for address;

    address constant CAT_TESTNET_TOKEN_CONTRACT_ADDRESS = <INSERT YOUR TOKEN ADDRESS>;

    function registerHandles() external {
        graph.registerHandle(CAT_TESTNET_TOKEN_CONTRACT_ADDRESS);
    }

    function onTransfer(EventDetails memory details, TransferEvent memory ev) external {
        // Get global state to track holder count
        Global memory global = graph.getGlobal("1");

        // Handle sender balance
        if (ev.from != address(0)) {
            // Skip if minting
            User memory sender = graph.getUser(ev.from);
            if (sender.balance == ev.value) {
                // User is transferring their entire balance
                global.totalHolders -= 1; // Decrease holder count
            }
            sender.balance -= ev.value;
            graph.saveUser(sender);
        }

        // Handle receiver balance
        User memory receiver = graph.getUser(ev.to);
        if (receiver.balance == 0 && ev.value > 0) {
            // New holder
            global.totalHolders += 1; // Increase holder count
        }
        receiver.balance += ev.value;
        graph.saveUser(receiver);

        // Save global state
        graph.saveGlobal(global);

        // Create and save transfer record
        Transfer memory transfer = graph.getTransfer(details.uniqueId());
        transfer.from = ev.from;
        transfer.to = ev.to;
        transfer.amount = ev.value;
        
        // Store transaction metadata
        transfer.block = details.block;
        transfer.emitter = details.emitter;
        transfer.logIndex = details.logIndex;
        transfer.transactionHash = details.transactionHash;
        transfer.txIndex = details.txIndex;
        transfer.timestamp = details.timestamp;
        
        graph.saveTransfer(transfer);
    }
}
```

7. Compile and deploy your GhostGraph. After a few seconds, you should see GhostGraph has successfully indexed your contract.

![ghostgraph_playground](/img/guides/indexers/ghost/ghostgraph_playground.png)

8. Clicking on the playground will take you to the GraphQL playground, where you can ensure the data is indexed correctly. Let’s copy and paste this into our playground and click the play button to fetch the data from GhostGraph.

```graphql
query FetchRecentTransfers {
  transfers(
    orderBy: "block", 
    orderDirection: "desc"
    limit: 50
  ) {
    items {
      amount
      block
      emitter
      from
      id
      logIndex
      timestamp
      to
      transactionHash
      txIndex
    }
  }
}
```

![graphql_playground](/img/guides/indexers/ghost/graphql_playground.png)

:::tip
Try submitting additional transactions by running the transfer script again. You should see that GhostGraph automatically indexes the new transactions.
:::

## Conclusion

You have now successfully created a GhostGraph to track onchain data for your contract. The next step is to connect it to your frontend.

The Ghost team has created end-to-end tutorials on how to do just that [here](https://docs.tryghost.xyz/blog/connect_ghost_graph_to_frontend/)

---

# How to index every WMON transfer using QuickNode Streams

URL: https://docs.monad.xyz/guides/indexers/quicknode-streams.md

In this guide, you will learn how to use QuickNode Streams to index every [WMON](https://testnet.monadexplorer.com/token/0x760AfE86e5de5fa0Ee542fc7B7B713e1c5425701) transfer, including internal transactions, on Monad Testnet.

## What is QuickNode Streams?

[QuickNode Streams](https://www.quicknode.com/docs/streams/getting-started) is a web3 data streaming solution supporting real-time and historical Monad data that offers:

- **Reliable Data Delivery** - Exactly-once, guaranteed delivery, seamlessly integrating with your data lake. Streams ensures every block, receipt, or trace is delivered exactly-once in the order of dataset finality, preventing issues like corrupt or missing data
- **Real-Time Data Consistency** - Consistent, live data streaming
- **Efficient Historical Data Handling** - Configurable date ranges and destinations for streamlined historical data management
- **Easy Integration** - Simple setup through a user-friendly interface
- **Transparent User Experience** - Clear logging, metrics, and usage tracking

## Setup Guide

### 1. Initial setup

1. Sign up for [QuickNode](https://dashboard.quicknode.com/?prompt=signup) and log into your dashboard.

2. Click on "Streams" in the left sidebar.

![QuickNode Dashboard](/img/guides/indexers/quicknode-streams/1.png)

3. Click on "Create Stream".

![Create Stream Button](/img/guides/indexers/quicknode-streams/2.png)

### 2. Configure Stream range

1. Give your stream a name. In this example we will name it `monad-quicknode-stream`.

2. In the "Network" section, select `Monad` from the dropdown.

3. In the "Stream Start" section you can choose to start the stream from the latest block or from a specific block number.

![Stream Configuration](/img/guides/indexers/quicknode-streams/3.png)

4. In the "Stream End" section you can choose to end the stream until manually paused or at a specific block number.

5. In the "Latest block delay" section, you can set a block number as a delay in receiving data. For this guide we will receive data as soon as it is available.<br/><br/>For example: If the block delay is `3`, you will receive data only when there is **new data available** for `3` blocks including latest block, this helps in case there is a reorg. 

6. In the "Restream on reorg" section you can decide if you would like to get updated data restreamed in case of a reorg. For this guide we will keep this off.

7. Once done click "Next".

![Additional Settings](/img/guides/indexers/quicknode-streams/4.png)

### 3. Set up dataset

1. In the "Dataset" dropdown you can select the dataset of your choice according to the use case. For this guide we will select `Block with Receipts` since we want to filter logs with events emitted by WMON contract.

- Optional: Enable "Batch messages" to receive multiple blocks in a single message. This can be useful when the stream is not starting from the latest block.

![Dataset Selection](/img/guides/indexers/quicknode-streams/5.png)

2. Feel free to test it out by entering a block number and clicking "Fetch payload".

![Raw Payload Example](/img/guides/indexers/quicknode-streams/6.png)

### 4. Create WMON Transfer filter

1. In the "Modify the stream payload" section, you can define filters by clicking **"Customize your payload"**. For this guide, we will filter to only retrieve receipts involving WMON transfers.

![modify stream image](/img/guides/indexers/quicknode-streams/7.png)

2. QuickNode has a set of filter templates. Select the **Decoded ERC20 transfers** template:

![image for filter](/img/guides/indexers/quicknode-streams/8.png)

3. The editor will appear:

![image of filter editor](/img/guides/indexers/quicknode-streams/9.png)

The current filter allows all ERC20 transfers through. Replace the filter code with:

```js
function main(stream) {  
  const erc20Abi = `[{
    "anonymous": false,
    "inputs": [
      {"indexed": true, "type": "address", "name": "from"},
      {"indexed": true, "type": "address", "name": "to"},
      {"indexed": false, "type": "uint256", "name": "value"}
    ],
    "name": "Transfer",
    "type": "event"
  }]`;
  
  const data = stream.data ? stream.data : stream;
  
  // Decodes logs from the receipts that match the Transfer event ABI
  var result = decodeEVMReceipts(data[0].receipts, [erc20Abi]);
  
  // Filter for receipts with decoded logs
  result = result.filter(receipt => {
        // Check if there are any ERC20 transfers
        if(receipt.decodedLogs) {
            // Check if there are any WMON transfers
            receipt.decodedLogs = receipt.decodedLogs.filter(log => log.address == "0x760AfE86e5de5fa0Ee542fc7B7B713e1c5425701");
            
            // Return receipt if there logs which indicate a WMON transfer.
            return receipt.decodedLogs.length > 0;
        }

        // Return nothing if there are no ERC20 transfers.
        return false;
    });
  
  return { result };
}
```

4. Test the filter with "Run test"

![run test image](/img/guides/indexers/quicknode-streams/10.png)

5. "Save & close" to save the filter.

![save & close image](/img/guides/indexers/quicknode-streams/11.png)

6. Click "Next"

### 5. Set up Stream destination

For this guide we will keep the stream destination simple and use `Webhook` as the "Destination Type".

1. Let's use a site like [Svix Play](https://www.svix.com/play/) to quickly get a webhook and test the stream.

![svix play image](/img/guides/indexers/quicknode-streams/12.png)

2. Copy the webhook url from Svix Play:

![svix play copy url image](/img/guides/indexers/quicknode-streams/13.png)

3. In QuickNode:
  - Select `Webhook` as destination type
  - Paste your webhook URL
  - We can keep the rest of the settings as default

![webhook dropdown image](/img/guides/indexers/quicknode-streams/14.png)

4. Click on "Check Connection" to test the webhook url. Check if you received the "PING" message in the Svix Play dashboard.

![check connection image](/img/guides/indexers/quicknode-streams/15.png)

![ping message image](/img/guides/indexers/quicknode-streams/16.png)

5. Click "Send Payload" to send a test payload to the webhook.

![add send payload image](/img/guides/indexers/quicknode-streams/17.png)

![svix payload image](/img/guides/indexers/quicknode-streams/18.png)

6. Finally click "Create a Stream" to create the stream.

![create stream image](/img/guides/indexers/quicknode-streams/19.png)

### 6. Launch and Monitor

You should now be able to see the stream delivering the messages to the webhook!

![stream delivering image](/img/guides/indexers/quicknode-streams/20.png)

![svix streaming receiving message video](/img/guides/indexers/quicknode-streams/1.gif)

You can pause the stream by clicking the switch in the top right corner.

![pause switch image](/img/guides/indexers/quicknode-streams/21.png)


## Next Steps

- Monitor your stream's performance in the QuickNode dashboard
- Adjust filter parameters as needed
- Connect to your production webhook endpoint when ready

Your stream will now track all WMON transfers until manually paused or until reaching your specified end block.
---

# How to build a transfer notification bot with Envio HyperIndex

URL: https://docs.monad.xyz/guides/indexers/tg-bot-using-envio.md

In this guide, you will learn how to use [Envio](https://envio.dev/) HyperIndex to create a Telegram bot that sends notifications whenever WMON tokens are transferred on the Monad Testnet. We'll walk through setting up both the indexer and the Telegram bot.

Envio HyperIndex is an open development framework for building blockchain application backends. It offers real-time indexing, automatic indexer generation from contract addresses, and triggers for external API calls.

## Prerequisites

You'll need the following installed:

- Node.js v18 or newer
- pnpm v8 or newer
- Docker Desktop (required for running the Envio indexer locally)

## Setting up the project

First, create and enter a new directory:

```shell
mkdir envio-mon && cd envio-mon
```

### Get the contract ABI

1. Create an `abi.json` file:
```shell
touch abi.json
```

2. Copy the [WrappedMonad](https://testnet.monadexplorer.com/token/0x760AfE86e5de5fa0Ee542fc7B7B713e1c5425701?tab=Contract) ABI from the explorer

![image of explorer](/img/guides/indexers/tg-bot-using-envio/1.png)

3. Paste the ABI into your `abi.json` file

### Initialize the project

Run the initialization command:
```shell
pnpx envio init
```

Follow the prompts:
1. Press Enter when asked for a folder name (to use current directory)
2. Select `TypeScript` as your language
3. Choose `Evm` as the blockchain ecosystem
4. Select `Contract Import` for initialization
5. Choose `Local ABI` as the import method
6. Enter `./abi.json` as the path to your ABI file
7. Select only the `Transfer` event to index
8. Choose `<Enter Network Id>` and input `10143` (Monad Testnet chain ID)
9. Enter `WrappedMonad` as the contract name
10. Input the contract address: `0x760AfE86e5de5fa0Ee542fc7B7B713e1c5425701`
11. Select `I'm finished` since we're only indexing one contract
12. Choose whether to create or add an existing API token. If you choose to create a new token, you'll be taken to a page that looks like this:

<img src="/img/guides/indexers/tg-bot-using-envio/2.png" alt="new API token view" width="600"/>

Once the project is initialized, you should see the following project structure in your project directory.

<img src="/img/guides/indexers/tg-bot-using-envio/3.png" alt="envio dashboard" width="400"/>

Add the following code to `config.yaml` file, to make transaction hash available in event handler:

```yaml
# default config...
field_selection:
    transaction_fields:
      - hash
```

*More details about the `field_selection` config [here](https://docs.envio.dev/docs/HyperIndex/configuration-file#field-selection)*

## Starting the indexer

Start Docker Desktop.

To start the indexer run the following command in the project directory:

```shell
pnpx envio dev
```

You should see something similar to the below image in your terminal; this means that the indexer is syncing and will eventually reach the tip of the chain.

<img src="/img/guides/indexers/tg-bot-using-envio/4.png" alt="envio indexer syncing" width="600"/>


You will also see this page open in your browser automatically, the password is `testing`.
<img src="/img/guides/indexers/tg-bot-using-envio/5.png" alt="hasura local page" width="600"/>

We can use this interface to query the indexer using GraphQL. Results will depend on the sync progress:

![query interface](/img/guides/indexers/tg-bot-using-envio/6.png)

Currently, the indexer is catching up to the tip of the chain. Once syncing is complete the indexer will be able to identify latest WMON transfers.

We can shut down the indexer for now, so we can proceed with Telegram integration.

## Creating the Telegram bot

1. Visit [BotFather](https://t.me/botfather) to create your bot and get an API token
2. Add these environment variables to your `.env` file:
```
ENVIO_BOT_TOKEN=<your_bot_token>
ENVIO_TELEGRAM_CHAT_ID=<your_chat_id>
```

To get your chat ID:
1. Create a Telegram group and add your bot
2. Send `/start` to the bot: `@YourBot /start`
3. Visit `https://api.telegram.org/bot<YourBOTToken>/getUpdates`
4. Look for the channel chat ID (it should start with "-")

:::note
If you don't see the chat ID, try removing and re-adding the bot to the group.
:::

The Telegram bot is now ready.

## Integrating Telegram API to HyperIndex Event Handler

Create a folder `libs` inside `src` folder in the project directory, create a file inside it `telegram.ts` and add the following code

```ts
// src/libs/telegram.ts

export const sendMessageToTelegram = async (message: string): Promise<void> => {
  try {
    const apiUrl = `https://api.telegram.org/bot${BOT_TOKEN}/sendMessage`;

    await axios.post(apiUrl, {
      chat_id: CHAT_ID,
      text: message,
      parse_mode: "HTML",
    });
  } catch (error) {
    console.error("Error sending message:", error);
  }
};
```

You will come across some errors, let's fix them.

Install `axios` package

```bash
pnpm i axios
```

Create a file in `src` folder called `constants.ts` and add the following code:

```ts
// src/constants.ts

export const EXPLORER_URL_MONAD = "https://testnet.monadexplorer.com/";

// Threshold for WMON transfer amount above which the bot sends a notification
export const THRESHOLD_WEI: string = process.env.ENVIO_THRESHOLD_WEI ?? "1000000000000000000"; // in wei

export const BOT_TOKEN = process.env.ENVIO_BOT_TOKEN; // Telegram bot token
export const CHAT_ID = process.env.ENVIO_TELEGRAM_CHAT_ID; // WMON Transfers Notification Channel ID

// Function to get explorer url for the provided address
export const explorerUrlAddress = (address: string) =>
  EXPLORER_URL_MONAD + "address/" + address;

// Function to get explorer url for the provided transaction hash
export const explorerUrlTx = (txHash: string) =>
  EXPLORER_URL_MONAD + "tx/" + txHash;
```

We can now edit the `EventHandlers.ts` in `src` folder, to add the code for sending the telegram message:

```ts
// src/EventHandlers.ts

// Other event handlers can be removed...

WrappedMonad.Transfer.handler(async ({ event, context }) => {
    const from_address = event.params.src;
    const to_address = event.params.dst;

  if (isIndexingAtHead(event.block.timestamp) && event.params.wad >= BigInt(THRESHOLD_WEI)) {
    // Only send a message when the indexer is indexing event from the time it was started and not historical transfers, and only message if the transfer amount is greater than or equal to THRESHOLD_WEI.

    // Example message
    // WMON Transfer ALERT: A new transfer has been made by 0x65C3564f1DD63eA81C11D8FE9a93F8FFb5615233 to 0xBA5Cf1c0c1238F60832618Ec49FC81e8C7C0CF01 for 2.0000 WMON! 🔥 - View on Explorer

    const msg = `WMON Transfer ALERT: A new transfer has been made by <a href="${explorerUrlAddress(from_address)}">${from_address}</a> to <a href="${explorerUrlAddress(to_address)}">${to_address}</a> for ${weiToEth(event.params.wad)} WMON! 🔥 - <a href="${explorerUrlTx(
      event.transaction.hash
    )}">View on Explorer</a>`;

    await sendMessageToTelegram(msg);
  }
});
```

Let us now fix the import error.

Create a file called `helpers.ts` in `src/libs` folder, paste the following code in it:

```ts
// src/libs/helpers.ts

// Used to ensure notifications are only sent while indexing at the head and not historical sync
const INDEXER_START_TIMESTAMP = Math.floor(new Date().getTime() / 1000);

export const isIndexingAtHead = (timestamp: number): boolean => {
    return timestamp >= INDEXER_START_TIMESTAMP;
}

// Convert wei to ether for human readability
export const weiToEth = (bigIntNumber: bigint): string => {
  // Convert BigInt to string
  const numberString = bigIntNumber.toString();

  const decimalPointsInEth = 18;

  // Extract integer part and decimal part
  const integerPart = numberString.substring(
    0,
    numberString.length - decimalPointsInEth
  );

  const decimalPart = numberString.slice(-decimalPointsInEth);

  // Insert decimal point
  const decimalString =
    (integerPart ? integerPart : "0") +
    "." +
    decimalPart.padStart(decimalPointsInEth, "0");

  // Add negative sign if necessary
  return decimalString.slice(0, -14);
};
```

That's it! We can now run the indexer, and the telegram bot will start sending messages in the telegram channel when the indexer detects a WMON transfer!

![example bot message](/img/guides/indexers/tg-bot-using-envio/9.png)
*Note: Screenshot was taken before message format was changed. The message will be slightly different if you followed the guide.*

:::note
You may not immediately start seeing messages because the indexer take some time to catch up to the tip of the the recent blocks.

The bot will only send notifications for transfers when the indexer detects a WMON transfer in finalized blocks, with timestamp greater than or equal to the indexer start time.
:::
---

# How to build an MCP server that can interact with Monad Testnet

URL: https://docs.monad.xyz/guides/monad-mcp.md

In this guide, you will learn how to build a [Model Context Protocol](https://github.com/modelcontextprotocol) (MCP) server that allows an MCP Client (Claude Desktop) to query Monad Testnet to check the MON balance of an account.

## What is MCP?

The [Model Context Protocol](https://github.com/modelcontextprotocol) (MCP) is a standard that allows AI models to interact with external tools and services.


## Prerequisites

- Node.js (v16 or later)
- `npm` or `yarn`
- Claude Desktop

## Getting started

1. Clone the [`monad-mcp-tutorial`](https://github.com/monad-developers/monad-mcp-tutorial) repository. This repository has some code that can help you get started quickly.

```shell
git clone https://github.com/monad-developers/monad-mcp-tutorial.git
```

2. Install dependencies:

```
npm install
```

## Building the MCP server

Monad Testnet-related configuration is already added to `index.ts` in the `src` folder.

### Define the server instance

```ts
// Create a new MCP server instance
const server = new McpServer({
  name: "monad-mcp-tutorial",
  version: "0.0.1",
  // Array of supported tool names that clients can call
  capabilities: ["get-mon-balance"]
});
```

### Define the MON balance tool

Below is the scaffold of the `get-mon-balance` tool:

```ts
server.tool(
    // Tool ID 
    "get-mon-balance",
    // Description of what the tool does
    "Get MON balance for an address on Monad testnet",
    // Input schema
    {
        address: z.string().describe("Monad testnet address to check balance for"),
    },
    // Tool implementation
    async ({ address }) => {
        // code to check MON balance
    }
);
```

Let's add the MON balance check implementation to the tool:

```ts
server.tool(
    // Tool ID 
    "get-mon-balance",
    // Description of what the tool does
    "Get MON balance for an address on Monad testnet",
    // Input schema
    {
        address: z.string().describe("Monad testnet address to check balance for"),
    },
    // Tool implementation
    async ({ address }) => {
        try {
            // Check MON balance for the input address
            const balance = await publicClient.getBalance({
                address: address as `0x${string}`,
            });

            // Return a human friendly message indicating the balance.
            return {
                content: [
                    {
                        type: "text",
                        text: `Balance for ${address}: ${formatUnits(balance, 18)} MON`,
                    },
                ],
            };
        } catch (error) {
            // If the balance check process fails, return a graceful message back to the MCP client indicating a failure.
            return {
                content: [
                    {
                        type: "text",
                        text: `Failed to retrieve balance for address: ${address}. Error: ${
                        error instanceof Error ? error.message : String(error)
                        }`,
                    },
                ],
            };
        }
    }
);
```

### Initialize the transport and server from the `main` function

```ts
async function main() {
    // Create a transport layer using standard input/output
    const transport = new StdioServerTransport();
    
    // Connect the server to the transport
    await server.connect(transport);
}
```

### Build the project

```shell
npm run build
```

The server is now ready to use!

### Add the MCP server to Claude Desktop

1. Open "Claude Desktop"

![claude desktop](https://github.com/monad-developers/monad-mcp-tutorial/blob/main/static/1.png?raw=true)

2. Open Settings

Claude > Settings > Developer

![claude settings](https://github.com/monad-developers/monad-mcp-tutorial/blob/main/static/claude_settings.gif?raw=true)

3. Open `claude_desktop_config.json` 

![claude config](https://github.com/monad-developers/monad-mcp-tutorial/blob/main/static/config.gif?raw=true)

4. Add details about the MCP server and save the file.

```json
{
  "mcpServers": {
    ...
    "monad-mcp": {
      "command": "node",
      "args": [
        "/<path-to-project>/build/index.js"
      ]
    }
  }
}
```

5. Restart "Claude Desktop"

### Use the MCP server

You should now be able to see the tools in Claude!

![tools](https://github.com/monad-developers/monad-mcp-tutorial/blob/main/static/tools.gif?raw=true)

Here's the final result

![final result](https://github.com/monad-developers/monad-mcp-tutorial/blob/main/static/final_result.gif?raw=true)

## Further resources

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/introduction)
- [Monad Documentation](https://docs.monad.xyz/)
- [Viem Documentation](https://viem.sh/)






---

# How to connect a wallet to your app with Reown AppKit

URL: https://docs.monad.xyz/guides/reown-guide.md

In this guide, you'll learn how to use Reown AppKit to enable wallet connections and interact with the Monad network.

With AppKit, you can provide seamless wallet connections, including email and social logins, smart accounts, one-click authentication, and wallet notifications, all designed to deliver an exceptional user experience.

In this tutorial, you will learn how to:

1. Initialize a new project using AppKit CLI and setting up the Project ID.
2. Configure the project with Monad Testnet.

This guide takes approximately 5 minutes to complete.

## Setup

In this section, you'll set up the development environment to use AppKit with Monad. 

For this tutorial, we'll be using Next.js, though you can use any other framework compatible with AppKit.

:::note
AppKit is available on eight frameworks, including React, Next.js, Vue, JavaScript, React Native, Flutter, Android, iOS, and Unity.
:::

Now, let’s create a Next app using the CLI. In order to do so, please run the command given below:

```bash
npx @reown/appkit-cli
```

The above command uses the AppKit CLI to allow you to effortlessly set up a simple web app configured with Reown AppKit.

After running the command, you will be prompted to confirm the installation of the CLI. Upon your confirmation, the CLI will request the following details:

1. **Project Name**: Enter the name for your project. (eg: `my-monad-appkit-app`)
2. **Framework**: Select your preferred framework or library. Currently, you have three options: `React`, `Next.js`, and `Vue`.
3. **Network-Specific libraries**: Choose whether you want to install Wagmi, Ethers, Solana, or Multichain (EVM + Solana). In this case, you need to either pick `Wagmi` or `Ethers` since Monad is an EVM compatible blockchain. We will be choosing `Wagmi` for the sake of this tutorial.

After providing the project name and selecting your preferences, the CLI will install a minimal example of AppKit with your preferred blockchain library. 

When the example installation is complete, you can enter the project directory by running the below command in your terminal.

```bash
cd my-monad-appkit-app
```

Now, you need to install the dependencies required to run the AppKit project. In order to do this, please run the command given below.

```bash
npm install
```

:::note
You can also use other package managers such as `yarn`, `bun`, `pnpm`, etc. 
:::

### Create a new project on Reown Cloud

We need to get a Project ID from Reown Cloud that we will use to set up AppKit with Wagmi config. Navigate to [cloud.reown.com](https://cloud.reown.com) and sign in. If you have not created an account yet, please do so before we proceed.

After you have logged in, please navigate to the "Projects" section of the Cloud and click on "Create Project". 

![Create Project](/img/guides/reown-guide/1.png)

Now, enter the name for your project and click on "Continue".

![Enter Project Name](/img/guides/reown-guide/2.png)

Select the product as "AppKit" and click on "Continue". 

![Select Product](/img/guides/reown-guide/3.png)

Select the framework as "Next.js" and click on "Create". Reown Cloud will now create a new project for you which will also generate a Project ID. 

![Select Framework](/img/guides/reown-guide/4.png)

You will notice that your project was successfully created. On the top left corner, you will be able to find your Project ID. Please copy that as you will need that later. 

![Project ID](/img/guides/reown-guide/5.png)

### Set up the Project ID

Before we build the app, let’s first configure our `.env` file. Open the project that you created using the AppKit CLI in your preferred code editor.

On the root level of your code directory, create a new file named `.env`.

Open that file and create a new variable `NEXT_PUBLIC_PROJECT_ID`. We will assign the project Id that we copied in the previous step to this environment variable that we just created. This is what it will look like:

```js
NEXT_PUBLIC_PROJECT_ID ="YOUR_PROJECT_ID_HERE"
```

:::warning
Note: Please make sure you follow the best practices when you are working with secret keys and other sensitive information. Environment variables that start with `NEXT_PUBLIC` will be exposed by your app which can be misused by bad actors. 
:::

## Configure AppKit with Monad Testnet

Navigate to `/src/config/index.ts` file.

Within this code file, you can notice that the networks configured with AppKit are being pulled from `@reown/appkit/networks`. Please update the corresponding import statement as shown below.

```ts
<CustomDocCardContainer>
    <CustomDocCard
        icon={<Foundry />}
        link="/guides/verify-smart-contract/foundry"
        title="Foundry"
        description="Verify a smart contract on Monad using Foundry"
    />
    <CustomDocCard
        icon={<Hardhat />}
        link="/guides/verify-smart-contract/hardhat"
        title="Hardhat"
        description="Verify a smart contract on Monad using Hardhat"
    />
</CustomDocCardContainer>


---

# Verify a smart contract on Monad Explorer using Foundry

URL: https://docs.monad.xyz/guides/verify-smart-contract/foundry

Once your contract is deployed to a live network, the next step is to verify its source code on the block explorer.

Verifying a contract means uploading its source code, along with the settings used to compile the code, to a
repository (typically maintained by a block explorer). This allows anyone to compile it and compare the generated
bytecode with what is deployed on chain. Doing this is extremely important in an open platform like Monad.

In this guide we'll explain how to do this on [MonadExplorer](https://testnet.monadexplorer.com) using [Foundry](https://getfoundry.sh/).

<Tabs>
    <TabItem
        value="with-foundry-monad"
        label="Foundry Monad template (Recommended)"
        default
    >
        If you are using [`foundry-monad`](https://github.com/monad-developers/foundry-monad) template, you can simply run the below command:

        ```sh
        forge verify-contract \
            <contract_address> \
            <contract_name> \
            --chain 10143 \
            --verifier sourcify \
            --verifier-url https://sourcify-api-monad.blockvision.org
        ```

        Example:

        ```sh
        forge verify-contract \
            0x195B9401D1BF64D4D4FFbEecD10aE8c41bEBA453 \
            src/Counter.sol:Counter \
            --chain 10143 \
            --verifier sourcify \
            --verifier-url https://sourcify-api-monad.blockvision.org
        ```
    </TabItem>
    <TabItem value="default-foundry-project"
        label="Default Foundry Project">
        :::tip
        If you use [`foundry-monad`](https://github.com/monad-developers/foundry-monad) you can skip the configuration step
        :::

        ## 1. Update `foundry.toml` with Monad Configuration

        ```toml
        [profile.default]
        src = "src"
        out = "out"
        libs = ["lib"]
        metadata = true
        metadata_hash = "none"  # disable ipfs
        use_literal_content = true # use source code

        # Monad Configuration
        eth-rpc-url="https://testnet-rpc.monad.xyz"
        chain_id = 10143
        ```

        ## 2. Verify the contract using the command below:

        ```sh
        forge verify-contract \
            <contract_address> \
            <contract_name> \
            --verify \
            --verifier sourcify \
            --verifier-url https://sourcify-api-monad.blockvision.org
        ```

        Example:

        ```sh
        forge verify-contract \
            0x195B9401D1BF64D4D4FFbEecD10aE8c41bEBA453 \
            src/Counter.sol:Counter \
            --verify \
            --verifier sourcify \
            --verifier-url https://sourcify-api-monad.blockvision.org
        ```
    </TabItem>

</Tabs>

On successful verification of smart contract, you should get a similar output in your terminal:

```sh
Start verifying contract `0x195B9401D1BF64D4D4FFbEecD10aE8c41bEBA453` deployed on 10143

Submitting verification for [Counter] "0x195B9401D1BF64D4D4FFbEecD10aE8c41bEBA453".
Contract successfully verified
```

Now check the contract on [Monad Explorer](https://testnet.monadexplorer.com/).
---

# Verify a smart contract on Monad Explorer using Hardhat

URL: https://docs.monad.xyz/guides/verify-smart-contract/hardhat

Once your contract is deployed to a live network, the next step is to verify its source code on the block explorer.

Verifying a contract means uploading its source code, along with the settings used to compile the code, to a
repository (typically maintained by a block explorer). This allows anyone to compile it and compare the generated
bytecode with what is deployed on chain. Doing this is extremely important in an open platform like Monad.

In this guide we'll explain how to do this on [MonadExplorer](https://testnet.monadexplorer.com) using [Hardhat](https://hardhat.org/).

## 1. Update your `hardhat.config.ts` file to include the `monadTestnet` configuration if not already present

```ts
const config: HardhatUserConfig = {
  solidity: {
    version: "0.8.28",
    settings: {
      metadata: {
        bytecodeHash: "none", // disable ipfs
        useLiteralContent: true, // use source code
      },
    },
  },
  networks: {
    monadTestnet: {
      url: "https://testnet-rpc.monad.xyz",
      chainId: 10143,
    },
  },
  sourcify: {
    enabled: true,
    apiUrl: "https://sourcify-api-monad.blockvision.org",
    browserUrl: "https://testnet.monadexplorer.com",
  },
  // To avoid errors from Etherscan
  etherscan: {
    enabled: false,
  },
};

export default config;
```

## 3. Verify the smart contract

Use the following command to verify the smart contract:

```sh
npx hardhat verify <contract_address> --network monadTestnet
```

On successful verification of smart contract, the output should be similar to the following:

```
Successfully verified contract Lock on Sourcify.
https://testnet.monadexplorer.com/contracts/full_match/10143/<contract_address>/
```

Using the link in the output above, you can view the verified smart contract on the explorer.

Now check the contract on [Monad Explorer](https://testnet.monadexplorer.com/).

---


# Introduction

Monad is a high-performance Ethereum-compatible L1. Monad materially advances the efficient frontier in the balance between decentralization and scalability.

Monad introduces optimizations in four major areas, resulting in a blockchain with throughput of over 10,000 transactions per second (tps):

-   [MonadBFT](monad-arch/consensus/monad-bft.mdx)
-   [Asynchronous Execution](monad-arch/consensus/asynchronous-execution.mdx)
-   [Parallel Execution](monad-arch/execution/parallel-execution.md)
-   [MonadDb](monad-arch/execution/monaddb.md)

Monad's improvements address existing bottlenecks while preserving seamless compatibility for application developers (full EVM bytecode equivalence) and users (Ethereum [RPC API](reference/json-rpc) compatibility).

For an executive summary, see [Monad for Users](/introduction/monad-for-users) or [Monad for Developers](/introduction/monad-for-developers).

## Architecture

The Monad client is built with a focus on performance and is written from scratch in C++ and Rust. 

The subsequent pages survey the major [architectural changes](monad-arch) in Monad as well as the interface for users.

## Testnet

Monad's public testnet is live! Head to [Network information](developer-essentials/network-information) to get started.

Many leading Ethereum developer tools support Monad testnet. See the [Tooling and Infrastructure](tooling-and-infra/README.md) page for a list of supported providers by category.

---


# Monad for Developers

Monad is an Ethereum-compatible Layer-1 blockchain with 10,000 tps of throughput, 500ms block frequency, and 1s finality.

Monad's implementation of the Ethereum Virtual Machine complies with the [Cancun fork](https://www.evm.codes/?fork=cancun); simulation of historical Ethereum transactions with the Monad execution environment produces identical outcomes. Monad also offers full Ethereum RPC compatibility so that users can interact with Monad using familiar tools like Etherscan, Phantom, or MetaMask.

Monad accomplishes these performance improvements, while preserving backward compatibility, through the introduction of several major innovations:

-   [MonadBFT](monad-arch/consensus/monad-bft.mdx) (pipelined HotStuff consensus with additional research improvements)
-   [Asynchronous Execution](monad-arch/consensus/asynchronous-execution.mdx) (pipelining between consensus and execution to significantly increase the execution budget)
-   [Parallel Execution](monad-arch/execution/parallel-execution.md)
-   [MonadDb](monad-arch/execution/monaddb.md) (high-performance state backend)

Although Monad features parallel execution and pipelining, it's important to note that blocks in Monad are linear, and transactions are linearly ordered within each block.

## Transactions

<table data-header-hidden>
    <thead>
        <tr>
            <th width="248.5"></th>
            <th></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Address space</td>
            <td>Same address space as Ethereum (20-byte addresses using ECDSA)</td>
        </tr>
        <tr>
            <td>Transaction format/types</td>
            <td>[Same as Ethereum](https://ethereum.org/en/developers/docs/transactions/). Monad transactions use the same typed transaction envelope introduced in [EIP-2718](https://eips.ethereum.org/EIPS/eip-2718), encoded with [RLP](https://ethereum.org/en/developers/docs/data-structures-and-encoding/rlp/).<br/><br/>Transaction type 0 ("legacy"), 1 ("EIP-2930"), and 2 ("EIP-1559"; now the default in Ethereum) are supported. See [transaction type reference](https://ethereum.org/en/developers/docs/transactions/#typed-transaction-envelope).<br/><br/>Transaction type 3 ("EIP-4844") is not yet supported in testnet.</td>
        </tr>
        <tr>
            <td>Wallet compatibility</td>
            <td>Monad is compatible with standard Ethereum wallets such as Phantom or MetaMask. The only change required is to alter the RPC URL and chain id.</td>
        </tr>
        <tr>
            <td>Gas pricing</td>
            <td>Monad is EIP-1559-compatible; base fee and priority fee work as in Ethereum.<br/>
            <br/>Transactions are ordered according to a Priority Gas Auction (descending total gas price).<br/>
            <br/>In testnet, base fee is hard-coded to 50 gwei, although it will become dynamic in the future.<br/>
            <br/>In testnet, **transactions are charged based on gas limit rather than gas usage**, i.e. total tokens deducted from the sender's balance is `value + gas_price * gas_limit`. This is a DOS-prevention measure for asynchronous execution.<br/>
            <br/>See [Gas in Monad](/developer-essentials/gas-on-monad.md) for more details.</td>
        </tr>
    </tbody>
</table>

## Smart contracts

<table data-header-hidden>
    <thead>
        <tr>
            <th width="248.5"></th>
            <th></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Opcodes</td>
            <td>Monad supports EVM bytecode, and is bytecode-equivalent to Ethereum. All [opcodes](https://www.evm.codes/?fork=cancun) (as of the Cancun fork, e.g. TLOAD, TSTORE, and MCOPY) are supported.</td>
        </tr>
        <tr>
            <td>Opcode pricing</td>
            <td>Opcode pricing matches Ethereum as of the Cancun fork.</td>
        </tr>
        <tr>
            <td>Max contract size</td>
            <td>128 kb (up from 24 kb in Ethereum)</td>
        </tr>
    </tbody>
</table>

## Consensus

<table data-header-hidden>
    <thead>
        <tr>
            <th width="191.5"></th>
            <th></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Sybil resistance mechanism</td>
            <td>Proof-of-Stake (PoS)</td>
        </tr>
        <tr>
            <td>Delegation</td>
            <td>Allowed (in-protocol)</td>
        </tr>
        <tr>
            <td>Consensus mechanism and pipelining</td>
            <td>[MonadBFT](monad-arch/consensus/monad-bft.mdx) is a leader-based algorithm for reaching agreement about transaction order and inclusion under partially synchronous conditions. Broadly characterized, it is a derivative of HotStuff with additional research improvements.<br/><br/>MonadBFT is a pipelined 2-phase BFT algorithm with linear communication overhead in the common case. As in most BFT algorithms, communication proceeds in phases. At each phase, the leader sends a signed message to the voters, who send back signed responses.  Pipelining allows the quorum certificate (QC) or timeout certificate (TC) for block <code>k</code> to piggyback on the proposal for block <code>k+1.</code> Timeouts incur quadratic messaging.</td>
        </tr>
        <tr>
            <td>Block Frequency</td>
            <td>500 ms</td>
        </tr>
        <tr>
            <td>Finality</td>
            <td>1 second</td>
        </tr>
        <tr>
            <td>Mempool</td>
            <td>Leaders maintain a local mempool. When an RPC receives a transaction, it forwards it to the next 3 leaders who keep it in their local mempool. Additional forwarding may be added at a later time.</td>
        </tr>
        <tr>
            <td>Consensus participants</td>
            <td>Direct consensus participants vote on block proposals and serve as leaders. To serve as a direct participant, a node must have at least <code>MinStake</code> staked and be in the top <code>MaxConsensusNodes</code> participants by stake weight. These parameters are set in code.</td>
        </tr>
        <tr>
            <td>Transaction hashing</td>
            <td>For efficiency, block proposals refer to transactions by hash only.  If a node does not have a transaction, it will request the transaction by hash from a neighbor.</td>
        </tr>
        <tr>
            <td>Asynchronous execution</td>
            <td>
                <p>In Monad, consensus and execution occur in a pipelined fashion.  Nodes come to consensus on the official transaction order <em>prior</em> to executing that ordering ([Asynchronous Execution](monad-arch/consensus/asynchronous-execution.mdx)); the outcome of execution is <em>not</em> a prerequisite to consensus.</p>
                <p></p>
                <p>In blockchains where execution <em>is</em> a prerequisite to consensus, the time budget for execution is a small fraction of the block time.  Pipelining consensus and execution allows Monad to expend the full block time on <em>both</em> consensus and execution.<br/></p>
                <p>Block proposals consist of an ordered list of transaction hashes and a state merkle root from <code>D</code> blocks ago.  The delay parameter <code>D</code> is set in code; it is expected that <code>D = 3</code> initially.</p>
                <p></p>
                <p>To prevent spam, nodes validate that the account balance is sufficient to pay for `value + gas_price * gas_limit` for transactions submitted during the delay period of <code>D</code> blocks.</p>
                <p></p>
                <p>An account's available balance computed by consensus (as of <code>D</code> blocks ago) is effectively a budget for "in-flight" orders; it exists to ensure that the account can pay for all submitted transactions.</p>
            </td>
        </tr>
        <tr>
            <td>State determinism </td>
            <td>Finality occurs at consensus time; the official ordering of transactions is enshrined at this point, and the outcome is fully deterministic for any full node, who will generally execute the transactions for that new block in under 1 second.<br/><br/>The <code>D</code>-block delay for state merkle roots is only for state root verification, for example for allowing a node to ensure that it didn't make a computation error.</td>
        </tr>
    </tbody>
</table>

## Execution

The execution phase for each block begins after consensus is reached on that block, allowing the node to proceed with consensus on subsequent blocks.

### Parallel Execution

Transactions are linearly ordered; the job of execution is to arrive at the state that results from executing that list of transactions serially. The naive approach is just to execute the transactions one after another. Can we do better? Yes we can!

Monad implements [parallel execution](monad-arch/execution/parallel-execution.md):

-   An executor is a virtual machine for executing transactions. Monad runs many executors in parallel.
-   An executor takes a transaction and produces a **result**. A result is a list of **inputs** to and **outputs** of the transactions, where inputs are (ContractAddress, Slot, Value) tuples that were SLOADed in the course of execution, and outputs are (ContractAddress, Slot, Value) tuples that were SSTOREd as a result of the transaction.
-   Results are initially produced in a pending state; they are then committed in the original order of the transactions. When a result is committed, its outputs update the current state. When it is a result’s turn to be committed, Monad checks that its inputs still match the current state; if they don’t, Monad reschedules the transaction. As a result of this concurrency control, Monad’s execution is guaranteed to produce the same result as if transactions were run serially.
-   When transactions are rescheduled, many or all of the required inputs are cached, so re-execution is generally relatively inexpensive. Note that upon re-execution, a transaction may produce a different set of Inputs than the previous execution did;

### MonadDb: high-performance state backend

All active state is stored in [MonadDb](monad-arch/execution/monaddb.md), a storage backend for solid-state drives (SSDs) that is optimized for storing merkle trie data. Updates are batched so that the merkle root can be updated efficiently.

MonadDb implements in-memory caching and uses [asio](https://think-async.com/Asio/) for efficient asynchronous reads and writes. Nodes should have 32 GB of RAM for optimal performance.

## Comparison to Ethereum: User's Perspective

<table>
    <thead>
        <tr>
            <th width="218">Attribute</th>
            <th width="264.3333333333333">Ethereum</th>
            <th width="255">Monad</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Transactions/second</strong> (smart contract calls and transfers)</td>
            <td>~10</td>
            <td>~10,000</td>
        </tr>
        <tr>
            <td><strong>Block Frequency</strong></td>
            <td>12s</td>
            <td>500 ms</td>
        </tr>
        <tr>
            <td><strong>Finality</strong></td>
            <td>[2 epochs](https://hackmd.io/@prysmaticlabs/finality) (12-18 min)</td>
            <td>1s</td>
        </tr>
        <tr>
            <td><strong>Bytecode standard</strong></td>
            <td>EVM ([Cancun fork](https://www.evm.codes/?fork=cancun))</td>
            <td>EVM ([Cancun fork](https://www.evm.codes/?fork=cancun))</td>
        </tr>
        <tr>
            <td><strong>Max contract size</strong></td>
            <td>24 kb</td>
            <td>128 kb</td>
        </tr>
        <tr>
            <td><strong>RPC API</strong></td>
            <td>[Ethereum RPC API](https://ethereum.org/en/developers/docs/apis/json-rpc/)</td>
            <td>[Monad RPC API](/reference/json-rpc) (generally identical to Ethereum RPC API, see [differences](reference/rpc-differences.md))</td>
        </tr>
        <tr>
            <td><strong>Cryptography</strong></td>
            <td>ECDSA</td>
            <td>ECDSA</td>
        </tr>
        <tr>
            <td><strong>Accounts</strong></td>
            <td>Last 20 bytes of keccak-256 of public key under ECDSA</td>
            <td>Last 20 bytes of keccak-256 of public key under ECDSA</td>
        </tr>
        <tr>
            <td><strong>Consensus mechanism</strong></td>
            <td>Gasper <br/>(Casper-FFG finality gadget + <br/>LMD-GHOST fork-choice rule)</td>
            <td>[MonadBFT](monad-arch/consensus/monad-bft.mdx) (pipelined HotStuff with additional research improvements)</td>
        </tr>
        <tr>
            <td><strong>Mempool</strong></td>
            <td>Yes</td>
            <td>Yes</td>
        </tr>
        <tr>
            <td><strong>Transaction ordering</strong></td>
            <td>Leader's discretion (in practice, PBS)</td>
            <td>Leader's discretion (default behavior: priority gas auction)</td>
        </tr>
        <tr>
            <td><strong>Sybil-resistance mechanism</strong></td>
            <td>PoS</td>
            <td>PoS</td>
        </tr>
        <tr>
            <td><strong>Delegation allowed</strong></td>
            <td>No; pseudo-delegation through LSTs</td>
            <td>Yes</td>
        </tr>
        <tr>
            <td><strong>Hardware Requirements</strong> (full node)</td>
            <td>4-core CPU<br/>16 GB RAM<br/>1 TB SSD<br/>25 Mbit/s bandwidth</td>
            <td>16-core CPU<br/>32 GB RAM<br/>2 x 2 TB SSD (one dedicated for MonadDB)<br/>100 Mbit/s bandwidth</td>
        </tr>
    </tbody>
</table>

## Testnet

Monad's public testnet is live. Head to [Network Information](developer-essentials/network-information.md) to get started.

Many leading Ethereum developer tools support Monad testnet. See the [Tooling and Infrastructure](tooling-and-infra/README.md) for a list of supported providers by category.

---


# Monad for Users

Monad is a high-performance Ethereum-compatible L1, offering users the best of both worlds: **portability** and **performance**.

From a portability perspective, Monad offers **full bytecode compatibility** for the Ethereum Virtual Machine (EVM), so that applications built for Ethereum can be ported to Monad without code changes, and **full Ethereum RPC compatibility**, so that infrastructure like Etherscan or The Graph can be used seamlessly.

From a performance perspective, Monad offers **10,000 tps** of throughput, i.e. 1 billion transactions per day, while offering **500ms block frequency** and **1 second finality**. This allows Monad to support many more users and far more interactive experiences than existing blockchains, while offering far cheaper per-transaction costs.

## What's familiar about Monad?

From a user perspective, Monad behaves very similarly to Ethereum. You can use the same wallets (e.g. Phantom, MetaMask) or block explorers (e.g. Etherscan) to sign or view transactions. The same apps built for Ethereum can be ported to Monad without code changes, so it is expected that you'll be able to use many of your favorite apps from Ethereum on Monad. The address space in Monad is the same as in Ethereum, so you can reuse your existing keys.

Like Ethereum, Monad features linear blocks, and linear ordering of transactions within a block.&#x20;

Like Ethereum, Monad is a proof-of-stake network maintained by a decentralized set of validators. Anyone can run a node to independently verify transaction execution, and significant care has been taken to keep hardware requirements minimal.

## What's different about Monad?

Monad makes exceptional performance possible by introducing **parallel execution** and **superscalar pipelining** to the Ethereum Virtual Machine.

**Parallel execution** is the practice of utilizing multiple cores and threads to strategically execute work in parallel while still committing the results in the original order. Although transactions are executed in parallel "under the hood", from the user and developer perspective they are executed serially; the result of a series of transactions is always the same as if the transactions had been executed one after another.

**Superscalar pipelining** is the practice of creating stages of work and executing the stages in parallel. A simple diagram tells the story:

<figure>
    <img src="/img/pipelining.png" alt="Pipelining, Laundry Day" class="center"> </img>
    <figcaption id="center">Pipelining laundry day. Top: Naive; Bottom: Pipelined. Credit: [Prof. Lois Hawkes, FSU](https://www.cs.fsu.edu/~hawkes/cda3101lects/chap6/index.html?$$$F6.1.html$$$)</figcaption>
</figure>

When doing four loads of laundry, the naive strategy is to wash, dry, fold, and store the first load of laundry before starting on the second one. The pipelined strategy is to start washing load 2 when load 1 goes into the dryer. Pipelining gets work done more efficiently by utilizing multiple resources simultaneously.

**Monad** introduces pipelining to address existing bottlenecks in state storage, transaction processing, and distributed consensus. In particular, Monad introduces pipelining and other optimizations in four major areas:

-   [MonadBFT](/monad-arch/consensus/monad-bft.mdx) (pipelined HotStuff consensus with additional research improvements)
-   [Asynchronous Execution](/monad-arch/consensus/asynchronous-execution.mdx) (pipelining between consensus and execution to significantly increase the execution budget)
-   [Parallel Execution](/monad-arch/execution/parallel-execution.md)
-   [MonadDb](/monad-arch/execution/monaddb.md) (high-performance state backend)

Monad's client, which was written from scratch in C++ and Rust, reflect these architectural improvements and result in a platform for decentralized apps that can truly scale to world adoption.

## Why should I care?

Decentralized apps are replacements for centralized services with several significant advantages:

-   **Open APIs / composability**: decentralized apps can be called atomically by other decentralized apps, allowing developers to build more complex functionality by stacking existing components.
-   **Transparency**: app logic is expressed purely through code, so anyone can review the logic for side effects. State is transparent and auditable; proof of reserves in DeFi is the default.
-   **Censorship-resistance and credible neutrality:** anyone can submit transactions or upload applications to a permissionless network.
-   **Global reach**: anyone with access to the internet can access crucial financial services, including unbanked/underbanked users.

However, decentralized apps need cheap, performant infrastructure to reach their intended level of impact. A single app with 1 million daily active users (DAUs) and 10 transactions per user per day would require 10 million transactions per day, or 100 tps. A quick glance at [L2Beat](https://l2beat.com/scaling/activity) - a useful website summarizing the throughput and decentralization of existing EVM-compatible L1s and L2s - shows that no EVM blockchain supports even close to that level of throughput right now.

Monad materially improves on the performance of an EVM-compatible blockchain network, pioneering several innovations that will hopefully become standard in Ethereum in the years to come.

With Monad, developers, users, and researchers can reuse the wealth of existing applications, libraries, and applied cryptography research that have all been built for the EVM.

## Testnet

Monad's public testnet is live. Head to [Network Information](developer-essentials/network-information.md) to get started.

---


# Why Blockchain?

A blockchain is decentralized agreement among a diverse set of participants about two things:

1. An official **ordering** (ledger) of transactions
2. An official **state of the world**, including balances of accounts and the state of various programs.

In modern blockchains such as Ethereum, transactions consist of balance transfers, creation of new programs, and function calls against existing programs. The aggregate result of all transactions up to now produces the current state, which is why _agreement about (1) above implies agreement about (2)._

A blockchain system has a set of protocol rules, also known as a consensus mechanism, which describe how a distributed set of nodes which are currently in sync will communicate with each other to agree upon additional transactions to add to the ledger. ([MonadBFT](/monad-arch/consensus/monad-bft.mdx) is an example of a consensus mechanism.)

Induction keeps the nodes in sync: they start with the same state and apply the same transactions, so at the end of applying a new list of transactions, they still have consistent state.

Shared global state enables the development of decentralized apps - apps that live "on the blockchain", i.e. on each of the nodes in the blockchain system. A decentralized app is a chunk of code (as well as persistent, app-specific state) that can get invoked by any user, who does so by submitting a transaction pointing to a function on that app. Each of the nodes in the blockchain is responsible for correctly executing the bytecode being called; duplication keeps each node honest.

## An example app

Decentralized apps can implement functionality that we might otherwise expect to be implemented in a centralized fashion. For example, a very simple example of a decentralized app is a _Virtual Bank_ (typically referred to in crypto as a Lending Protocol).

In the physical world, a bank is a business that takes deposits and loans them out at a higher rate. The bank makes the spread between the high rate and the low rate; the borrower gets a loan to do something economically productive; and you earn interest on your deposits. Everyone wins!

A Virtual Bank is simply an app with four major methods: `deposit`, `withdraw`, `borrow`, and `repay`. The logic for each of those methods is mostly bookkeeping to ensure that deposits and loans are being tracked correctly:

```
class VirtualBank:
  def deposit(sender, amount):
    # transfer amount from sender to myself (the bank)
    # do internal bookkeeping to credit the sender

  def withdraw(sender, amount):
    # ensure the sender had enough on deposit
    # do internal bookkeeping to debit the sender
    # transfer amount from myself (the bank) to sender

  def borrow(sender, amount):
    # ...

  def repay(sender, amount);
    # ...
```

In Ethereum, or in Monad, someone can write code for this Virtual Bank and upload it; then anyone can utilize it for borrowing and lending, potentially far more easily than when trying to get access to banking services in their home country.

This simple example shows the power of decentralized apps. Here are a few other benefits to call out:

-   **Open APIs / composability**: decentralized apps can be called atomically by other decentralized apps, allowing developers to build more complex functionality by stacking existing components.
-   **Transparency**: app logic is expressed purely through code, so anyone can review the logic for side effects. State is transparent and auditable; proof of reserves in DeFi is the default.
-   **Censorship-resistance and credible neutrality:** anyone can submit transactions or upload applications to a permissionless network.
-   **Global reach**: anyone with access to the internet can access crucial financial services, including unbanked/underbanked users.

---

# Why Monad: Decentralization + Performance

## Decentralization matters

A blockchain has several major components:

-   Consensus mechanism for achieving agreement on transactions to append to the ledger
-   Execution/storage system for maintaining the active state

In increasing the performance of these components, one could cut corners, for example by requiring all of the nodes to be physically close to each other (to save on the overhead of consensus), or by requiring a huge amount of RAM (to keep much or all of the state in memory), but it would be at a significant cost to decentralization.

And decentralization is the whole point!

As discussed in [Why Blockchain?](/introduction/why-blockchain.md), decentralized shared global state allows many parties to coordinate while relying on a single, shared, objective source of truth. Decentralization is key to the matter; a blockchain maintained by a small group of node operators (or in the extreme case, a single operator!) would not offer benefits such as trustlessness, credible neutrality, and censorship-resistance.

For any blockchain network, decentralization should be the principal concern. Performance improvements should not come at the expense of decentralization.

## Today's performance bottlenecks

Ethereum's current execution limits (1.25M gas per second) are set conservatively, but for several good reasons:

-   Inefficient storage access patterns
-   Single-threaded execution
-   Very limited execution budget, because consensus can't proceed without execution
-   Concerns about state growth, and the effect of state growth on future state access costs

Monad addresses these limitations through algorithmic improvements and architectural changes, pioneering several innovations that will hopefully become standard in Ethereum in the years to come. Maintaining a high degree of decentralization, while making material performance improvements, is the key consideration.

## Addressing these bottlenecks through optimization

Monad enables pipelining and other optimizations in four major areas to enable exceptional Ethereum Virtual Machine performance and materially advance the decentralization/scalability tradeoff. Subsequent pages describe these major areas of improvement:

-   [MonadBFT](/monad-arch/consensus/monad-bft.mdx)
-   [Asynchronous Execution](/monad-arch/consensus/asynchronous-execution.mdx)
-   [Parallel Execution](/monad-arch/execution/parallel-execution.md)
-   [MonadDb](/monad-arch/execution/monaddb.md)

---

# Monad Architecture

<CustomDocCardContainer>
    <CustomDocCard
        icon={<Monad />}
        link="/monad-arch/concepts"
        title="Concepts"
        description="Explaining high level themes (async io and pipelining) that recur in Monad"
    />
    <CustomDocCard
        icon={<Monad />}
        link="/monad-arch/consensus"
        title="Consensus"
        description="Algorithms for maintaining a globally distributed, decentralized validator set"
    />
    <CustomDocCard
        icon={<Monad />}
        link="/monad-arch/concepts"
        title="Execution"
        description="Algorithms for executing EVM transactions efficiently"
    />
    <CustomDocCard
        icon={<Monad />}
        link="/monad-arch/transaction-lifecycle"
        title="Transaction Lifecycle"
        description="Mapping the path of a transaction in Monad"
    />
    <CustomDocCard
        icon={<Monad />}
        link="/monad-arch/hardware-requirements"
        title="Hardware Requirements"
        description="A core principle has been high performance on reasonable hardware"
    />
    <CustomDocCard
        icon={<Monad />}
        link="/monad-arch/other-details"
        title="Other Details"
        description=""
    />
</CustomDocCardContainer>


---

# Concepts

---

# Asynchronous I/O

_Asynchronous I/O_ is a form of input/output processing that allows the CPU to continue executing concurrently while communication is in progress.

Disk and network are orders of magnitude slower than the CPU.  Rather than initiating an I/O operation and waiting for the result, the CPU can initiate the I/O operation as soon as it's known that the data will be needed, and continue executing other instructions which do not depend on the result of the I/O operation.

Some rough comparisons for illustration purposes:

<table>
    <thead>
        <tr>
            <th width="260">Device</th>
            <th width="191.33333333333331">Latency</th>
            <th>Bandwidth</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>CPU L3 Cache</td>
            <td>10 ns</td>
            <td>>400 GB/s</td>
        </tr>
        <tr>
            <td>Memory</td>
            <td>100 ns</td>
            <td>100 GB/s</td>
        </tr>
        <tr>
            <td>Disk (NVMe SSD)</td>
            <td>400 us</td>
            <td>380 MB/s</td>
        </tr>
        <tr>
            <td>Network</td>
            <td>50 - 200 ms</td>
            <td>1 Gb/s (125 MB/s)</td>
        </tr>
    </tbody>
</table>


(actual disk stats as reported by fio for random reads of size 2KB - \~190k IOPS)

Fortunately, SSD drives can perform operations concurrently, so the CPU can initiate several requests at the same time, continue executing, and then receive the results of multiple operations around the same time.

Some databases (such as lmdb / mdbx) use memory-mapped storage to read and write to disk. Unfortunately, memory-mapped storage is implemented by the kernel (mmap) and is not asynchronous, so execution is blocked while waiting for the operation to complete.

More about asynchronous I/O can be read [here](https://en.wikipedia.org/wiki/Asynchronous\_I/O).
---

# Pipelining

_Pipelining_ is a technique for implementing parallelism by dividing tasks into a series of smaller tasks which can be processed in parallel.

Pipelining is used in computer processors to increase the throughput of executing a series of instructions sequentially at the same clock rate. (There are other techniques used in processors to increase throughput as well.)  More about instruction-level parallelism (ILP) can be read [here](https://en.wikipedia.org/wiki/Instruction\_pipelining).

A simple example of pipelining:

<figure>
    <img src="/img/pipelining.png" alt="Pipelining, Laundry Day" class="center"> </img>
    <figcaption id="center">Pipelining laundry day. Top: Naive; Bottom: Pipelined. Credit: [Prof. Lois Hawkes, FSU](https://www.cs.fsu.edu/~hawkes/cda3101lects/chap6/index.html?$$$F6.1.html$$$)</figcaption>
</figure>

When doing four loads of laundry, the naive strategy is to wash, dry, fold, and store the first load of laundry before starting on the second one.  The pipelined strategy is to start washing load 2 when load 1 goes into the dryer.  Pipelining gets work done more efficiently by utilizing multiple resources simultaneously.
---

# Consensus

<CustomDocCardContainer>
    <CustomDocCard
        icon={<Monad />}
        link="/monad-arch/consensus/monad-bft"
        title="MonadBFT"
        description="High-performance consensus: pipelined 2-phase HotStuff"
    />
    <CustomDocCard
        icon={<Monad />}
        link="/monad-arch/consensus/local-mempool"
        title="Local Mempool"
        description="Policies for sharing pending transactions to leaders while minimizing bandwidth"
    />
    <CustomDocCard
        icon={<Monad />}
        link="/monad-arch/consensus/asynchronous-execution"
        title="Asynchronous Execution"
        description="Moving execution out of the hot path of consensus so it can use the full block time"
    />
    <CustomDocCard
        icon={<Monad />}
        link="/monad-arch/consensus/raptorcast"
        title="RaptorCast"
        description="Efficient block propagation of large blocks that retains BFT"
    />
    <CustomDocCard
        icon={<Monad />}
        link="/monad-arch/consensus/statesync"
        title="Statesync"
        description="Algorithms for bootstrapping a node from peers"
    />
    <CustomDocCard
        icon={<Monad />}
        link="/monad-arch/consensus/blocksync"
        title="Blocksync"
        description="Algorithms for catching up on missed traffic"
    />
</CustomDocCardContainer>


---


 bottom: asynchronous.*
</center>

## Determined ordering implies state determinism

Although execution lags consensus, the true state of the world is determined as soon as the ordering is determined. Execution is required to unveil the truth, but the truth is already determined.

It's worth noting that in Monad, like in Ethereum, it is fine for transactions in a block to "fail" in the sense that the intuitive outcome did not succeed.(For example, there could be a transaction included in a block in which Bob tries to send 10 tokens to Alice but only has 1 token in his account. The transfer 'fails' but the transaction is still valid.

The outcome of any transaction, including failure, is deterministic.



<center>
<img src={TransactionOrdering} style={{width: 800}}/>
*Example of transaction determinism even when some transactions fail*
</center>

## Finer details

### Delayed Merkle Root

As mentioned above, Monad block proposals don't include the merkle root of the state trie, since that would require execution to have already completed.

All nodes should stay in sync because they're all doing the same work. But it'd be nice to be sure! As a precaution, proposals also includes a merkle root from `D` blocks ago, allowing nodes to detect if they're diverging. `D` is a systemwide parameter (currently set in the testnet to `3`).

Delayed merkle root validity is part of block validity, so if the leader proposes a block but the delayed merkle root is wrong, the block will be rejected.

As a result of this delayed merkle root:

1. After the network comes to consensus (2/3 majority vote) on block `N` (typically upon receiving block `N+2`, which contains a QC-on-QC for block `N`), it means that the network has agreed that the official consequence of block `N-D` is a state rooted in merkle root `M`. Light clients can then query full nodes for merkle proofs of state variable values at block `N-D`.
2. Any node with an error in execution at block `N-D` will fall out of consensus starting at block `N`. This will trigger a rollback on that node to the end state of block `N-D-1`, followed by re-execution of the transactions in block `N-D` (hopefully resulting in the merkle root matching), followed by re-execution of the transactions in block `N-D+1`, `N-D+2`, etc.

Ethereum's approach uses consensus to enforce _state machine replication_ in a very strict way: after nodes come to consensus, we know that the supermajority agrees about the official ordering and the state resulting from that ordering. However, this strictness comes at great cost because interleaved execution limits execution throughput. Asynchronous execution achieves state machine replication without this limitation, and the delayed merkle root serves an additional precaution.

<center>
<img src={DelayedMerkleRoot} style={{width: 700}}/>
*Delayed merkle root*
</center>

### Balance validation at time of consensus

Leaders build blocks with a delayed view of the state.

To defend against a Denial-of-Service attack where someone submitting a bunch of transactions from an account with a zero balance, Monad nodes validate that the balance of the account is sufficient to fullfill the maximum debit in the user's account associated with in-flight transactions.

For each transaction, the max expenditure is:

```
max_expenditure = value + gas_limit * max_fee_per_gas
```

Available balance is tracked at consensus time and is decremented as transactions are included in blocks. If the available balance is insufficient to pay for `max_expenditure`, the transaction is not included in a block.

You can think of the available balance tracked by consensus as decrementing in realtime as transactions are included in blocks. Although the node's view of the full state is lagged, the available balance always reflects up-to-date expenditures.


### Speculative execution

In MonadBFT, nodes receive a proposed block `N` at slot `N`, but it is not finalized until slot `N+2`. During the intervening time, a node can still locally execute the proposed block (without the guarantee that it will become voted or finalized). This allows a few nice properties:
1. In the likely event that the proposed block is finalized, the validator node has already done the work and can immediately update its merkle root pointer to the result.
2. Transactions can be simulated (in `eth_call` or `eth_estimateGas`) against the speculative state which is likely more up-to-date.


### Transactions from newly-funded accounts

Because consensus runs slightly ahead of execution, newly-funded accounts which previously had zero balance cannot send transactions until the transfer that credits them with tokens has proceeded to the [`Verified` state](#block-states).

In practice, this means that if you send tokens from account `A` into an account `B` (which has 0 balance), then you should wait until seeing the transaction receipt (indicative that that block has reached `Finalized` stage), and then wait another 1.5 seconds.

Alternatively, depending on the nature of intended transaction from `B`, it may be possible to write a smart contract callable by `A` which combines the funding operation and whatever `B` was intending to do, requiring no delay between funding and spending.


## Block states

To summarize, Monad blocks can be in one of the following states:

1. **Proposed** - The block has been proposed by a leader but has not been voted upon. If execution is not lagging behind consensus, a node may speculatively execute the proposed block.
2. **Voted** - After a supermajority of validator nodes have voted affirmatively for a proposed block, it is considered **voted**. It has a Quorum Certificate (QC).
3. **Finalized** - A block becomes **finalized** when a supermajority of validator nodes successfully vote on the **next** block. At this point, there is no chance of a reversion.
4. **Verified** - A **verified** block has execution outputs and a state root produced by a supermajority of validator nodes. Concretely, the latest verified block will be the latest_finalized_block - execution_delay.

<center>
<img src={BlockStates} style={{width: 400}}/>
*Classification of historical blocks based on the latest **proposed** block N.*
</center>

---


# Blocksync

## Summary

Blocksync is a mechanism that nodes can use to acquire missing blocks. A block is considered missing when a Quorum Certificate is observed that references an unknown block.

Blocks can be missing from a node in one of two scenarios:

1. After the node completes statesync and its local block height is close enough to the network tip.
2. During ordinary consensus operations, the node does not receive enough RaptorCast chunks to decode the block. This can be due to packet loss or a network partition.

## Blocksync procedure

1. A single header request is made for a range of `num_blocks` blocks, starting with `last_block_id`.
2. A chain of `num_blocks` headers are received, forming a cryptographically verifiable chain back to `last_block_`.
3. For each of the `num_blocks` headers received, concurrent (up to a max concurrency factor) body requests are made containing the `body_id` included in the header.
4. Each body response is cryptographically verifiable by comparing against the corresponding header `body_id`.

![Blocksync procedure](/img/monad-arch/consensus/blocksync/blocksync_procedure.svg)

---


# Local Mempool

## Summary

Most blockchains use a global mempool with peer-to-peer gossipping for transaction propagation. This approach is not suitable for high-performance distributed consensus for a few reasons:
1. It is slow because it may involve many hops for a transaction to reach a leader, increasing time to inclusion.
2. It is wasteful on bandwidth because the gossip protocol involves many retransmissions.
3. It ignores the leader schedule which is typically known well in advance.

In Monad, there is no global mempool; instead, each validator maintains a local mempool, and RPC nodes forward transactions to the next few leaders for inclusion in their local mempool. This is much more efficient on bandwidth usage and allows transactions to be included more quickly.


## Background

A mempool is a collection of pending transactions. Many blockchain networks use a global mempool design, using peer-to-peer gossip protocols to keep roughly the same mempool state across all nodes in the network. A primary motivation of a global mempool design is that no matter who is leader, they will have access to the same set of pending transactions to include in the next block.

A global mempool is effective for low-throughput networks, where network bandwidth is typically not a bottleneck. However, at thousands of transactions per second, the gossip protocols (and especially the required retransmission at each node) can easily consume the entire network bandwidth budget. Moreover, a global mempool is wasteful since the leader schedule is typically known well in advance.


## Transaction Lifecycle in Monad

There is no global mempool in Monad. Validators maintain local mempools; RPC nodes forward transactions to upcoming leaders to ensure that those transactions are available for inclusion.

More precisely, transaction flow is as follows:
1. A transaction is submitted to the RPC process of a node (typically a full non-validator node). We'll call this node the **"owner node"** of the transaction, since it assumes responsibility for communicating the status with the user.
2. The RPC process performs some static checks on the transaction.
3. The RPC process passes the transaction to the consensus process.
4. The consensus process performs static checks and dynamic checks against local state in MonadDb, such as checking the sender's account balance and nonce.
5. If the transaction is valid, the consensus process forwards the transaction to `N` upcoming leader validator nodes. Currently, `N` is set to 3 in Monad Testnet.
6. Each of those `N` validators performs the same checks before inserting valid transactions into their local mempools.
7. When it is a leader's turn to create a proposal, it selects transactions from its local mempool.
8. The owner node of the transaction monitors for that transaction in subsequent blocks. If it doesn't see the transaction in the next `N` blocks, it will re-send to the next `N` leaders. It repeats this behavior for a total of `K` times. Currently `K` is set to 3 in Monad Testnet.

 The behavior of this transaction flow is chosen to reduce time-to-inclusion while minimizing the number of messages.


<figure>
    <img src="/img/monad-arch/consensus/local-mempool/tx_path.png" alt="Transaction path to leader" class="center"> </img>
    <figcaption id="center">Transaction path from RPC to leader (through the local mempool).</figcaption>
</figure>


## Local mempool eviction
Transactions are evicted from a validator's local mempool for the following reasons:
1. Whenever a validator finalizes a block, any replicas of transactions in that block are pruned from the local mempool.
2. Validators periodically check the validity of each transaction in the mempool and evict invalid transactions (e.g. nonces are too low, account balances are insufficient).
3. If the local mempool's size reaches a soft limit, older transactions will be evicted.

---


 validators directly message to the next leader.*
</center>

MonadBFT is a pipelined consensus mechanism that proceeds in rounds. The below description gives a high-level intuitive understanding of the protocol.

As is customary, let there be `n = 3f+1` nodes, where `f` is the max number of Byzantine nodes, i.e. `2f+1` (i.e. 2/3) of the nodes are non-Byzantine. In the discussion below, let us also treat all nodes as having equal stake weight; in practice all thresholds can be expressed in terms of stake weight rather than in node count.

-   In each round, the leader sends out a new block as well as either a QC or a TC (more on this shortly) for the previous round.
-   Each validator reviews that block for adherence to protocol and, if they agree, send signed YES votes to the leader of the next round. That leader derives a QC (quorum certificate) by aggregating (via threshold signatures) YES votes from `2f+1` validators. Note that communication in this case is _linear_: leader sends block to validators, validators send votes directly to next leader.
    -   Alternatively, if the validator does not receive a valid block within a pre-specified amount of time, it multicasts a signed timeout message to _all_ of its peers. This timeout message also includes the highest QC that the validator has observed. If any validator accumulates `2f+1` timeout messages, it assembles these (again via threshold signatures) into a TC (timeout certificate) which it then sends directly to the next leader.
-   Each validator finalizes the block proposed in round `k` upon receiving a QC for round `k+1` (i.e. in the communication from the leader of round `k+2`). Specifically:
    -   Alice, the leader of round `k`, sends a new block to everyone[^1].
    -   If `2f+1` validators vote YES on that block by sending their votes to Bob (leader of round `k+1`), then the block in `k+1` will include a QC for round `k`. However, seeing the QC for round `k` _at this point_ is not enough for Valerie the validator to know that the block in round `k` has been enshrined, because (for example) Bob could have been malicious and only sent the block to Valerie. All that Valerie can do is vote on block `k+1`, sending her votes to Charlie (leader of round `k+2`).
    -   If `2f+1` validators vote YES on block `k+1`, then Charlie publishes a QC for round `k+1` as well as a block proposal for round `k+2`. As soon as Valerie receives this block, she knows that the block from round `k` (Alice's block) is finalized.
    -   Say that Bob had acted maliciously, either by sending an invalid block proposal at round `k+1`, or by sending it to fewer than `2f+1` validators. Then at least `f+1` validators will timeout, triggering the other non-Byzantine validators to timeout, leading to at least one of the validators to produce a TC for round `k+1`. Then Charlie will publish the TC for round `k+1` in his proposal - no QC will be available for inclusion.
    -   We refer to this commitment procedure as a 2-chain commit rule, because as soon as a validator sees 2 adjacent certified blocks `B` and `B'`, they can commit `B` and all of its ancestors.

[^1]: The block is also sent with a QC or TC for round `k-1`, but this is irrelevant for this example.

<center>
<img src={MonadBFTPipelining} style={{width: 800}}/>
*Block proposals in MonadBFT are pipelined. Same diagram as the previous, but zoomed out to include one more round.*
</center>

References:

-   Maofan Yin, Dahlia Malkhi, Michael K. Reiter, Guy Golan Gueta, and Ittai Abraham. [HotStuff: BFT Consensus in the Lens of Blockchain](https://arxiv.org/abs/1803.05069), 2018.
-   Mohammad M. Jalalzai, Jianyu Niu, Chen Feng, Fangyu Gai. [Fast-HotStuff: A Fast and Resilient HotStuff Protocol](https://arxiv.org/abs/2010.11454), 2020.
-   Rati Gelashvili, Lefteris Kokoris-Kogias, Alberto Sonnino, Alexander Spiegelman, and Zhuolun Xiang. [Jolteon and ditto: Network-adaptive efficient consensus with asynchronous fallback](https://arxiv.org/pdf/2106.10362.pdf). arXiv preprint arXiv:2106.10362, 2021.
-   The Diem Team. [DiemBFT v4: State machine replication in the diem blockchain](https://developers.diem.com/papers/diem-consensus-state-machine-replication-in-the-diem-blockchain/2021-08-17.pdf). 2021.

## BLS Multi-Signatures

Certificates (QCs and TCs) can be naively implemented as a vector of ECDSA signatures on the secp256k1 curve. These certificates are explicit and easy to construct and verify. However, the size of the certificate is linear with the number of signers. It poses a limit to scaling because the certificate is included in almost every consensus message, except vote message.

Pairing-based BLS signature on the BLS12-381 curve helps with solving the scaling issue. The signatures can be incrementally aggregated into one signature. Verifying the single valid aggregated signature provides proof that the stakes associated with the public keys have all signed on the message.

BLS signature is much slower than ECDSA signature. So for performance reasons, MonadBFT adopts a blended signature scheme where BLS signature is only used on aggregatable message types (votes and timeouts). Message integrity and authenticity is still provided by ECDSA signatures.

---


# RaptorCast

## Summary

RaptorCast is a specialized multicast message delivery protocol used in MonadBFT to send block proposals from leaders to validators. Block proposals are converted into erasure-coded chunks using the Raptor code in [RFC 5053](https://datatracker.ietf.org/doc/html/rfc5053). Each chunk is sent to all validators through a two-level broadcast tree, where the first level is a single non-leader node. Each non-leader node is responsible for serving as the first-level node for a different set of chunks; the proportion of chunk assignments is equal to the validator's stake weight.

RaptorCast thus utilizes the full upload bandwidth of the entire network to propagate block proposals to all validators, while preserving Byzantine fault tolerance.

## Introduction


:::info
The technical description of RaptorCast below relates to block propagation amongst **validator** nodes participating in consensus. In particular, [block propagation to full nodes](#full-node-dissemination) is handled differently.
:::

In MonadBFT, leaders need to send block proposals to every validator. Getting block proposals from a leader to the rest of the network is one of the challenging problems in high-performance distributed consensus because block proposals are large and the network is not reliable.

Consider the following two naive approaches to addressing this problem:
1. Sending messages directly from the leader to each validator. This is the simplest approach, but it would impose very high upload bandwidth requirements for a leader because block proposals are large - for example, 10,000 transactions at 200 bytes per transaction is 2MB.

2. Sending messages from the leader to a few peers, who each re-broadcast to a few peers. This approach would reduce the upload bandwidth requirements for the leader, but it would increase maximum latency to all of the nodes, and it risks message loss if some of the peers are Byzantine and fail to forward the message.

RaptorCast is the multicast message delivery protocol that solves this problem, offering the best tradeoff between bandwidth requirements, latency, and fault-tolerance. RaptorCast was developed specifically for MonadBFT, and satisfies the following requirements.

In the below discussion, the "message" is the block proposal, and the "message originator" is the leader.

## Design requirements

- Reliable message delivery to all participating consensus nodes is guaranteed if a `2/3` supermajority of the stake weight is non-faulty (honest and online).

- Upload bandwidth requirements for a validator are linearly proportional to message size and are independent of the total number of participating validators.[^1]

[^1]: This holds when participating validators are (approximately) equally staked. In situations with (very) unevenly distributed stake weights, we need to deviate from the equal-upload property in order to maintain reliable message delivery for every possible scenario where two-thirds of the stake weight corresponds to non-faulty nodes. 

- The worst-case message propagation time is twice the worst-case one-way latency between any two nodes. In other words, the propagation of a message to all intended recipients happens within the round-trip time (RTT) between the two most distant nodes in the network.

- Messages are transmitted with a configurable amount of redundancy (chosen by the node operator). Increased redundancy mitigates packet loss and reduces message latency (recipient can decode sooner and more quickly).

## How RaptorCast works

### Erasure coding

Messages are erasure-coded by the message originator. Erasure coding means that the message is encoded into a set of chunks, and the message can be decoded from any sufficiently-large subset of the chunks.

The specific code used by RaptorCast is a variant of the Raptor code documented in [RFC 5053](https://datatracker.ietf.org/doc/html/rfc5053), with some Monad-specific modifications to 

- improve the encoding efficiency of small messages
- reduce the computational complexity of message encoding (at the cost of a slight increase in decoding complexity)

### Message and chunk distribution model

RaptorCast uses a two-level broadcast tree for each chunk. The message originator is the root of the tree, a single non-originator node lives at level 1, and every other node lives at level 2.

Each chunk of the encoded message potentially corresponds to a different broadcast tree, but the current implementation uses the same broadcast tree for contiguous ranges of the encoded message chunk space.

The following diagram illustrates this chunk distribution model:

<figure>
    <img src="/img/monad-arch/consensus/raptorcast/raptorcast_generic.png" alt="RaptorCast Broadcast Tree" class="center"> </img>
    <figcaption id="center">Generic view of the two-hop Raptorcast broadcast tree.</figcaption>
</figure>


Using a two-level broadcast tree minimizes latency for message delivery. Each level of the tree has worst-case latency of the one-way latency between any two nodes in the network (the network’s “latency diameter”), so the worst case delivery time under RaptorCast is the round-trip-time of the network.

### Fault tolerance

:::info
RaptorCast runs directly over UDP, with a single message chunk per UDP packet.
:::

Note that the broadcast tree is unidirectional. Unlike TCP, RaptorCast does not include a recovery mechanism for downstream nodes in the tree to detect packet loss and request retransmission, since this would violate latency expectations. To compensate, RaptorCast transmits the message in a redundant fashion, with a redundancy factor chosen by the message originator based on the network’s expected packet loss.

For example, under the following assumptions:

- 20% network packet loss
- maximum 33% of the network is faulty or malicious

then the message originator should expect in the worst case that (1 - 0.2) * (1 - 0.33) or ~53.6% of chunks reach the intended destination. To offset that worst case loss, the originator should send 1 / 0.536 - 1 or roughly 87% *additional* chunks. 

The default [MTU](https://en.wikipedia.org/wiki/Maximum_transmission_unit) used is 1480 bytes. After subtracting RaptorCast header overhead for the default Merkle tree depth of 6, this leaves 1220 bytes per packet for an encoded Raptor payload.  A 2.000.000 byte block maps to 2e6 / 1220 = 1640 source chunks. Using the current redundancy factor of 3, 4920 encoded chunk will then be distributed to other validators by proportionate stake weight. 

If there are 100 validators, those 4920 encoded chunks will be divided into 99 (the originator is excluded) distinct chunk ranges and the leader will initiate a broadcast tree for each validator corresponding to its unique chunk range (and payload). If the validators had equal stake, each would receive 4920 / 99 = 50 chunks in contiguous ranges.

<figure>
    <img src="/img/monad-arch/consensus/raptorcast/raptorcast_expansion.png" alt="RaptorCast encoding and redundancy" class="center"> </img>
    <figcaption id="center">A 2 MB block is split into chunks, expanded and disseminated.</figcaption>
</figure>

Note that the two-stage distribution model allows participating consensus nodes to receive a copy of a message even if direct network connectivity with the message originator is intermittently or entirely faulty.

<figure>
    <img src="/img/monad-arch/consensus/raptorcast/raptorcast_monad.png" alt="Block proposal" class="center"> </img>
    <figcaption id="center">RaptorCast used to send erasure-encoded chunks from a leader to each validator.</figcaption>
</figure>

The message originator (leader) typically[^2] distributes generated chunks to the first-hop recipients according to stake weight. For example:

* Validator 1 has stake 1
* Validator 2 has stake 2
* Validator 3 has stake 3
* Validator 4 has stake 4

When Validator 1 is the leader, they will send:
* 2 / (2 + 3 + 4) of generated chunks to validator 2
* 3 / (2 + 3 + 4) of generated chunks to validator 3
* 4 / (2 + 3 + 4) of generated chunks to validator 4

The leader _currently_ sends chunks in contiguous ranges but development work is currently being done to enable dissemination at a more granular level. With the new algorithm, individual or much smaller sets of chunks would be sent randomly to first-hop validators without replacement, weighted by stake. This approach produces better utilization of the network as all validators can start processing chunks as they arrive and send for redistribution (start the second-hop).

[^2]: The pure stake-weighted distribution scheme can break down when the number of required chunks is sufficiently small, e.g. 12 chunks distributed to 100 validators. This corner case is actively being addressed.

### Chunk transport integrity

The originator signs every encoded chunk, so intermediate nodes (level one) in the broadcast tree can verify the integrity of an encoded chunk before forwarding it.

Furthermore, the number of source chunks `K` is encoded in the message. For given `K`, the recipient currently accepts encoded chunks in the range of 0 to `7 * K - 1`. This gives the originator sufficient freedom to specify a high degree of redundancy (up to 7), while also limiting the potential for network spam by a rogue validator.

To amortize the cost of generating and verifying these signatures over many chunks, RaptorCast aggregates contiguous ranges of encoded message chunks in variable-depth Merkle trees, and produces a single signature for every Merkle tree root.

### Other uses of RaptorCast

RaptorCast is not only used for broadcasting a block (in chunks) from the leader. Transaction forwarding, e.g. from a full node to the next three validator hosts, is also performed via RaptorCast, benefiting from its properties of speed and robustness. In this context, only one hop is required - the receiver should not rebroadcast.

### Full node dissemination

Currently, validator nodes configure a list of downstream full nodes. A given validator will send every valid chunk it originates or receives to every full node in this list. 

<figure>
    <img src="/img/monad-arch/consensus/raptorcast/raptorcast_full_node.png" alt="Dissemination to full nodes" class="center"> </img>
    <figcaption id="center">Each node in the broadcast tree disseminates all received (or produced) chunks to configured full nodes.</figcaption>
</figure>

Design and implementation for full node peer discovery and more efficient and scalable dissemination is underway.
---


# Statesync

## Summary

Statesync is the process for synchronizing state to a target block close to the current tip. A synchronizing node ("client") requests data from other up-to-date validators ("servers") to help it progress from its current view to its target view; the servers rely on metadata in MonadDb to efficiently respond to the request.

Since the current tip is a moving target, following completion of statesync, the client makes another statesync request to get closer to the current tip, or replays queued blocks if within striking distance.

## Approach

Statesync is the process of synchronizing state stored in [MonadDb](/monad-arch/execution/monaddb) to a target block close to the current tip.

The current tip is a moving target, so as statesync is running, the syncing node stores new blocks beyond the target block and, upon completion of statesync, replays these additional blocks through normal execution to catch up to the current tip. The target block may be updated several times during this process.

Statesync follows a client-server model, where the statesync requester is the client and the validator node servicing a statesync request is the server.

## Data included in statesync

MonadDb stores a variety of data relating to the execution of blocks. However, only a subset is required for full participation in the active set and thus included in statesync:

- accounts, including balances, code, and storage
- the last 256 block headers (to verify correctness)

In an effort to evenly distribute load, each of the aforementioned is spliced into chunks. The client assigns each chunk to a server who remains the peer for that chunk until synchronization is complete.

Servers are randomly selected from the list of available peers. The client maintains a certain number of sessions up to a configured maximum. In the event that servers are unresponsive, the client’s statesync request will timeout and request from a different server.

## Versioning and verification
For efficiency, the client requests state from least- to most-recently updated, converging on the tip near the end of the process. Servers serve diffs relative to the client's latest block.


![statesync_requests](/img/monad-arch/consensus/statesync/statesync_requests.png)

In the example above, the statesync client makes three consecutive requests to the statesync server assigned to prefix p. For each request, there are five parameters specified:

- `prefix` - the prefix of the Merkle Patricia Trie
- `i` - the start block number
- `j` - the end block number
- `target` - the target block number
- `last_target` - last target block number, this is used to deduce deletions to send

Because there may be multiple rounds of statesync (as statesync occurs, the chain is progressing and the target block may need to adjust), `j` is buffered by some offset B from the target block to avoid retransmitting most recently used nodes in the MPT.  When `i` and `target` block are sufficiently close, as in the last round above, the statesync client will request `j = target`.

At this point, if `target` is less than 500 blocks from the tip of the chain, statesync is concluded and the state root will be validated and then the satisfied statesync client will begin [blocksync](./blocksync). If `target` is greater than 500 blocks from the tip of the chain, a new round of statesync will begin.

During block execution, the server stores the version alongside node contents. As such, upon receipt of a statesync request, the server is able to quickly narrow down the relevant subtrie and submit read requests, which are embarrassingly parallel. 

## Trust assumptions

Statesync clients trust that the requested data (including state root and parent hash) from statesync servers is correct. This is currently sampled randomly from the validator set according to stakeweight, but clients can optionally whitelist specific known providers as statesync servers.

The current implementation validates the data transmitted when the whole transfer is complete by comparing the state root. Because the work is split between multiple servers, a single server sending invalid data can cause a state root mismatch, without attribution to the faulty server. The only recourse in this situation is to retry the whole transfer, giving the faulty server an opportunity to fail the operation again.

Changes are currently being implemented to verify the data transmitted on a per-server basis. In the event of a faulty server sending invalid data, the statesync client can discard and retry *only* the affected prefix. Further, it can identify the faulty server, log the error and potentially blacklist it from subsequent requests.

---

# Execution

<CustomDocCardContainer>
    <CustomDocCard
        icon={<Monad />}
        link="/monad-arch/execution/parallel-execution"
        title="Parallel Execution"
        description="Optimistic parallel execution"
    />
    <CustomDocCard
        icon={<Monad />}
        link="/monad-arch/execution/monaddb"
        title="MonadDb"
        description="Custom database for storing the Ethereum Merkle Patricia Trie natively on SSD"
    />
</CustomDocCardContainer>


---


# MonadDb

## Summary

MonadDb is a critical component in Monad for maintaining full Ethereum compatibility while delivering high performance. It is a custom-built key-value database designed for storing authenticated blockchain data. MonadDb, specifically, is optimized for efficiently storing Merkle Patricia Trie nodes on disk.

## Merkle Patricia Trie Structured Database

Most Ethereum clients use generic key-value databases that are implemented as either B-Tree (e.g. [LMDB](https://www.symas.com/lmdb)) or LSM-Tree (e.g. [LevelDB,](https://github.com/google/leveldb) [RocksDB](https://rocksdb.org/)) data structures. However Ethereum uses the [Merkle Patricia Trie](https://ethereum.org/en/developers/docs/data-structures-and-encoding/patricia-merkle-trie/) (MPT) data structure for storing state and other authenticated fields like receipts and transactions. This results in a suboptimal solution where one data structure is embedded into another data structure. MonadDb implements a [Patricia Trie](https://en.wikipedia.org/wiki/Radix_tree) (a specific variant of radix tree) data structure natively, both on-disk and in-memory. Despite the opinionated design, MonadDb is a flexible key-value store capable of storing any type of data. For instance, MonadDb is also used to store block headers and payloads for Monad.

## Asynchronous IO

Monad executes multiple transactions in [parallel](/monad-arch/execution/parallel-execution). In order to enable this, reads should not block continued operation, and this goal motivates [asynchronous I/O](/monad-arch/concepts/asynchronous-io) (async I/O) for the database. The above-mentioned key-value databases lack proper async I/O support (although there are some efforts to improve in this area). MonadDb fully utilizes the latest kernel support for async I/O (on Linux this is [io_uring](https://unixism.net/loti/index.html)). This avoids spawning a large number of kernel threads to handle pending I/O requests in an attempt to perform work asynchronously.

## Filesystem bypass

Modern filesystems provide a convenient abstraction for applications, but introduce overhead when building high-throughput I/O software. These often hidden costs include block allocation, fragmentation, read/write amplification, and metadata management. The abstraction of files and a set of system calls allows applications to interact with the file data as if it were stored contiguously. The complexity of managing exact physical disk locations is abstracted away from the applications (and their developers). However, the actual content on disk might be fragmented into multiple non-contiguous pieces.  Accessing or writing to such a file usually involves more than one simple I/O operation.

To minimize overhead, MonadDb provides operators the option to bypass the filesystem. MonadDb implements its own indexing system based on the Patricia trie data structure, eliminating filesystem dependencies. Users have the flexibility to operate MonadDb on either regular files or block devices. For optimal performance, it is recommended to run MonadDb directly on block devices. This approach avoids all filesystem-related overhead, allowing MonadDb to fully unlock SSD performance.

## Concurrency Control

The Monad blockchain consists of multiple clients, each interacting with the database as either reader or writer. To support this functionality, MonadDb must efficiently synchronize between a single writer (execution) and multiple readers (consensus and RPC).

MonadDb implements a persistent (or immutable) Patricia trie. When a branch in the trie is updated, new versions of the nodes on that branch are created, and the previous version of the trie is preserved. This approach facilitates versioning within the database and significantly simplifies synchronization between readers and the writer. It ensures that all reads are accurate and consistent while guaranteeing that writes are both complete and atomic from the perspective of the readers.

## Write Performance on Modern SSD

The usage of a persistent data structure also allows us to perform sequential writes, which offers better performance than random writes on modern SSDs. Modern SSD garbage collection occurs at the block level. On sequential writes, an entire block gets filled before the next one, which dramatically simplifies garbage collection. Garbage collection is much more expensive for random writes. Sequential writes also distribute data more efficiently, thereby reducing write amplification and increasing SSD longevity.

## Compaction

As historical versions accumulate, the amount of data written to disk will grow. Given the limited disk capacity it operates on, it is impossible to retain complete historical records. MonadDb stores recent versions of blockchain data and state and dynamically adjusts the history length based on available disk space. As newer versions are stored and older versions are pruned, the underlying storage space becomes fragmented. To address this, MonadDb performs compaction inline with updates, consolidating active data and releasing unused storage for recycling. This reduces fragmentation while maintaining performance and data integrity.
---

# Parallel Execution

## Summary

Monad executes transactions in parallel.  While at first it might seem like this implies different execution semantics than exist in Ethereum, it actually does not.  Monad blocks are the same as Ethereum blocks - a linearly ordered set of transactions.  The result of executing the transactions in a block is identical between Monad and Ethereum.


## Optimistic Execution

At a base level, Monad uses optimistic execution. This means that Monad will start executing transactions before earlier transactions in the block have completed.  Sometimes (but not always) this results in incorrect execution.

Consider two transactions (in this order in the block):

1. Transaction 1 reads and updates the balance of account A (for example, it receives a transfer from account B).
2. Transaction 2 also reads and updates the balance of account A (for example, it makes a transfer to account C).

If these transactions are run in parallel and transaction 2 starts running before transaction 1 has completed, then the balance it reads for account A may be different than if they were run sequentially.  This could result in incorrect execution.

The way optimistic execution solves this is by tracking the inputs used while executing transaction 2 and comparing them to the outputs of transaction 1.  If they differ, we have detected that transaction 2 used incorrect data while executing and it needs to be executed again with the correct data.

While Monad executes transactions in parallel, the updated state for each transaction is "merged" sequentially in order to check the condition mentioned above.

Related computer science topics are [optimistic concurrency control](https://en.wikipedia.org/wiki/Optimistic\_concurrency\_control) (OCC) and [software transactional memory](https://en.wikipedia.org/wiki/Software\_transactional\_memory) (STM).


## Optimistic Execution Implications

In a naïve implementation of optimistic execution, one doesn't detect that a transaction needs to be executed again until earlier transactions in the block have completed.  At that time, the state updates for all the earlier transactions have been merged so it's not possible for the transaction to fail due to optimistic execution a second time.

There are steps in executing a transaction that do not depend on state. An example is signature recovery, which is an expensive computation.  This work does not need to be repeated when executing the transaction again.

Furthermore, when executing a transaction again due to failure to merge, often the account(s) and storage accessed will not change.  This state is still be cached in memory, so again this is expensive work that does not need to be repeated.


## Scheduling

A naïve implementation of optimistic execution will try to start executing the next transaction when the processor has available resources.  There may be long "chains" of transactions which depend on each other in the block.  Executing these transactions in parallel would result in a significant number of failures.

Determining dependencies between transactions ahead of time allows Monad to avoid this wasted effort by only scheduling transactions for execution when prerequisite transactions have completed.  Monad has a static code analyzer that tries to make such predictions.  In a good case Monad can predict many dependencies ahead of time; in the worst case Monad falls back to the naïve implementation.


## Further Work

There are other opportunities to avoid re-executing transactions which are still being explored.
---

# Hardware Requirements

The following hardware requirements are expected to run a Monad full node:

* CPU: 16 core CPU with 4.5 ghz+ base clock speed, e.g. AMD Ryzen 7950X
* Memory: 32 GB RAM
* Storage: 2 x 2 TB NVMe SSDs (one dedicated to MonadDB)
* Bandwidth: 100 Mb/s
---

# Other Details

## Accounts

Accounts in Monad are identical to [Ethereum accounts](https://ethereum.org/en/developers/docs/accounts/).  Accounts use the same address space (20-byte addresses using ECDSA).  As in Ethereum, there are Externally-Owned Accounts (EOAs) and Contract Accounts.



## Transactions

The transaction format in Monad [matches Ethereum](https://ethereum.org/en/developers/docs/transactions/), i.e. it complies with [EIP-2718](https://eips.ethereum.org/EIPS/eip-2718), and transactions are encoded with [RLP](https://ethereum.org/en/developers/docs/data-structures-and-encoding/rlp/).

Access lists ([EIP-2930](https://eips.ethereum.org/EIPS/eip-2930)) are supported but not required.



## Linearity of Blocks and Transactions

Blocks are still linear, as are transactions within a block.  Parallelism is utilized strictly for efficiency; it never affects the true outcome or end state of a series of transactions.



## Gas

[Gas](https://ethereum.org/en/developers/docs/gas/) (perhaps more clearly named "compute units") functions as it does in Ethereum, i.e. each opcode costs a certain amount of gas. Gas costs per opcode are identical to Ethereum in Monad, although this is subject to change in the future.

When a user submits a transaction, they include a gas limit (a max number of units of gas that this function call can consume before erroring) as well as a gas price (a bid, in units of native token, per unit of gas).

Leaders in the default Monad client use a priority gas auction (PGA) to order transactions, i.e. they order transactions by descending (effective) priority fee per gas. Currently, baseFeePerGas on Monad is hardcoded to 50 gwei.  As a consequence of Monad’s [delayed execution](./consensus/asynchronous-execution), block gas limits and base fees are both applied to the gasLimit parameter of transactions rather than the actual gas used by execution.  As with EIP-1559, base fees are burned.

A finalized transaction fee mechanism is still under active development.
---

# Transaction Lifecycle in Monad

## Transaction Submission

The lifecycle of a transaction starts with a user preparing a signed transaction and submitting it to an RPC node.

Transactions are typically prepared by an application frontend, then presented to the user's wallet for signing. Most wallets make an `eth_estimateGas` RPC call to populate the gas **limit** for this transaction, although the user can also override this in their wallet. The user is also typically asked to choose a gas **price** for the transaction, which is a number of NativeTokens per unit of gas.

After the user approves the signing in their wallet, the signed transaction is submitted to an RPC node using the `eth_sendTransaction` or `eth_sendRawTransaction` API call.

## Mempool Propagation

As described in [Local Mempool](/monad-arch/consensus/local-mempool.md):

The RPC node performs validity checks:
- signature verification
- nonce not too low
- gas limit below block gas limit

before forwarding the pending transaction to the next `N` leaders.

Each of those leaders replicate those validity checks before adding the pending transaction to their local mempool.

If the transaction isn't included in any of the blocks proposed by those leaders, the RPC node repeats this process, sending to the next `N` leaders. The process is repeated up to `K` times.

## Block Inclusion

 Pending transactions are included in a block only if further dynamic checks pass:
 - account balance is sufficient to pay for gas (see: [Balance Validation at Time of Consensus](/monad-arch/consensus/asynchronous-execution.mdx#balance-validation-at-time-of-consensus))
 - nonce is contiguous
 - there is space in the block and the leader has chosen to include this transaction

## Block Propagation

Blocks are propagated through the network as discussed in [MonadBFT](/monad-arch/consensus/monad-bft.mdx), using the [RaptorCast](/monad-arch/consensus/raptorcast.md) messaging protocol for outbound messages from the leader.

Under MonadBFT, a block progresses from the Proposed phase to the Voted phase (after 1 block) and then to the Finalized phase (after 2 blocks).

Once the block is Finalized, the transaction has officially "occurred" in the history of the blockchain. Since its order is determined, its truth value (i.e., whether it succeeds or fails, and what the outcome is immediately after that execution) is determined.

## Local Execution

As soon as a node finalizes a block, it begins executing the transactions from that block. For efficiency reasons, transactions are executed [optimistically in parallel](/monad-arch/execution/parallel-execution.md), but it is as if the transactions were executed serially, since results are always committed in the original order.

## Querying the Outcome

The user can query the result of the transaction by calling `eth_getTransactionByHash` or `eth_getTransactionReceipt` on any RPC node. The RPC node will return as soon as execution completes locally on the node.

---


# Official Links

| What                     | Where                                                                                  |
| ------------------------ | -------------------------------------------------------------------------------------- |
| Website                  | [https://monad.xyz](https://monad.xyz)                                                 |
| Testnet Hub              | [https://testnet.monad.xyz](https://testnet.monad.xyz)                                 |
| Ecosystem Directory      | [https://monad.xyz/ecosystem](https://monad.xyz/ecosystem)                             |
| Testnet Block Explorer   | [https://testnet.monadexplorer.com](https://testnet.monadexplorer.com)                 |
| X                        | [https://x.com/monad\_xyz](https://x.com/monad\_xyz)                                   |
| Monad Eco X              | [https://x.com/monad\_eco](https://x.com/monad\_eco)                                   |
| DevNads on X             | [https://x.com/monad\_dev](https://x.com/monad\_dev)                                   |
| The Pipeline on X        | [https://x.com/pipeline\_xyz](https://x.com/pipeline\_xyz)                             |
| Substack                 | [https://monadxyz.substack.com](https://monadxyz.substack.com)                         |
| Discord                  | [https://discord.gg/monad](https://discord.gg/monad)                                   |
| Developer Discord        | [https://discord.gg/monaddev](https://discord.gg/monaddev)                             |
| Monad Foundation Jobs    | [https://jobs.ashbyhq.com/monad.foundation](https://jobs.ashbyhq.com/monad.foundation) |




---


# RPC Overview

Monad is a fully EVM-equivalent Layer 1 blockchain. Getting started building on Monad should feel familiar to anyone who has previously developed for Ethereum. 

Monad supports a [JSON-RPC](https://www.jsonrpc.org/specification) interface for interacting with the blockchain. 

Monad is in active development, currently operating in private devnet. Details on public RPC endpoints will be available in the coming months.

---

# Reference

Monad is a fully EVM-equivalent Layer 1 blockchain. Getting started building on Monad should feel familiar to anyone who has previously developed for Ethereum.

## RPC

Monad supports a [JSON-RPC](https://www.jsonrpc.org/specification) interface for interacting with the blockchain. 

* [RPC Error Codes](rpc-error-codes.md)
* [RPC Differences](rpc-differences.md)
* [RPC API Reference](reference/json-rpc)
---


# JSON-RPC API

This section provides an interactive reference for the Monad's JSON-RPC API.

View the JSON-RPC API methods by selecting a method in the left sidebar. You can test the methods directly in the page using the API playground, with pre-configured examples or custom parameters.
---


# RPC Behaviorial Differences

Monad aims to match the RPC behavior as close as possible to Geth’s behavior, but due to fundamental architectural differences, there are some discrepancies listed below.

1. `eth_getLogs` currently have a maximum block range of 100 blocks. Because Monad blocks are much larger than Ethereum blocks, we recommend using small block ranges (e.g. 1-10 blocks) for optimal performance. When requesting more, requests can take a long time to be fulfilled and may timeout.
2. `eth_sendRawTransaction` may not immediately reject transactions with a nonce gap or insufficient gas balance. Due to asynchronous execution, the RPC server may not have the latest world view. Thus these transactions are allowed as they may become valid transactions during block creation.
3. `eth_call` and `eth_estimateGas` do not accept EIP-4844 transaction type yet. This is temporary.
4. `eth_maxPriorityFeePerGas` currently returns a hardcoded suggested fee of 2 gwei. This is temporary.
5. `eth_feeHistory` currently also returns default values as the base fee in testnet is hardcoded. This is temporary.
6. Websockets are not yet supported. This is temporary.
---


# RPC Error Codes

Monad supports a [JSON-RPC](https://www.jsonrpc.org/specification) interface for interacting with the blockchain.
 Monad JSON-RPC aims to be equivalent to Ethereum JSON-RPC, however some error codes slightly deviate due to lack of standardization across Ethereum clients.

## Monad Error Codes Reference

<table>
   <thead>
      <tr>
         <th width="218">Error Code</th>
         <th width="264.3333333333333">Message</th>
         <th width="255">Explanation</th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td><strong>-32601</strong></td>
         <td>Parse error</td>
         <td>Unable to parse JSON-RPC request</td>
      </tr>
      <tr>
         <td><strong>-32601</strong></td>
         <td>Invalid request</td>
         <td>Invalid request such as request that exceeds size limit</td>
      </tr>
      <tr>
         <td><strong>-32601</strong></td>
         <td>Method not found</td>
         <td>Method that is not part of the JSON-RPC spec</td>
      </tr>
      <tr>
         <td><strong>-32601</strong></td>
         <td>Method not supported</td>
         <td>Method that is part of the JSON-RPC spec but not yet supported by Monad</td>
      </tr>
      <tr>
         <td><strong>-32602</strong></td>
         <td>Invalid block range</td>
         <td>eth_getLogs filter range is limited to 100 blocks</td>
      </tr>
      <tr>
         <td><strong>-32602</strong></td>
         <td>Invalid params</td>
         <td>Request contains incorrect parameters associated to the particular method</td>
      </tr>
      <tr>
         <td><strong>-32603</strong></td>
         <td>Internal error</td>
         <td>Request that cannot be fulfilled due to internal error</td>
      </tr>
      <tr>
         <td><strong>-32603</strong></td>
         <td>Execution reverted</td>
         <td>eth_call and eth_estimateGas simulates transaction to revert</td>
      </tr>
      <tr>
         <td><strong>-32603</strong></td>
         <td>Transaction decoding error</td>
         <td>Request contains raw transaction that cannot be decoded</td>
      </tr>
   </tbody>
</table>


---

# Tooling and Infrastructure

Many leading Ethereum developer tools support Monad Testnet. The enclosing pages survey each category of tools and attempt a feature comparison.

* [Account Abstraction](account-abstraction/README.md)
  * [AA Infrastructure](account-abstraction/infra-providers.md)
  * [Wallet-as-a-Service](account-abstraction/wallet-providers.md)
* [Block Explorers](block-explorers.md)
* [Cross-Chain](cross-chain.md)
* [Indexers](indexers/README.md)
  * [Common Data](indexers/common-data.md)
  * [Indexing Frameworks](indexers/indexing-frameworks.md)
* [Oracles](oracles.md)
* [RPC Providers](rpc-providers.md)
* [Toolkits](toolkits.md)
* [Wallets](wallets.md)

---

# Account Abstraction

Account Abstraction means utilizing smart contract wallets in place of EOAs. Subproblems include:
1. how UserOperations get submitted (and paid for)
2. the authentication mechanism
3. the SDK for interacting with the wallet
4. the UI of the wallet

Roughly speaking, subproblem (1) is handled by bundlers and paymasters, while subproblems (2) through (4) are handled by Wallet-as-a-Service (WaaS) providers.

## UserOperation Submission

Under ERC-4337, transactions (actually UserOperations) get submitted via a bundler and may be paid for by a paymaster.

See **[AA Infra Providers](infra-providers.md)** for a list of providers supporting the Monad Testnet.


## Wallet Mechanics

Wallet-as-a-Service (WaaS) providers offer modular components that allow application developers to customize how transactions will be signed and submitted.

See **[AA Wallet Providers](wallet-providers.md)** for a list of WaaS providers supporting the Monad Testnet.


---

# AA Infra Providers

AA Providers provide bundler and paymaster services, enabling features like sponsored transactions or payment via custom tokens.

## Definitions
| Service | Description |
| --------- | --------- |
| Bundler   | Operates a custom mempool for UserOperations; simulates and assembles bundles of UserOperations |
| Paymaster | Enables sponsored transactions; enables users to pay for gas with a custom token                |

## Provider Summary

These providers support Monad Testnet:

<table>
  <thead>
    <tr>
      <th>Provider</th>
      <th>Docs</th>
      <th>Supported services</th>
      <th>How to get started</th>
    </tr>
  </thead>
  <tbody>   
    
    <tr>
      <td>[Alchemy](https://www.alchemy.com/smart-wallets)</td>
      <td>[Docs](https://accountkit.alchemy.com/)</td>
      <td>[Gas Manager](https://docs.alchemy.com/docs/gas-manager-services) (aka Paymaster)<br/>
          [Bundler](https://docs.alchemy.com/docs/bundler-services)</td>
      <td>[Dashboard](https://dashboard.alchemy.com/accounts?a=smart-wallets)</td>
    </tr>

    
    <tr>
      <td>[Biconomy](https://biconomy.io)</td>
      <td>[Docs](https://docs.biconomy.io)</td>
      <td>[Paymaster](https://docs.biconomy.io/infra/paymaster/integration)<br/>
        [Bundler](https://docs.biconomy.io/infra/bundler/integration)</td>
      <td>[Quickstart](https://docs.biconomy.io/tutorials/simple)</td>
    </tr>

    
    <tr>
      <td>[FastLane](https://www.fastlane.xyz/)</td>
      <td>[Docs](https://docs.shmonad.xyz/)</td>
      <td>[Paymaster and Bundler](https://github.com/FastLane-Labs/4337-bundler-paymaster-script/tree/main)</td>
      <td>[Dashboard](https://shmonad.xyz/)</td>
    </tr>

        
    <tr>
      <td>[Gelato Relay](https://docs.gelato.network/web3-services/relay)</td>
      <td>[Docs](https://docs.gelato.network/web3-services/relay)</td>
      <td>Gelato [Relay](https://docs.gelato.network/web3-services/relay)</td>
      <td>[Quickstart](https://docs.gelato.network/web3-services/relay/quick-start)</td>
    </tr>

    
    <tr>
      <td>[Openfort](https://openfort.io/)</td>
      <td>[Docs](https://www.openfort.io/docs)</td>
      <td>[Paymaster and Bundler](https://www.openfort.io/docs/overview/infrastructure)</td>
      <td>[Quickstart](https://www.openfort.io/docs/overview/start)</td>
    </tr>

    
    <tr>
      <td>[Pimlico](https://pimlico.io/)</td>
      <td>[Docs](https://docs.pimlico.io/)</td>
      <td>[Paymaster](https://docs.pimlico.io/infra/paymaster)<br/>
        [Bundler](https://docs.pimlico.io/infra/bundler)</td>
      <td>[Tutorial](https://docs.pimlico.io/permissionless/tutorial/tutorial-1)</td>
    </tr>

    
    <tr>
      <td>[thirdweb](https://thirdweb.com/)</td>
      <td>[Docs](https://portal.thirdweb.com/)</td>
      <td>[Paymaster and Bundler](https://portal.thirdweb.com/connect/account-abstraction/infrastructure)</td>
      <td>[Quickstart](https://portal.thirdweb.com/typescript/v5/account-abstraction/get-started)</td>
    </tr>

    
    <tr>
      <td>[ZeroDev](https://zerodev.app/)</td>
      <td>[Docs](https://docs.zerodev.app/)</td>
      <td>[Meta AA infrastructure](https://docs.zerodev.app/meta-infra/intro) for bundlers and paymasters</td>
      <td>[Dashboard](https://dashboard.zerodev.app/)</td>
    </tr>
  </tbody>
</table>


## Provider Details

### Alchemy

Alchemy powers the [#1 most used](https://www.bundlebear.com/factories/all) smart accounts today with account abstraction that eliminates gas fees and signing for users. Their accounts support ERC-4337, EIP-7702, and ERC-6900, a modular account standard co-authored with the Ethereum Foundation, Circle, and Trust Wallet.

To get started, sign up for an [Alchemy account](https://dashboard.alchemy.com/accounts), visit the [documentation](https://accountkit.alchemy.com/), follow the [quickstart](https://accountkit.alchemy.com/react/quickstart) guide. To learn more, check out their [smart wallets](https://www.alchemy.com/smart-wallets) and demo [here](https://demo.alchemy.com/).

Supported Networks
- Monad Testnet


### Biconomy

[Biconomy](https://biconomy.io) is the most comprehensive smart account and execution infrastructure platform that enables seamless, user-friendly experiences across single or multiple chains. With Biconomy, developers can build superior onchain UX through gas abstraction, sessions, batching, and one-click signatures for complex actions on any number of networks.

To get started, visit the [documentation](https://docs.biconomy.io/) or follow the [simple tutorial](https://docs.biconomy.io/tutorials/simple).

Supported Networks
- Monad Testnet


### FastLane

[FastLane](https://www.fastlane.xyz/) is an MEV protocol for validators + apps with an integrated 4337 bundler, an on-chain task scheduler, and the first holistic LST.

To get started, vist the [shMonad](https://docs.shmonad.xyz/) Documentation or try the shMonad bundler using the following example [project](https://github.com/FastLane-Labs/4337-bundler-paymaster-script/tree/main).

Supported Networks
- Monad Testnet


### Gelato Relay

[Gelato Relay](https://docs.gelato.network/web3-services/relay) allows you to sponsor gas cost on behalf of the user, enable gasless cross-chain transactions and Account Abstraction.




To get started, visit the [documentation](https://docs.gelato.network/web3-services/relay/what-is-relaying) or follow the [quickstart](https://docs.gelato.network/web3-services/relay/quick-start) guide.

Supported Networks:
- Monad Testnet


### Openfort

[Openfort](https://openfort.io) is a developer platform that helps projects onboard and and activates wallets. It does so by creating wallets with it’s SSS and passkeys,sending transactions via sponsored paymasters and session keys or directly using backend wallets for automated onchain actions.

To get started, visit the [documentation](https://www.openfort.io/docs/overview/start) or follow the [quickstart](https://www.openfort.io/docs/guides/react) guide.

Supported Networks
- Monad Testnet


### Pimlico

[Pimlico](https://pimlico.io/) is the world's most advanced ERC-4337 account abstraction infrastructure platform. Pimlico provides a suite of tools and services to help you build, deploy, and manage smart accounts on Ethereum and other EVM-compatible chains.

To get started, visit the [documentation](https://docs.pimlico.io/) or follow the [quickstart](https://docs.pimlico.io/permissionless/tutorial/tutorial-1) guide.

Supported Networks
- Monad Testnet


### thirdweb

[thirdweb](https://portal.thirdweb.com/connect/account-abstraction/overview) offers a complete platform to leverage account abstraction.

Remove the clunky user experience of requiring gas & signatures for every onchain action:

* Abstract away gas
* Pre-audited account factory contracts
* Built-in infra:
* Sponsorship policies

To get started:

1. Sign up for a [free thirdweb account](https://thirdweb.com/team)
2. Visit [Account Abstraction Documentation](https://portal.thirdweb.com/connect/account-abstraction/how-it-works) and [Account Abstraction Playground](https://playground.thirdweb.com/connect/account-abstraction/connect)

Supported Networks
- Monad Testnet


### Zerodev

[ZeroDev](https://zerodev.app) is the most powerful smart account development platform. With ZeroDev, you can build Web3 experiences without gas, confirmations, seed phrases, and bridging.

To get started, visit the [documentation](https://docs.zerodev.app/) or follow the [quickstart](https://docs.zerodev.app/sdk/getting-started/quickstart) guide.

Supported Networks
- Monad Testnet
---

# AA Wallet Providers

Under ERC-4337, smart wallets perform authentication (signature verification) inside of a smart contract. 
Depending on the signature scheme, signing may be done locally (on the user's computer) or in a remote environment (e.g. TEEs).

Full-stack Wallet-as-a-Service (WaaS) providers offer a smart contract wallet, the associated infrastructure for signing UserOperations, an SDK for interacting with the wallet, and UI components for sign-in and for authorizing transactions.

## Authentication Features
| Features | Description |
| --------- | --------- |
| Passkey sign-in | Authentication with [WebAuthn](https://developer.mozilla.org/en-US/docs/Web/API/Web_Authentication_API) (passkey) |
| Social sign-in | Authentication with social accounts (google, X, etc) |
| Email sign-in | Authentication with OTP via email |
| SMS sign-in | Authentication with OTP via SMS |

## Key Management Features
| Features  | Description |
| --------- | ----------- |
| MPC | Multi-party computation |
| SSS | Shamir's Secret Sharing |
| TEE | Storage of private keys in a cloud-based Trusted Execution Environment, like AWS Nitro Enclaves |
| TSS | Threshold Signature Scheme |
| Embedded wallet | A wallet interface local to a website or mobile app, utilizing browser session keys for signing |
| Server-delegated actions | Allow app to request permission to sign on the user's behalf |
| Session keys | Scoped keys that grant access only for specific apps, useful for bots/AI agents |

## Provider Summary

These WaaS providers support the Monad Testnet:

<table>
  <thead>
    <tr>
      <th>Provider</th>
      <th>Docs</th>
      <th>Supported services</th>
      <th>Security Method</th>
      <th>How to get started</th>
    </tr>
  </thead>
  <tbody>   
    
    <tr>
      <td>[Alchemy](https://www.alchemy.com/smart-wallets)</td>
      <td>[Docs](https://accountkit.alchemy.com/)</td>
      <td>[Embedded wallets](https://accountkit.alchemy.com/react/quickstart)<br/>
      Auth: [passkey](https://accountkit.alchemy.com/signer/authentication/passkey-signup), [social](https://accountkit.alchemy.com/signer/authentication/social-login), [email](https://accountkit.alchemy.com/signer/authentication/email-otp) sign-in</td>
      <td></td>
      <td>[Quickstart](https://accountkit.alchemy.com/react/quickstart)</td>
    </tr>

    
    <tr>
      <td>[Biconomy](https://biconomy.io)</td>
      <td>[Docs](https://docs.biconomy.io)</td>
      <td>[Nexus: Smartest & most gas-efficient smart account](https://docs.biconomy.io/overview)<br/>
      Auth: [passkey](https://docs.biconomy.io/modules/validators/passkeyValidator), [multisig](https://docs.biconomy.io/modules/validators/ownableValidator), [ECDSA](https://docs.biconomy.io/modules/validators/k1Validator#k1validator-overview) sign-in; [session keys](https://docs.biconomy.io/modules/validators/smartSessions)</td>
      <td>Bring Your Own Signer</td>
      <td>[Quickstart](https://docs.biconomy.io/tutorials/smart-sessions)</td>
    </tr>

    
    <tr>
      <td>[Dynamic](https://dynamic.xyz/)</td>
      <td>[Docs](https://docs.dynamic.xyz/)</td>
      <td>[Embedded wallets](https://docs.dynamic.xyz/wallets/embedded-wallets/dynamic-embedded-wallets)<br/>
      Auth: [passkey](https://docs.dynamic.xyz/wallets/v1-embedded/transactional-mfa/passkeys#passkeys), [email/social/SMS](https://docs.dynamic.xyz/authentication-methods/email-social-sms) sign-in</td>
      <td>TEE; TSS-MPC (just added)</td>
      <td>[Get started](https://www.dynamic.xyz/get-started)</td>
    </tr>

    
    <tr>
      <td>[Openfort](https://openfort.io/)</td>
      <td>[Docs](https://www.openfort.io/docs)</td>
      <td>[Embedded wallets](https://www.openfort.io/docs/guides/react/configuration), [Backend wallets](https://www.openfort.io/docs/guides/server/dev), [Ecosystem wallets](https://www.openfort.io/docs/guides/ecosystem)<br/>
      Auth: [passkeys](https://www.openfort.io/docs/guides/javascript/auth), [social](https://www.openfort.io/docs/guides/javascript/auth), [email](https://www.openfort.io/docs/guides/javascript/auth)</td>
      <td>SSS</td>
      <td>[Quickstart](https://www.openfort.io/docs/guides/react)</td>
    </tr>

    
    <tr>
      <td>[Para](https://www.getpara.com/)</td>
      <td>[Docs](https://docs.getpara.com/)</td>
      <td>[Embedded wallets](https://docs.getpara.com/getting-started/initial-setup/react-nextjs); robust policy engine for sessions<br/>
      Auth: [email](https://docs.getpara.com/customize-para/email-login), [social](https://docs.getpara.com/customize-para/oauth-social-logins), [SMS](https://docs.getpara.com/customize-para/phone-login) sign-in</td>
      <td>MPC + DKG</td>
      <td>[Quickstart](https://docs.getpara.com/integration-guides/overview-evm)</td>
    </tr>

    
    <tr>
      <td>[Phantom](https://phantom.com/learn/developers)</td>
      <td>[Docs](https://docs.phantom.com/)</td>
      <td>[Embedded wallets](https://github.com/phantom/wallet-sdk) (Web SDK & Native Mobile SDK)<br/>
      Auth: [Google](https://phantom.com/learn/blog/deep-dive-log-in-to-phantom-with-email) sign-in</td>
      <td>SSS</td>
      <td>[Quickstart](https://docs.phantom.com/embedded/getting-started-with-phantom-embedded-wallets)</td>
    </tr>

    
    <tr>
      <td>[Pimlico](https://pimlico.io/)</td>
      <td>[Docs](https://docs.pimlico.io/)</td>
      <td>[permissionless.js](https://docs.pimlico.io/permissionless), a flexible SDK for interfacing with various smart accounts, bundlers/paymasters, and signers.</td>
      <td></td>
      <td>[Tutorial](https://docs.pimlico.io/permissionless/tutorial/tutorial-1)</td>
    </tr>

    
    <tr>
      <td>[Privy](https://privy.io/)</td>
      <td>[Docs](https://docs.privy.io/)</td>
      <td>[Embedded wallets](https://docs.privy.io/guide/embedded-wallets), [server wallets](https://docs.privy.io/guide/overview-server-wallets), [server-delegated actions](https://docs.privy.io/guide/server-delegated-actions)<br/>
      Auth: [passkey](https://docs.privy.io/guide/authentication), [social](https://docs.privy.io/guide/authentication), [email](https://docs.privy.io/guide/authentication), [SMS](https://docs.privy.io/guide/authentication)</td>
      <td>TEE + SSS</td>
      <td>[Quickstart](https://docs.privy.io/guide/react/quickstart)</td>
    </tr>

    
    <tr>
      <td>[Reown](https://reown.com/) (formerly WalletConnect)</td>
      <td>[Docs](https://docs.reown.com/)</td>
      <td>Popular UI component for selecting a wallet<br/>
      Embedded wallet with social/email sign-in</td>
      <td></td>
      <td>[Quickstart](https://docs.reown.com/quickstart)</td>
    </tr>

    
    <tr>
      <td>[thirdweb](https://thirdweb.com/)</td>
      <td>[Docs](https://portal.thirdweb.com/connect/wallet/overview)</td>
      <td>Embedded wallets<br/>
      Auth: [passkey](https://portal.thirdweb.com/connect/wallet/sign-in-methods/configure), [social](https://portal.thirdweb.com/connect/wallet/sign-in-methods/configure), [email](https://portal.thirdweb.com/connect/wallet/sign-in-methods/configure), [SMS](https://portal.thirdweb.com/connect/wallet/sign-in-methods/configure), OIDC, or generic auth</td>
      <td></td>
      <td>[Quickstart](https://portal.thirdweb.com/connect/wallet/get-started)</td>
    </tr>

    
    <tr>
      <td>[Turnkey](https://www.turnkey.com/)</td>
      <td>[Docs](https://docs.turnkey.com/)</td>
      <td>[Embedded wallet](https://docs.turnkey.com/reference/embedded-wallet-kit), [policy engine](https://docs.turnkey.com/concepts/policies/overview), [delegated access](https://docs.turnkey.com/concepts/policies/delegated-access), [signing automation](https://docs.turnkey.com/signing-automation/overview), [sessions](https://docs.turnkey.com/authentication/sessions)<br/>
      [Server-side SDKs](https://docs.turnkey.com/sdks/introduction) for auth, wallet management, and policies<br/>
      Auth: [passkey](https://docs.turnkey.com/authentication/passkeys/introduction), [social](https://docs.turnkey.com/authentication/social-logins), [email](https://docs.turnkey.com/authentication/email), [SMS](https://docs.turnkey.com/authentication/sms) login</td>
      <td>TEE</td>
      <td>[Quickstart](https://docs.turnkey.com/getting-started/quickstart)</td>
    </tr>

    
    <tr>
      <td>[Web3Auth](https://web3auth.io/)</td>
      <td>[Docs](https://web3auth.io/docs)</td>
      <td>Embedded wallet<br/>
      Auth: [passkey](https://web3auth.io/docs/features/passkeys), [social](https://web3auth.io/docs/auth-provider-setup/social-providers/), [email](https://web3auth.io/docs/auth-provider-setup/email-provider/), [SMS](https://web3auth.io/docs/auth-provider-setup/sms-provider/)</td>
      <td>MPC-SSS/TSS</td>
      <td>[Quickstart](https://web3auth.io/docs/quick-start)</td>
    </tr>

    
    <tr>
      <td>[ZeroDev](https://zerodev.app/)</td>
      <td>[Docs](https://docs.zerodev.app/)</td>
      <td>[Smart contract accounts](https://docs.zerodev.app/sdk/core-api/create-account)<br/>
      [Session keys](https://docs.zerodev.app/sdk/permissions/intro) with several options for signature schemes (ECDSA, Passkey, Multisig), policies, and actions.</td>
      <td></td>
      <td>[Quickstart](https://docs.zerodev.app/sdk/getting-started/quickstart)</td>
    </tr>
  </tbody>
</table>

## Provider Details

### Alchemy

[Account Kit](https://accountkit.alchemy.com/) is a complete solution for account abstraction. Using Account Kit, you can create a smart contract wallet for every user that leverages account abstraction to simplify every step of your app's onboarding experience. It also offers Gas Manager and Bundler APIs for sponsoring gas and batching transactions.

To get started, sign up for an [Alchemy account](https://www.alchemy.com/), visit the [documentation](https://accountkit.alchemy.com/), follow the [quickstart](https://accountkit.alchemy.com/react/quickstart) guide or check out the demo [here](https://demo.alchemy.com/).

Alchemy helps you to replace 3rd-party pop-up wallets with native in-app auth. Drop in branded sign-in modals for email, passkeys, and social logins with plug-n-play components.

To get started, sign up for an [Alchemy account](https://dashboard.alchemy.com/accounts), visit the [documentation](https://accountkit.alchemy.com/), follow the [quickstart](https://accountkit.alchemy.com/react/quickstart) guide. To further streamline UX with no gas fees or signing for users, see Alchemy's [AA infra offering](infra-providers.md#alchemy) and a demo [here](https://demo.alchemy.com/).

Supported Networks
- Monad Testnet


### Biconomy

[Biconomy](https://biconomy.io) is the most comprehensive smart account and execution infrastructure platform that enables seamless, user-friendly experiences across single or multiple chains. With Biconomy, developers can build superior onchain UX through gas abstraction, sessions, batching, and one-click signatures for complex actions on any number of networks.

To get started, visit the [documentation](https://docs.biconomy.io/) or follow the [simple tutorial](https://docs.biconomy.io/tutorials/simple).

Supported Networks
- Monad Testnet


### Dynamic

[Dynamic](https://dynamic.xyz/) offers smart and beautiful login flows for crypto-native users, simple onboarding flows for everyone else, and powerful developer tools that go beyond authentication.

To get started, visit the [documentation](https://docs.dynamic.xyz/) or follow the [quickstart](https://docs.dynamic.xyz/docs/quickstart) guide.

Supported Networks
- Monad Testnet


### MetaKeep

[MetaKeep](https://www.metakeep.xyz/) is the #1 self-custody infra for users & AI. Onboard 300x more users in 1 API call, 5 mins.

To get started, setup an [onboarding call](https://mtkp.xyz/priority-meet) with the team.

Supported Networks
- Monad Testnet


### Para

[Para](https://www.getpara.com/) is the easiest and most secure way to onboard all your users and support them throughout their crypto journey. We support projects throughout their growth, ranging from personal projects to many of the most trusted teams in crypto and beyond.

Para's cross-app embedded wallets work universally across apps, chains, and ecosystem, so whether users start transacting on EVM, Solana, or Cosmos, they can onboard once and transact forever, all with the same wallet.

To get started, visit the [documentation](https://docs.getpara.com/welcome) or follow the [quickstart](https://docs.getpara.com/welcome#adding-para-to-your-app) guide.

Supported Networks
- Monad Testnet


### Phantom

[Phantom](https://phantom.com/) is the world's leading crypto wallet for managing digital assets and accessing decentralized applications. 

Phantom embedded wallets enable seamless, seedless onboarding with in-app, non-custodial access--no app switching or seed phrases required.

To get started, visit the [documentation](https://docs.phantom.com/) or follow the [quickstart](https://docs.phantom.com/embedded/getting-started-with-phantom-embedded-wallets) guide.

Supported Networks
- Monad Testnet


### Pimlico

[Pimlico](https://pimlico.io/) is the world's most advanced ERC-4337 account abstraction infrastructure platform. Pimlico provides a suite of tools and services to help you build, deploy, and manage smart accounts on Ethereum and other EVM-compatible chains.

To get started, visit the [documentation](https://docs.pimlico.io/) or follow the [quickstart](https://docs.pimlico.io/permissionless/tutorial/tutorial-1) guide.

Supported Networks
- Monad Testnet


### Privy

[Privy](https://privy.io/) helps you onboard any user to crypto no matter how familiar they are with the space. Power flexible, powerful wallets under the hood for any application, securely.

To get started, visit the [documentation](https://docs.privy.io/) or follow the [quickstart](https://docs.privy.io/guide/react/quickstart) guide.

Supported Networks
- Monad Testnet


### Reown

[Reown](https://reown.com/) gives developers the tools to build user experiences that make digital ownership effortless, intuitive, and secure. 

#### AppKit

AppKit is a powerful, free, and fully open-source SDK for developers looking to integrate wallet connections and other Web3 functionalities into their apps on any EVM and non-EVM chain. In just a few simple steps, you can provide your users with seamless wallet access, one-click authentication, social logins, and notifications—streamlining their experience while enabling advanced features like on-ramp functionality, in-app token swaps and smart accounts.

To get started, visit the [documentation](https://docs.reown.com/) or follow the [quickstart](https://reown.com/blog/how-to-get-started-with-reown-appkit-on-monad-testnet) guide.

Supported Networks
- Monad Testnet


### thirdweb

[thirdweb](https://thirdweb.com/) provides client-side SDKs for user onboarding, identity and transactions.

* Onboard new users to your apps with every wallet & login method
* create a complete picture of all your users via user analytics & identity linking
* facilitate onchain transactions via onramps, swaps & bridging

To get started:

1. Sign up for a [free thirdweb account](https://thirdweb.com/team)
2. Visit [Connect Documentation](https://portal.thirdweb.com/connect/sign-in/ConnectButton) and [Connect Playground](https://playground.thirdweb.com/connect/sign-in/button)

Supported Networks
- Monad Testnet


### Turnkey

[Turnkey](https://www.turnkey.com/) is secure, flexible, and scalable wallet infrastructure. Create millions of embedded wallets, eliminate manual transaction flows, and automate onchain actions - all without compromising on security.

To get started, visit the [documentation](https://docs.turnkey.com/) or follow the [quickstart](https://docs.turnkey.com/getting-started/embedded-wallet-quickstart) guide.

Supported Networks
- Monad Testnet


### Web3Auth

[Web3Auth](https://web3auth.io/) simplifies Web3 access with social logins, customisable wallet UI, and advanced security, with non custodial MPC wallet management.

To get started, visit the [documentation](https://web3auth.io/docs) or follow the [quickstart](https://web3auth.io/docs/quick-start) guide.

Supported Networks
- Monad Testnet


### Zerodev

[ZeroDev](https://zerodev.app) is the most powerful smart account development platform. With ZeroDev, you can build Web3 experiences without gas, confirmations, seed phrases, and bridging.

To get started, visit the [documentation](https://docs.zerodev.app/) or follow the [quickstart](https://docs.zerodev.app/sdk/getting-started/quickstart) guide.

Supported Networks
- Monad Testnet
---

# Analytics

A number of analytics providers are supporting Monad testnet.

## Flipside

[Flipside](https://flipsidecrypto.xyz/home/) generates the most reliable and comprehensive blockchain data. All for free.

To get started, check out the [Flipside documentation](https://docs.flipsidecrypto.xyz/) or [Flipside 101 dashboard](https://flipsidecrypto.xyz/charliemarketplace/flipside-101-ll5imK) to learn more!

Supported Networks
- Monad Testnet
---

# Block Explorers

## Provider Summary

The following block explorers support Monad Testnet:

| Block explorer         | Powered by  | URL                                  | Verifier Type & URL                                                                   |
| ---------------------- | ----------- | ------------------------------------ | ------------------------------------------------------------------------------------- |
| **MonadExplorer**      | BlockVision | https://testnet.monadexplorer.com/   | Sourcify: `https://sourcify-api-monad.blockvision.org`                                |
| **Monadscan**          | Etherscan   | https://testnet.monadscan.com/       | Etherscan: `https://api-testnet.monadscan.com/api`                                    |
| **SocialScan - Monad** | Hemera      | https://monad-testnet.socialscan.io/ | Etherscan: `https://api.socialscan.io/monad-testnet/v1/explorer/command_api/contract` |

## Provider Details

### BlockVision

[MonadExplorer](https://monadexplorer.com) is built by [BlockVision](https://blockvision.org/), a leading provider of next-gen data infrastructure and enterprise solutions for the blockchain. BlockVision is supports the EVM and Sui and specializes in explorer service, RPC nodes, indexing APIs and validator service.

To get started, visit the [documentation](https://docs.blockvision.org/reference/monad-indexing-api).

Supported Networks
- Monad Testnet

### Etherscan

[Monadscan](https://testnet.monadscan.com) is built by [Etherscan](https://etherscan.io/) to deliver trusted and high-performance access to on-chain data. Leveraging Etherscan’s industry-leading expertise, MonadScan provides robust explorer tools, developer-friendly APIs, and reliable infrastructure tailored for the Monad ecosystem.

To get started, visit the [documentation](https://docs.monadscan.com/).

Supported Networks
- Monad Testnet

### SocialScan

[Monad SocialScan](https://monad-testnet.socialscan.io/) is a high-performance block explorer built by the [SocialScan](https://socialscan.io/) team.

To get started, visit the [documentation](https://thehemera.gitbook.io/explorer-api).

Supported Networks
- Monad Testnet
---

# Cross-Chain

## Definitions

At a high level, bridges offer the following features:

<table>
    <thead>
    <tr>
        <th>Feature</th>
        <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>**Arbitrary Messaging Bridge (AMB)**</td>
        <td>Allows arbitrary messages to be securely relayed from a smart contract on chain 1 to a smart contract on chain 2.<br/><br/>
        AMB provides guarantees that messages delivered to chain 2 represent events finalized on chain 1.</td>
    </tr>
    <tr>
        <td>**Token Bridge**</td>
        <td>Allows user to lock native tokens or ERC20 tokens on chain 1 and mint claim tokens on chain 2.<br/><br/>
        Bridge maintains the invariant that, for each token minted on chain 2, there exists a corresponding token locked on chain 1.</td>
    </tr>
    </tbody>
</table>

## Provider Summary

The following providers support Monad Testnet:

<table>
    <thead>
    <tr>
        <th>Provider</th>
        <th>Docs</th>
        <th>Bridge Type</th>
        <th>Contract Addresses</th>
        <th>Explorer</th>
    </tr>
    </thead>
    <tbody>
    

    <tr>
        <td>Chainlink CCIP</td>
        <td>[Docs](https://docs.chain.link/ccip)</td>
        <td>AMB; Token Bridge</td>
        <td>
            - Router: [0x5f16e51e3Dcb255480F090157DD01bA962a53E54](https://testnet.monadexplorer.com/address/0x5f16e51e3Dcb255480F090157DD01bA962a53E54)
            - Supported lanes can be found [here](https://docs.chain.link/ccip/directory/testnet/chain/monad-testnet)
        </td>
        <td>[CCIP Explorer](https://ccip.chain.link/)</td>
    </tr>

    <tr>
        <td>Garden</td>
        <td>[Docs](https://docs.garden.finance/)</td>
        <td>Token Bridge</td>
        <td>
            Wrapped Assets:
            - [cbBTC](https://testnet.monadexplorer.com/address/0x1dC94FdcAd8Aee13cfd34Db8a26d26E31572805c)
            - [USDC](https://testnet.monadexplorer.com/address/0xE99D8A21d4F2ad2f90c279c869311503f9e25867)
        </td>
        <td></td>
    </tr>

    <tr>
        <td>Hyperlane</td>
        <td>[Docs](https://docs.hyperlane.xyz)</td>
        <td>AMB; Token Bridge</td>
        <td>
            - ISM Operator: [0x734628f55694d2a5f4de3e755ccb40ecd72b16d9](https://testnet.monadexplorer.com/address/0x734628f55694d2a5f4de3e755ccb40ecd72b16d9)
        </td>
        <td>[Hyperlane Explorer](https://explorer.hyperlane.xyz)</td>
    </tr>

    <tr>
        <td>LayerZero</td>
        <td>[Docs](https://docs.layerzero.network)</td>
        <td>AMB; Token Bridge</td>
        <td>
            [LayerZero V2 Address Reference](https://docs.layerzero.network/v2/deployments/deployed-contracts?chains=monad-testnet)

            - Endpoint V2: [0x6C7Ab2202C98C4227C5c46f1417D81144DA716Ff](https://testnet.monadexplorer.com/address/0x6C7Ab2202C98C4227C5c46f1417D81144DA716Ff)
            - Endpoint ID: 40204
            - SendUln302: [0xd682ECF100f6F4284138AA925348633B0611Ae21](https://testnet.monadexplorer.com/address/0xd682ECF100f6F4284138AA925348633B0611Ae21)
            - ReceiveUln302: [0xcF1B0F4106B0324F96fEfcC31bA9498caa80701C](https://testnet.monadexplorer.com/address/0xcF1B0F4106B0324F96fEfcC31bA9498caa80701C)
            - BlockedMessageLib: [0x926984a57b10a3a5c4cfdbac04daaa0309e78932](https://testnet.monadexplorer.com/address/0x926984a57b10a3a5c4cfdbac04daaa0309e78932)
            - LZ Executor: [0x9dB9Ca3305B48F196D18082e91cB64663b13d014](https://testnet.monadexplorer.com/address/0x9dB9Ca3305B48F196D18082e91cB64663b13d014)
            - LZ Dead DVN: [0xF49d162484290EAeAd7bb8C2c7E3a6f8f52e32d6](https://testnet.monadexplorer.com/address/0xF49d162484290EAeAd7bb8C2c7E3a6f8f52e32d6)
        </td>
        <td>[LayerZeroScan](https://testnet.layerzeroscan.com/)</td>
    </tr>

    <tr>
        <td>Wormhole</td>
        <td>[Docs](https://wormhole.com/docs)</td>
        <td>AMB; Token Bridge</td>
        <td>
            - Wormhole specific Chain ID: 48
            - Core Contract: [0xBB73cB66C26740F31d1FabDC6b7A46a038A300dd](https://testnet.monadexplorer.com/address/0xBB73cB66C26740F31d1FabDC6b7A46a038A300dd)
            - Token Bridge: [0xF323dcDe4d33efe83cf455F78F9F6cc656e6B659](https://testnet.monadexplorer.com/address/0xF323dcDe4d33efe83cf455F78F9F6cc656e6B659)
            - Wormhole Relayer: [0x362fca37E45fe1096b42021b543f462D49a5C8df](https://testnet.monadexplorer.com/address/0x362fca37E45fe1096b42021b543f462D49a5C8df)
        </td>
        <td>[WormholeScan](https://wormholescan.io/#/?network=Testnet)</td>
    </tr>
    </tbody>
</table>



## Provider Details

### Chainlink CCIP

[Chainlink](https://chain.link/) Cross-Chain Interoperability Protocol (CCIP) is the standard for cross-chain interoperability. CCIP enables developers to build secure cross-chain apps that can transfer tokens, send messages, and initiate actions across blockchains.

Through the [Cross-Chain Token (CCT)](https://blog.chain.link/ccip-v-1-5-upgrade/) standard, CCIP enables token developers to integrate new and existing tokens with CCIP in a self-serve manner in minutes, without requiring vendor lock-in, hard-coded functions, or external dependencies that may limit future optionality. CCTs support self-serve deployments, full control and ownership for developers, zero-slippage transfers, and enhanced programmability via configurable rate limits and reliability features such as Smart Execution. CCIP is powered by Chainlink decentralized oracle networks (DONs)—a proven standard with a track record of securing tens of billions of dollars and enabling over $19 trillion in onchain transaction value.

Key CCIP developer tools:
- [CCIP official documentation](https://docs.chain.link/ccip): start integrating CCIP into your cross-chain application.
- [CCIP Token Manager](https://tokenmanager.chain.link/): an intuitive front-end web interface for the deployment of new and management of existing CCTs by their developers, including no-code guided deployments and configuration tools.
- [CCIP SDK](https://docs.chain.link/ccip/ccip-javascript-sdk): a software development kit that streamlines the process of integrating CCIP, allowing developers to use JavaScript to create a token transfer frontend dApp.

Contract Addresses for Monad Testnet:
- [0x5f16e51e3Dcb255480F090157DD01bA962a53E54](https://testnet.monadexplorer.com/address/0x5f16e51e3Dcb255480F090157DD01bA962a53E54)

Supported Networks:
- Monad Testnet


### Garden

[Garden](https://garden.finance/) is transforming Bitcoin interoperability with its next-gen bridge. It is built by the renBTC team using an intents based architecture with trustless settlement, enabling cross-chain Bitcoin swaps in as little as 30 seconds with zero custody risk.

In its first year, Garden processed over $1 billion in volume—proving the market's demand for seamless, cost-effective Bitcoin bridging solutions.

Now, Garden is unlocking a new era of interoperability—supporting non-likewise assets, external liquidity, and a wallet-friendly API—to onboard the next wave of partners and users.

To get started, visit the [documentation](https://docs.garden.finance/).

Supported Networks
- Monad Testnet


### Hyperlane

[Hyperlane](https://hyperlane.xyz/) is a permissionless interoperability protocol for cross-chain communication. It enables message passing and asset transfers across different chains without relying on centralized intermediaries or requiring any permissions.

To get started, visit the [Hyperlane documentation](https://docs.hyperlane.xyz/).

#### Hyperlane Explorer

To view status of your cross chain transactions, please visit the [Hyperlane Explorer](https://explorer.hyperlane.xyz/).

Supported Networks:
- Monad Testnet


### LayerZero

[LayerZero](https://layerzero.network/) is an omnichain interoperability protocol that enables cross-chain messaging. Applications built on Monad can use the LayerZero protocol to connect to 35+ supported blockchains seamlessly.

To get started with integrating LayerZero, visit the LayerZero [documentation](https://docs.layerzero.network/v1/developers/evm/evm-guides/send-messages) and provided examples on [GitHub](https://github.com/LayerZero-Labs/endpoint-v1-solidity-examples).

Supported Networks:
- Monad Testnet


### Wormhole

[Wormhole](https://wormhole.com/) is a generic messaging protocol that provides secure communication between blockchains.

By integrating Wormhole, a Monad application can access users and liquidity on > 30 chains and > 7 different platforms.

See this [quickstart](https://wormhole.com/docs/tutorials/by-product/contract-integrations/cross-chain-contracts/) to get started with integrating Wormhole in your Monad project.

For more information on integrating Wormhole, visit their [documentation](https://wormhole.com/docs/) and the [provided GitHub examples](https://github.com/wormhole-foundation/wormhole-examples).

Contract Addresses for Monad Testnet:
- Core: [0xBB73cB66C26740F31d1FabDC6b7A46a038A300dd](https://testnet.monadexplorer.com/address/0xBB73cB66C26740F31d1FabDC6b7A46a038A300dd)
- Relayer: [0x362fca37E45fe1096b42021b543f462D49a5C8df](https://testnet.monadexplorer.com/address/0x362fca37E45fe1096b42021b543f462D49a5C8df)

Supported Networks:
- Monad Testnet
---

# Debugging Onchain

## Transaction introspection/tracing

* [Tenderly](https://dashboard.tenderly.co/explorer)
* [EthTx Transaction Decoder](https://ethtx.info/)
* [https://openchain.xyz/](https://openchain.xyz/)
* [Bloxy](https://bloxy.info/)
* [https://github.com/naddison36/tx2uml](https://github.com/naddison36/tx2uml) - OS tools for generating UML diagrams
* [https://github.com/apeworx/evm-trace](https://github.com/apeworx/evm-trace) - tracing tools

## Contract Decompilation

* [https://oko.palkeo.com/](https://oko.palkeo.com/): a hosted version of the [Panoramix](https://github.com/palkeo/panoramix) decompiler
---

# Indexers

The blockchain can be thought of as a list of blocks, transactions, and logs, as well as a series of global states. Indexers compute common transformations on this data to save downstream consumers the cost and complexity of doing so.

There are two main types of indexer services:
1. **[Data for common use cases](common-data.md)**: raw data (blocks, transactions, logs, traces) and derived data for common use cases (token balances, NFT holdings, DEX trades), computed across the entire blockchain
2. **[Indexing Frameworks](indexing-frameworks.md)** enable devleopers to build custom calculators for a specific smart contract

## Data for common use cases

Data providers offer raw and transformed data for common use cases via API or by streaming to your local environment.

Raw data includes:
- blocks, transactions, logs, traces (potentially decoded using contract ABIs)

Transformed data includes:
- balances (native tokens, ERC20s, NFTs)
- transfers (native tokens, ERC20s, NFTs)
- DEX trades
- market data
- and more

See [Common Data](common-data.md) for a fuller list of features and providers.

## Indexing Frameworks

Smart contract indexers are custom off-chain calculators for a specific smart contract. They maintain additional off-chain state and perform additional computation. Since blockchain data is public, anyone may deploy a subgraph involving state or logs from any smart contract.

See [Indexing Frameworks](indexing-frameworks.md) for a list of features and providers.
---

# Common Data

## Features

In order to improve developer understanding of feature coverage, we have collected the most common features offered by providers:

| Feature         | Sub-Feature | Description |
| --------------- | ----------- | ----------- |
| **Chain data**  |             | Raw data (blocks, transactions, logs, traces) in SQL-like format. Transactions and logs may optionally be decoded based on ABI  |
| **Balances**    | Native      | Native token holdings of an address, real-time or snapshot. May include price annotations |
|                 | ERC20       | ERC20 holdings of an address, real-time or snapshot. May include price annotations |
|                 | NFT         | NFT (ERC721 or ERC1155) holdings of an address, real-time or snapshot |
| **Transfers**   | Native      | Native token transfers involving a particular address. May include price annotations |
|                 | ERC20       | ERC20 transfers involving a particular address. May include price annotations |
|                 | NFT         | NFT transfers involving a particular address |
| **DEX trades**  |             | Normalized trade data across major DEXes |
| **Market data** |             | Market data for ERC20s |

Balances are nontrivial because each ERC20 and NFT collection stores its balances in contract storage. Transfers are nontrivial because they frequently occur as subroutines. Annotating with prices and normalizing across DEXes add additional convenience.

## Access models

- **APl**: Data lives on provider's servers; make queries via API
- **Stream**: Data is replicated to your environment


## Provider Summary

The following providers support Monad Testnet:

<table>
  <thead>
    <tr>
      <th>Provider</th>
      <th>Docs</th>
      <th>Supported services</th>
      <th>Access model</th>
    </tr>
  </thead>
  <tbody>
    
    <tr>
      <td>[Alchemy](https://www.alchemy.com/)</td>
      <td>[Docs](https://docs.alchemy.com/)</td>
      <td>Balances: [native, ERC20](https://docs.alchemy.com/reference/token-api-quickstart) and [NFT](https://docs.alchemy.com/reference/nft-api-quickstart)<br/>
      Transfers: [native, ERC20, and NFTs](https://docs.alchemy.com/reference/transfers-api-quickstart)<br/>
      [Webhooks](https://docs.alchemy.com/reference/notify-api-quickstart)</td>
      <td>API; Streaming (webhooks)</td>
    </tr>

    
    <tr>
      <td>[Allium](https://www.allium.so/)</td>
      <td>[Docs](https://docs.allium.so/)</td>
      <td>Chain data (blocks, transactions, logs, traces, contracts) (via [Explorer](https://docs.allium.so/data-products-analytics/allium-explorer) (historical) and [Datastreams](https://docs.allium.so/realtime/kafka-blockchains-70+) (realtime) products)<br/>
      Transfers (native, ERC20, and NFTs) (via [Developer](https://docs.allium.so/data-products-real-time/allium-developer/wallet-apis/activities) (realtime) product)</td>
      <td>API, except streaming for Datastreams product</td>
    </tr>
    

    
    <tr>
      <td>[Codex](https://www.codex.io/)</td>
      <td>[Docs](https://docs.codex.io/reference/overview)</td>
      <td>Token- and trading-centric data:<br/>
      [Token](https://docs.codex.io/reference/tokens-quickstart) charts, metadata, prices, events, and detailed stats (see [dashboard](https://www.defined.fi/tokens/discover?network=mon-test))<br/>
      [NFT](https://docs.codex.io/reference/nft-quickstart) metadata, events, and detailed stats</td>
      <td>API</td>
    </tr>

    
    <tr>
      <td>[Dune Echo](https://dune.com/echo/)</td>
      <td>[Docs](https://docs.dune.com/echo/overview)</td>
      <td>Transactions and logs (raw or decoded)<br/>
      Native token and ERC20 balances</td>
      <td>API</td>
    </tr>

    
    <tr>
      <td>[GoldRush](https://goldrush.dev/) (by Covalent)</td>
      <td>[Docs](https://goldrush.dev/docs/quickstart)</td>
      <td>Chain data: Blocks, enriched transactions and logs (raw and decoded)<br/>
      Balances: native, ERC20, NFTs & Portfolio<br/>
      Transactions: Full historical with decoded transfer events</td>
      <td>API</td>
    </tr>

    
    <tr>
      <td>[Goldsky](https://goldsky.com/)</td>
      <td>[Docs](https://docs.goldsky.com/)</td>
      <td>Chain data: blocks, enriched transactions, logs, and traces via [Mirror](https://docs.goldsky.com/mirror/introduction). [Fast scan](https://docs.goldsky.com/mirror/sources/direct-indexing#backfill-vs-fast-scan) is supported</td>
      <td>API; Streaming</td>
    </tr>

    
    <tr>
      <td>[Mobula](https://mobula.io/)</td>
      <td>[Docs](https://docs.mobula.io/introduction)</td>
      <td>Chain data<br/>
      Balances: [native, ERC20](https://docs.mobula.io/rest-api-reference/endpoint/wallet-portfolio) and [NFT](https://docs.mobula.io/rest-api-reference/endpoint/wallet-nfts)<br/>
      Transfers: [native, ERC20](https://docs.mobula.io/rest-api-reference/endpoint/wallet-transactions) and NFT<br/>
      [DEX trades](https://docs.mobula.io/rest-api-reference/endpoint/market-trades-pair)<br/>
      [Market data](https://docs.mobula.io/rest-api-reference/endpoint/market-data) for ERC20s</td>
      <td>API</td>
    </tr>

    
    <tr>
      <td>[Pangea](https://pangea.foundation/)</td>
      <td>[Docs](https://docs.pangea.foundation/)</td>
      <td>[Chain data](https://docs.pangea.foundation/chain-data/evm/blocks.html): blocks, transactions, logs<br/>
      Transfers: [ERC20](https://docs.pangea.foundation/standard-contracts/erc20-transfers.html)<br/>
      [DEX metadata and prices](https://docs.pangea.foundation/markets/uniswap-v3-reference.html): UniswapV3, UniswapV2, Curve</td>
      <td>API</td>
    </tr>

    
    <tr>
      <td>[Quicknode](https://www.quicknode.com/)</td>
      <td>[Docs](https://www.quicknode.com/docs/streams/getting-started)</td>
      <td>[Chain data](https://www.quicknode.com/docs/streams/getting-started): blocks, transactions, logs, traces (live or historical) (raw or filtered)</td>
      <td>Streaming</td>
    </tr>

    
    <tr>
      <td>[Reservoir](https://reservoir.tools/)</td>
      <td>[Docs](https://nft.reservoir.tools/reference/overview)</td>
      <td>Balances, transfers, trades, and market data for NFTs</td>
      <td>API</td>
    </tr>

    
    <tr>
      <td>[thirdweb](https://thirdweb.com/)</td>
      <td>[Docs](https://insight-api.thirdweb.com/guide/getting-started)</td>
      <td>Chain data: [blocks](https://insight-api.thirdweb.com/reference#tag/blocks), [transactions](https://insight-api.thirdweb.com/guide/blueprints#transactions-blueprint), [logs](https://insight-api.thirdweb.com/guide/blueprints#events-blueprint), [contracts](https://insight-api.thirdweb.com/reference#tag/contracts)<br/>
      [Balances](https://insight-api.thirdweb.com/guide/blueprints#tokens-blueprint): native, ERC20, NFTs<br/>
      [NFTs](https://insight-api.thirdweb.com/reference#tag/nfts)</td>
      <td>API</td>
    </tr>
    
    <tr>
      <td>[Unmarshal](https://unmarshal.io/)</td>
      <td>[Docs](https://docs.unmarshal.io)</td>
      <td>[Balances](https://docs.unmarshal.io/reference/fungibleerc20tokenbalances): ERC20 and NFT<br/>
      [Transactions](https://docs.unmarshal.io/reference/get-v3-chain-address-address-transactions) with price annotations<br/>
      [NFT API](https://docs.unmarshal.io/reference/get-v2-chain-address-address-nft-transactions) (transactions and metadata)</td>
      <td>API</td>
    </tr>
    
    
    <tr>
      <td>[Zerion](https://zerion.io/)</td>
      <td>[Docs](https://zerion.io/api)</td>
      <td>[Wallet info](https://developers.zerion.io/reference/wallets):<br/>
        [Balances](https://developers.zerion.io/reference/listwalletpositions) (native, ERC20, and NFTs)<br/>
        [Transactions](https://developers.zerion.io/reference/listwallettransactions) (multichain with prices)<br/>
        [Portfolio](https://developers.zerion.io/reference/getwalletportfolio), [PNL](https://developers.zerion.io/reference/getwalletpnl#/) and [Historical Positions](https://developers.zerion.io/reference/getwalletchart)<br/>
        [Notification Webhooks](https://developers.zerion.io/v1.0-subscriptions/reference/createsubscriptionwallettransactions)</td>
      <td>API; Webhooks</td>
    </tr>
  </tbody>
</table>

## Provider Details

### Alchemy

[Alchemy](https://www.alchemy.com) is a popular API provider and developer platform. Alchemy offers subgraphs as a hosted service, as well as offerings in other categories including smart wallets, AA infra, NFT APIs, Token APIs, Transfers APIs, Webhooks and RPC APIs.

Supported Networks
- Monad Testnet

### Allium

[Allium](https://www.allium.so/) is an Enterprise Data Platform that serves accurate, fast, and simple blockchain data. Allium offers near real-time Monad Testnet data for infrastructure needs and enriched Monad Testnet data (NFT, DEX, Decoded) for research and analytics.

Allium supports data delivery to multiple [destinations](https://docs.allium.so/integrations/overview), including Snowflake, Bigquery, Databricks, and AWS S3. To get started, contact Allium [here](https://www.allium.so/contact).

| Product        | Description / mechanism                                 | Monad Testnet data supported       |
| -------------- | ------------------------------------------------------- | ----------- |
| [Explorer](https://docs.allium.so/data-products-analytics/allium-explorer)       | Historical data (postgres/API)           | Chain data: blocks, transactions, logs, traces, contracts |
| [Developer](https://docs.allium.so/data-products-real-time/allium-developer) | Real-time data (postgres/API)           | [Transfers](https://docs.allium.so/products/allium-developer/wallet-apis/activities) (native, ERC20, ERC721, ERC1155) |
| [Datastreams](hhttps://docs.allium.so/data-products-real-time/allium-datastreams)   | Real-time data (streaming - Kafka, PubSub, or SNS) | Chain data: blocks, transactions, logs, traces, contracts |


Supported Networks
- Monad Testnet


### Codex

[Codex](https://www.codex.io/) API provides fast and accurate enriched data, meticulously structured to easily plug straight into your application.

To get started, visit the [documentation](https://docs.codex.io/reference/overview/) or sign up for an API key at [dashboard.codex.io](https://dashboard.codex.io/).

Supported Networks
- Monad Testnet


### Dune Echo

[Dune](https://dune.com/) Echo makes building multi-chain application seamless. These APIs power several of the best teams building on crypto.

#### Available APIs:
​
**Token Balances**: Access accurate and fast real time balances of native and ERC20 tokens of accounts on EVM blockchains.
**Transactions**: Access transactions for accounts in real time across EVM blockchains.

To get started, visit the [documentation](https://docs.dune.com/echo/overview).

Supported Networks
- Monad Testnet


### GoldRush (by Covalent)

[GoldRush](https://goldrush.dev/) provides multichain data APIs and toolkits for easy web3 development across 100+ chains including Monad.

GoldRush offers structured onchain data, including multichain wallet balances, full transaction histories and decoded log events, for building apps and powering AI Agents. Join hundreds of top teams that leverage GoldRush to cut down their development time and scale their multichain offerings with enterprise-grade onchain data.

To get started, visit the [documentation](https://goldrush.dev/docs/quickstart) or [sign up](https://goldrush.dev/platform/auth/register/) for an API key.

Supported Networks
- Monad Testnet


### Goldsky

[Goldsky](https://goldsky.com/) is the go-to data indexer for web3 builders, offering high-performance subgraph hosting and realtime data replication pipelines.

Goldsky offers two core self-serve products that can be used independently or in conjunction to power your data stack.

- **Subgraphs**: Flexible indexing with typescript, with support for webhooks and more.
- **Mirror**: Get live blockchain data in your database or message queues with a single yaml config.


To get started, visit the [documentation](https://docs.goldsky.com/) or follow the [quickstart](https://docs.goldsky.com/subgraphs/guides/create-a-no-code-subgraph) guide.

Supported Networks
- Monad Testnet


### Mobula

[Mobula](https://mobula.io/) provides curated datasets for builders: market data with Octopus, wallets data, metadata with Metacore, alongside with REST, GraphSQL & SQL interfaces to query them.

You can get started playing around with the [API endpoints](https://docs.mobula.io/rest-api-reference/introduction) for free, and sign-up to the API dashboard once you need API keys (queries without API keys aren’t production-ready).

To get started, visit the [documentation](https://docs.mobula.io/introduction).

Supported Networks
- Monad Testnet



### Pangea

[Pangea](https://pangea.foundation/)'s real-time data is ideal for anyone who needs fresh pricing data in DeFi and cross-chain.

To get started, visit the [documentation](https://docs.pangea.foundation/).

Supported Networks
- Monad Testnet


### QuickNode

[QuickNode Streams](https://www.quicknode.com/docs/streams/getting-started) is a managed, push-based blockchain streaming service that provides guaranteed delivery of both live and sequential historical data. With QuickNode Streams, you can choose to receive raw or filtered data—such as specific contract events—pushed directly to your stream destination (webhook, S3, Postgres, Snowflake, or Functions).

- **Guaranteed Delivery**: Robust retry logic and guaranteed delivery ensures that you never miss critical events.
- **Push-Based Streaming**: Events are pushed to your destination, eliminating the need for manual polling.
- **Live & Sequential Historical Data**: Process new events as they occur, and seamlessly backfill older blocks in chronological order.
- **Raw or Filtered Data**: Define filtering conditions (e.g., specific contract addresses or event signatures) using JavaScript to reduce noise and receive only the data you care about.
- **Managed Service**: QuickNode handles all node infrastructure, allowing you to focus on building without worrying about uptime or scaling.

To get started, visit the [documentation](https://www.quicknode.com/docs/streams/getting-started) for detailed instructions on creating streams, setting filters, and choosing a delivery method. Or, check out the [quickstart guide](https://www.quicknode.com/guides/quicknode-products/streams/how-to-use-filters-with-streams) to deploy a simple webhook receiver and configure a stream in minutes.

Supported Networks
- Monad Testnet


### Reservoir

[Reservoir](https://reservoir.tools/) is a developer platform that lets you interact with the NFT market using a single toolkit. With the tools, you can build custom marketplaces, embed buying and selling into your app, and get distribution for your protocol's liquidity, among many other use cases.

To get started, visit the [documentation](https://nft.reservoir.tools/reference/dashboard-sign-up).

Supported Networks
- Monad Testnet


### thirdweb

thirdweb [Insight](https://portal.thirdweb.com/insight) is a fast, reliable and fully customizable way for developers to index, transform & query onchain data across 30+ chains. Insight includes out-of-the-box APIs for transactions, events, tokens. Developers can also define custom API schemas, or blueprints, without the need for ABIs, decoding, RPC, or web3 knowledge to fetch blockchain data.

thirdweb Insight can be used to fetch:
* all assets (ERC20, ERC721, ERC115) for a given wallet address
* all sales of skins on your in-game marketplace
* monthly protocol fees in the last 12 months
* the total cost of all accounts created using ERC-4337
* metadata for a given token (ERC20, ERC721, ERC115)
* daily active wallets for your application or game
* and so much more

To get started, sign up for a [free thirdweb account](https://thirdweb.com/team) and visit the [thirdweb Insight documentation](https://portal.thirdweb.com/insight/get-started)


### Unmarshal

[Unmarshal](https://unmarshal.io/) is a leading decentralized multi-chain data network, enabling Web3 projects to access accurate, real-time blockchain data across 55+ chains (including Monad Testnet). 

Leveraging AI-driven solutions, Unmarshal enhances data accessibility and insights for RWA, DePIN, AI Agents, DeFi, NFT, and GameFi platforms. Through robust APIs, notification services, and no-code indexers, it empowers dApps to deliver seamless user experiences while ensuring transparency, scalability, and innovation at the forefront of Web3 advancements.

To get started, visit the [documentation](https://docs.unmarshal.io).

Reach out at support@unmarshal.io

Supported Networks
- Monad Testnet


### Zerion

The [Zerion API](https://zerion.io/api) can be used to build feature-rich web3 apps, wallets, and protocols with ease. Across all major blockchains, you can access wallets, assets, and chain data for web3 portfolios. Zerion's infrastructure supports all major chains!

To get started, visit the [documentation](https://zerion.io/api).

Supported Networks
- Monad Testnet
---

# Indexing Frameworks

## Background

**Smart contract indexers** are off-chain calculators that compute additional metrics specific to one smart contract. Calculators can be thought of as extensions to a smart contract that do additional off-chain computation and maintain additional off-chain state.

*Simple example:* the [UniswapV2Pair contract](https://github.com/Uniswap/v2-core/blob/master/contracts/UniswapV2Pair.sol) maintains minimal state for the pool and emits `Mint`, `Burn`, and `Swap` events. If we wanted to know the cumulative number and volume of swaps on the pair, we could write and deploy a custom indexer instead of adding additional state variables and computation to the contract.

Smart contract indexers typically produce object schemas using the [GraphQL](https://graphql.org/) schema language.

Smart contract indexing services usually provide a hosted service so that users can deploy their indexers without having to run their own infrastructure.


## Provider Summary

The following providers support Monad Testnet:

<table>
  <thead>
    <tr>
      <th>Provider</th>
      <th>Docs</th>
      <th>Language</th>
      <th>Framework</th>
      <th>Known for</th>
      <th>Hosted service</th>
      <th>Decen- tralized hosted service</th>
      <th>Onchain & offchain data</th>
      <th>Web- socket subscr- iptions</th>
      <th>Query layer</th>
    </tr>
  </thead>
  <tbody>
    
    <tr>
      <td>[Alchemy](https://www.alchemy.com/subgraphs)</td>
      <td>[Docs](https://docs.alchemy.com/reference/subgraphs-quickstart)</td>
      <td>Assembly-Script</td>
      <td>[subgraph](https://github.com/graphprotocol/graph-node)</td>
      <td>Uptime guarantee</td>
      <td>✅</td>
      <td>❌</td>
      <td>❌</td>
      <td>❌</td>
      <td>GraphQL</td>
    </tr>

    
    <tr>
      <td>[DipDup](https://dipdup.io/)</td>
      <td>[Docs](https://dipdup.io/docs/quickstart-evm)</td>
      <td>Python</td>
      <td>[dipdup](https://github.com/dipdup-io/dipdup)</td>
      <td>Python development</td>
      <td>❌</td>
      <td>❌</td>
      <td>✅</td>
      <td>✅</td>
      <td>GraphQL</td>
    </tr>

    
    <tr>
      <td>[Envio](https://envio.dev/)</td>
      <td>[Docs](https://docs.envio.dev/docs/HyperIndex/overview)</td>
      <td>JavaScript, TypeScript, Rescript</td>
      <td>[HyperIndex](https://github.com/enviodev/hyperindex)</td>
      <td>Performance and scale</td> 
      <td>✅</td>
      <td>❌</td>
      <td>✅</td>
      <td>✅</td>
      <td>GraphQL</td>
    </tr>

    
    <tr>
      <td>[Ghost](https://tryghost.xyz/)</td>
      <td>[Docs](https://docs.tryghost.xyz/ghostgraph/overview)</td>
      <td>Solidity</td>
      <td>GhostGraph</td>
      <td>Solidity development</td>
      <td>✅</td>
      <td>❌</td>
      <td>❌</td>
      <td>❌</td>
      <td>GraphQL</td>
    </tr>

    
    <tr>
      <td>[Goldsky](https://goldsky.com/)</td>
      <td>[Docs](https://docs.goldsky.com/)</td>
      <td>Assembly- Script</td>
      <td>[subgraph](https://github.com/graphprotocol/graph-node)</td>
      <td></td>
      <td>✅</td>
      <td>❌</td>
      <td>❌</td>
      <td>❌</td>
      <td>Custom GraphQL</td>
    </tr>

    
    <tr>
      <td>[Sentio](https://www.sentio.xyz/)</td>
      <td>[Docs](https://docs.sentio.xyz/docs/quickstart)</td>
      <td>JavaScript, TypeScript</td>
      <td>[sentio-sdk](https://github.com/sentioxyz/sentio-sdk)</td>
      <td>Performance; integrated alerting and visualization</td>
      <td>✅</td>
      <td>❌</td>
      <td>✅</td>
      <td>❌</td>
      <td>GraphQL & SQL</td>
    </tr>
        
    
    <tr>
      <td>[SQD](https://sqd.ai/)</td>
      <td>[Docs](https://docs.sqd.ai)</td>
      <td>TypeScript</td>
      <td>[squid-sdk](https://github.com/subsquid/squid-sdk)</td>
      <td>Performance, decentralization</td>
      <td>✅</td>
      <td>Partial[^1]</td>
      <td>✅</td>
      <td>✅</td>
      <td>GraphQL</td>
    </tr>

    
    <tr>
      <td>[SubQuery](https://subquery.network/)</td>
      <td>[Docs](https://academy.subquery.network/)</td>
      <td>TypeScript</td>
      <td>[subql](https://github.com/subquery/subql)</td>
      <td>Decentral- ization</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>❌</td>
      <td>GraphQL</td>
    </tr>

    
    <tr>
      <td>[The Graph](https://thegraph.com/)</td>
      <td>[Docs](https://thegraph.com/docs/en/subgraphs/quick-start/)</td>
      <td>Assembly- Script</td>
      <td>[subgraph](https://github.com/graphprotocol/graph-node)</td>
      <td>The original indexer</td>
      <td>✅</td>
      <td>✅</td>
      <td>❌</td>
      <td>❌</td>
      <td>Custom GraphQL</td>
    </tr>
  </tbody>
</table>

[^1]: SQD hosted service is semi-decentralized: the data lake is decentralized, but indexers run on proprietary infra.

## Provider Details

### Alchemy

[Alchemy](https://www.alchemy.com/) is a popular API provider and developer platform. Alchemy offers subgraphs as a hosted service, as well as offerings in other categories including smart wallets, AA infra, NFT and token APIs, and RPC APIs.

Supported Networks
- Monad Testnet


### DipDup

[DipDup](https://dipdup.io/) is a Python framework for building smart contract indexers.

To get started, visit the [documentation](https://dipdup.io/docs/quickstart-evm).

Supported Networks
- Monad Testnet


### Envio

[Envio](https://envio.dev/) is a full-featured data indexing solution that provides application developers with a seamless and efficient way to index and aggregate real-time and historical blockchain data for Monad Testnet. The indexed data is easily accessible through custom GraphQL queries, providing developers with the flexibility and power to retrieve specific information.

[Envio HyperSync](https://docs.envio.dev/docs/HyperIndex/hypersync) is an indexed layer of the Monad Testnet blockchain for the hyper-speed syncing of historical data (JSON-RPC bypass). What would usually take hours to sync ~100,000 events can now be done in the order of less than a minute.

Designed to optimize the user experience, Envio offers automatic code generation, flexible language support, multi-chain data aggregation, and a reliable, cost-effective hosted service.


To get started, visit the [documentation](https://docs.envio.dev/docs/HyperIndex/overview) or follow the [quickstart](https://docs.envio.dev/docs/HyperIndex/contract-import) guide.

Supported Networks
- Monad Testnet


### Ghost

With [GhostGraph](https://tryghost.xyz/graph), you can write your indexers in the same language as your contracts: Solidity. This means less context switching and faster time to market.

To get started, visit the [documentation](https://docs.tryghost.xyz/ghostgraph/overview/) or check out the [tutorial](../../guides/indexers/ghost.md).

Services supported:
- GhostGraph

Supported Networks
- Monad Testnet


### Goldsky

[Goldsky](https://goldsky.com/) is the go-to data indexer for web3 builders, offering high-performance subgraph hosting and realtime data replication pipelines.

Goldsky offers two core self-serve products that can be used independently or in conjunction to power your data stack.

- **Subgraphs**: Flexible indexing with typescript, with support for webhooks and more.
- **Mirror**: Get live blockchain data in your database or message queues with a single yaml config.


To get started, visit the [documentation](https://docs.goldsky.com/) or follow the [quickstart](https://docs.goldsky.com/subgraphs/guides/create-a-no-code-subgraph) guide.

Supported Networks
- Monad Testnet


### Sentio

[Sentio](https://www.sentio.xyz/) offers blazing-fast native processors and seamless subgraph hosting on Monad. With powerful database capabilities, intuitive dashboards, and comprehensive API functionalities, Sentio is built to provide an exceptional developer experience.

To get started, check out the [docs](https://docs.sentio.xyz/docs/supported-networks#monad) or visit the [quickstart](https://docs.sentio.xyz/docs/quickstart) guide.

Supported Networks
- Monad Testnet


### SQD

[SQD](https://sqd.ai/) enables permissionless, cost-efficient access to petabytes of high-value Web3 data. 

SQD is a decentralized hyper-scalable data platform optimized for providing efficient, permissionless access to large volumes of data. It currently serves historical on-chain data, including event logs, transaction receipts, traces, and per-transaction state diffs.

To get started, visit the [documentation](https://docs.sqd.ai) or see this [quickstart](https://docs.sqd.ai/sdk/quickstart/) with [examples](https://docs.sqd.ai/sdk/examples) on how to easily create subgraphs via Subsquid.

Supported Networks
- Monad Testnet


### SubQuery

[SubQuery](https://subquery.network/) is a leading blockchain data indexer that provides developers with fast, flexible, universal, open source and decentralised APIs for web3 projects. SubQuery SDK allows developers to get rich indexed data and build intuitive and immersive decentralised applications in a faster and more efficient way. SubQuery supports many ecosystems including Monad, Ethereum, Cosmos, Near, Polygon, Polkadot, Algorand, and more.

One of SubQuery's advantages is the ability to aggregate data not only within a chain but across multiple blockchains all within a single project. This allows the creation of feature-rich dashboard analytics and multi-chain block scanners.

#### Useful resources:

- [SubQuery Academy (Documentation)](https://academy.subquery.network/)
- [Monad Testnet Starter](https://github.com/subquery/ethereum-subql-starter/tree/main/Monad/monad-testnet-starter)
- [Monad Testnet Quick Start Guide](https://academy.subquery.network/indexer/quickstart/quickstart_chains/monad.html)

For technical questions and support reach out to us `start@subquery.network`

Supported Networks
- Monad Testnet


### The Graph

[The Graph](https://thegraph.com/) is an indexing protocol that provides an easy way to query blockchain data through APIs known as subgraphs.

With The Graph, you can benefit from:

- **Decentralized Indexing**: Enables indexing blockchain data through multiple indexers, thus eliminating any single point of failure
- **GraphQL Queries**: Provides a powerful GraphQL interface for querying indexed data, making data retrieval super simple.
- **Customization**: Define your own logic for transforming & storing blockchain data. Reuse subgraphs published by other developers on The Graph Network.

Follow this [quick-start](https://thegraph.com/docs/en/subgraphs/quick-start/) guide to create, deploy, and query a subgraph within 5 minutes.

Supported Networks
- Monad Testnet

---

# Local Nodes

Developers often find it helpful to be able to run a 1-node Ethereum network with modified parameters to test interaction with the blockchain:

-   [Anvil](https://github.com/foundry-rs/foundry/tree/master/crates/anvil) is a local Ethereum node packaged in the Foundry toolkit
-   [Hardhat Network](https://hardhat.org/hardhat-network/docs/overview) is a local Ethereum node packaged in the Hardhat toolkit

Installation is most easily done as part of installing the respective toolkit, described in the next section.
---

# Oracles

Oracles make off-chain data accessible on chain.

## Definitions

| Term                             | Description                                                          |
| -------------------------------- | -------------------------------------------------------------------- |
| Push oracle                      | Provider regularly pushes price data to the oracle contract on chain |
| Pull (on-demand) oracle          | User triggers price data update while calling a smart contract       |
| Custom oracle                    | A custom calculator                                                  |
| VRF (Verifiable Random Function) | Provides random numbers on chain                                     |



## Provider Summary

The following providers support Monad Testnet:

<table>
  <tr>
    <th>Provider</th>
    <th>Docs</th>
    <th>Contract addresses</th>
    <th>Live data</th>
    <th>Support notes</th>
  </tr>

  
  <tr>
    <td>[Blocksense](https://blocksense.network/)</td>
    <td>[Docs](https://docs.blocksense.network/)</td>
    <td>Separate address per pair:<br />
      WMON/USDT: [0x23eBeaDD97b4211525bcCe79014Dc68DD7C45F04](https://testnet.monadexplorer.com/address/0x23eBeaDD97b4211525bcCe79014Dc68DD7C45F04) <br />
      WBTC: [0x0f970fB74F6030ebbf8Ede3a701f29544Ef0dcca](https://testnet.monadexplorer.com/address/0x0f970fB74F6030ebbf8Ede3a701f29544Ef0dcca) <br />
      USDT: [0x23b7c30C0a85dbb1153FC0A087838B95aE20797A](https://testnet.monadexplorer.com/address/0x23b7c30C0a85dbb1153FC0A087838B95aE20797A) <br />
      USDC: [0xc4675f86BE0bE2bf3c9Ac2829d9CeD6F6d36465A](https://testnet.monadexplorer.com/address/0xc4675f86BE0bE2bf3c9Ac2829d9CeD6F6d36465A) <br />
      DAI: [0x1fb84Cb39e818816E4Ea5498A36937Fd9FeaC0d0](https://testnet.monadexplorer.com/address/0x1fb84Cb39e818816E4Ea5498A36937Fd9FeaC0d0)</td>
    <td></td>
    <td>Push oracle;<br/>
    Custom oracles</td>
  </tr>

  
  <tr>
    <td>[Chainlink](https://chain.link/)</td>
    <td>[Docs](https://docs.chain.link/)</td>
    <td>Data stream verifier proxy address: [0xC539169910DE08D237Df0d73BcDa9074c787A4a1](https://testnet.monadexplorer.com/address/0xC539169910DE08D237Df0d73BcDa9074c787A4a1)</td>
    <td></td>
    <td>Pull oracle</td>
  </tr>

  
  <tr>
    <td>[ChainSight](https://chainsight.network/)</td>
    <td>[Docs](https://docs.chainsight.network/)</td>
    <td></td>
    <td></td>
    <td>Custom oracles. Migrating to a new version</td>
  </tr>

  
  <tr>
    <td>[Chronicle](https://chroniclelabs.org/)</td>
    <td>[Docs](https://docs.chroniclelabs.org/)</td>
    <td>[Address reference](https://docs.chroniclelabs.org/Developers/testnet)</td>
    <td>[Dashboard](https://chroniclelabs.org/dashboard/oracles?blockchain=MON-TESTNET) (toggle dev mode)</td>
    <td>Push oracle; custom oracles</td>
  </tr>

  
  <tr>
    <td>[Gelato VRF](https://docs.gelato.network/web3-services/vrf/quick-start)</td>
    <td>[Docs](https://docs.gelato.network/web3-services/vrf/quick-start)</td>
    <td></td>
    <td></td>
    <td>VRF</td>
  </tr>

  
  <tr>
    <td>[Orochi](https://www.orochi.network/)</td>
    <td>[Docs](https://docs.orochi.network/)</td>
    <td>[Orocle](https://docs.orochi.network/Orocle/testnet) [oracle] addresses<br/>
      [Orand](https://docs.orochi.network/Orand/testnet) [VRF] addresses</td>
    <td></td>
    <td>[zkOracle](https://docs.orochi.network/Orocle/introduction);<br/>
    [VRF](https://docs.orochi.network/Orand/introduction)</td>
  </tr>

  
  <tr>
    <td>[Pyth](https://www.pyth.network/)</td>
    <td>[Docs](https://docs.pyth.network/)</td>
    <td>Price feeds: [0x2880aB155794e7179c9eE2e38200202908C17B43](https://testnet.monadexplorer.com/address/0x2880aB155794e7179c9eE2e38200202908C17B43)<br/><br/>
    Beta price feeds (incl MON/USDC): [0xad2B52D2af1a9bD5c561894Cdd84f7505e1CD0B5](https://testnet.monadexplorer.com/address/0xad2B52D2af1a9bD5c561894Cdd84f7505e1CD0B5)<br/><br/>
    Entropy: [0x36825bf3Fbdf5a29E2d5148bfe7Dcf7B5639e320](https://testnet.monadexplorer.com/address/0x36825bf3Fbdf5a29E2d5148bfe7Dcf7B5639e320)</td>
    <td>[Live data](https://www.pyth.network/price-feeds)<br/><br/>
    [Beta live data](https://www.pyth.network/developers/price-feed-ids#beta) (includes MON/USDC)</td>
    <td>[Pull oracle](https://docs.pyth.network/price-feeds/pull-updates);<br/>
    [VRF](https://docs.pyth.network/entropy)</td>
  </tr>

  
  <tr>
    <td>[Redstone](https://www.redstone.finance/)</td>
    <td>[Docs](https://docs.redstone.finance/)</td>
    <td>Push oracle [addresses](https://app.redstone.finance/app/feeds/?networks=10143)<br/>
    Update conditions for all: 0.5% deviation & 6h heartbeat</td>
    <td>[Live data](https://app.redstone.finance/app/tokens/)</td>
    <td>[Push oracle](https://app.redstone.finance/app/feeds/?networks=10143);<br/>
    [pull oracle](https://app.redstone.finance/app/pull-model/redstone-primary-prod)</td>
  </tr>

  
  <tr>
    <td>[Stork](https://stork.network/)</td>
    <td>[Docs](https://docs.stork.network/)</td>
    <td>Pull oracle (includes MON/USD): [0xacC0a0cF13571d30B4b8637996F5D6D774d4fd62](https://testnet.monadexplorer.com/address/0xacC0a0cF13571d30B4b8637996F5D6D774d4fd62)<br/>
    [Addresses](https://docs.stork.network/resources/contract-addresses/evm); [APIs](https://docs.stork.network/api-reference/contract-apis/evm); [Asset ID Registry](https://docs.stork.network/resources/asset-id-registry)</td>
    <td></td>
    <td>[Pull oracle](https://docs.stork.network/introduction/core-concepts)</td>
  </tr>

  
  <tr>
    <td>[Supra](https://supra.com/)</td>
    <td>[Docs](https://docs.supra.com/)</td>
    <td>Push oracle: [0x6Cd59830AAD978446e6cc7f6cc173aF7656Fb917](https://testnet.monadexplorer.com/address/0x6Cd59830AAD978446e6cc7f6cc173aF7656Fb917)<br/>
      (5% deviation threshold & 1h update frequency;<br/>
      Supported pairs: BTC/USDT, SOL/USDT, ETH/USDT)<br/><br/>
      Pull oracle: [0x443A0f4Da5d2fdC47de3eeD45Af41d399F0E5702](https://testnet.monadexplorer.com/address/0x443A0f4Da5d2fdC47de3eeD45Af41d399F0E5702)<br/><br/>
      dVRF: [0x6D46C098996AD584c9C40D6b4771680f54cE3726](https://testnet.monadexplorer.com/address/0x6D46C098996AD584c9C40D6b4771680f54cE3726)</td>
    <td>[Live data](https://supra.com/data)</td>
    <td>[Push oracle](https://docs.supra.com/oracles/data-feeds/push-oracle);<br/>
      [Pull oracle](https://docs.supra.com/oracles/data-feeds/pull-oracle);<br/>
      [dVRF](https://docs.supra.com/oracles/dvrf)</td>
  </tr>

  
  <tr>
    <td>[Switchboard](https://switchboard.xyz/)</td>
    <td>[Docs](https://docs.switchboard.xyz/)</td>
    <td>Pull oracle: [0x33A5066f65f66161bEb3f827A3e40fce7d7A2e6C](https://testnet.monadexplorer.com/address/0x33A5066f65f66161bEb3f827A3e40fce7d7A2e6C)<br/><br/>
    More info: [Deployments](https://docs.switchboard.xyz/product-documentation/data-feeds/evm)</td>
    <td>[Live data](https://ondemand.switchboard.xyz/monad/testnet)</td>
    <td>[Pull oracle](https://docs.switchboard.xyz/product-documentation/data-feeds/evm);<br/>
    [Oracle aggregator](https://docs.switchboard.xyz/product-documentation/aggregator);<br/>
    [VRF](https://docs.switchboard.xyz/product-documentation/randomness)</td>
  </tr>
</table>


## Provider Details

### Blocksense

[Blocksense](https://blocksense.network/)'s programmable oracles let you access high performance price feeds plus any custom internet data, using familiar DeFi-compatible interfaces, all while enjoying super low costs powered by ZK technology.

To get started, check out the [documentation](https://docs.blocksense.network/) or reach out to the team on [Discord](https://discord.com/invite/mYujUXwrMr).

Supported Networks:
- Monad Testnet


### Chainlink

[Chainlink](https://chain.link/) Data Streams deliver low-latency market data offchain, which can be verified onchain. This approach provides decentralized applications (dApps) with on-demand access to high-frequency market data backed by decentralized, fault-tolerant, and transparent infrastructure.

Traditional push-based oracles update onchain data at set intervals or when certain price thresholds are met. In contrast, Chainlink Data Streams uses a pull-based design that preserves trust-minimization with onchain verification.

To get started, check out the [documentation](https://docs.chain.link/data-streams).

Supported Networks:
- Monad Testnet



### ChainSight

[Chainsight](https://chainsight.network/) redefines oracles with no-code tools, lowering costs, reducing single-operator risks, and driving scalable, open innovation.

To get started, check out the [documentation](https://docs.chainsight.network/).

Supported Networks:
- Monad Testnet


### Chronicle

Chronicle's decentralized oracle network was originally built within MakerDAO for the development of DAI and is now available to builders on Monad.

- **Data Feeds**: Builders can choose from 90+ data feeds, including crypto assets, yield rates, and RWAs. Chronicle's data is sourced via custom-built data models, only utilizing Tier 1 sources.
- **Transparency & Integrity**: Chronicle's oracle network is fully transparent and verifiable via the [Chronicle dashboard](https://chroniclelabs.org/dashboard/oracles?blockchain=MON-TESTNET). Users can cryptographically challenge the integrity of every oracle update using the 'verify' feature. Data is independently sourced by a [community of Validators](https://chroniclelabs.org/validators) including Gitcoin, Etherscan, Infura, DeFi Saver, and MakerDAO.
- **Gas Efficiency**: Pioneering the Schnorr-based oracle architecture, Chronicle's oracles use 60-80% less gas per update than other oracle providers. This lowest cost per update allows Push oracle updates to be made more frequently, enabling granular data reporting.
- Every oracle implementation is customized to fit your needs. Implement one of our existing data models or contact Chronicle to develop custom oracle data feeds via [Discord](https://discord.gg/CjgvJ9EspJ).

Developers can dive deeper into Chronicle Protocol's architecture and unique design choices via the [docs](https://docs.chroniclelabs.org/).

Supported Networks:
- Monad Testnet



### Gelato VRF

[Gelato VRF](https://docs.gelato.network/web3-services/vrf) (Verifiable Random Function) provides a unique system offering trustable randomness on Monad Testnet.

See [this](https://docs.gelato.network/web3-services/vrf/quick-start) guide to learn how to get started with Gelato VRF.

Supported Networks:
- Monad Testnet

### Orochi

[Orochi Network](https://www.orochi.network/) is the world’s first Verifiable Data Infrastructure, addressing scalability, privacy, and data integrity challenges.

To get started, visit the [Orochi documentation](https://docs.orochi.network/orochi-network/orand-orocle.html).

Supported Networks:
- Monad Testnet


### Pyth
The [Pyth Network](https://www.pyth.network/) is one of the largest first-party oracle networks, delivering real-time data across a number of chains. Pyth introduces a low-latency [pull oracle](https://docs.pyth.network/price-feeds/pull-updates) design. Data providers push price updates to [Pythnet](https://docs.pyth.network/price-feeds/how-pyth-works/pythnet) every 400 ms. Users pull aggregated prices from Pythnet onto Monad when needed, enabling everyone in the onchain environment to access that data point most efficiently.

Pyth Price Feeds features:
- 400ms latency
- [First-party](https://www.pyth.network/publishers) data sourced directly from financial institutions
- [Price feeds](https://www.pyth.network/developers/price-feed-ids) ranging from crypto, stocks, FX, and metals
  - See also: [beta price feeds](https://www.pyth.network/developers/price-feed-ids#beta) (testnet MON/USD is a beta price feed)
- Available on [many](https://docs.pyth.network/price-feeds/contract-addresses) major chains

Contract Addresses for Monad Testnet:
- Price feeds: [0x2880aB155794e7179c9eE2e38200202908C17B43](https://testnet.monadexplorer.com/address/0x2880aB155794e7179c9eE2e38200202908C17B43)
  - Beta price feeds: [0xad2B52D2af1a9bD5c561894Cdd84f7505e1CD0B5](https://testnet.monadexplorer.com/address/0xad2B52D2af1a9bD5c561894Cdd84f7505e1CD0B5) (testnet MON/USD is a beta price feed)
- Entropy: [0x36825bf3Fbdf5a29E2d5148bfe7Dcf7B5639e320](https://testnet.monadexplorer.com/address/0x36825bf3Fbdf5a29E2d5148bfe7Dcf7B5639e320)

Supported Networks:
- Monad Testnet

:::note
The testnet `MON/USD` price feed is currently a beta feed on Pyth Network. To use the MON/USD feed, integrate the [beta price feed](https://testnet.monadexplorer.com/address/0xad2B52D2af1a9bD5c561894Cdd84f7505e1CD0B5) contract instead of the primary price feed contract.

To get the MON/USD price feed offchain, use the beta hermes endpoint: [https://hermes-beta.pyth.network](https://hermes-beta.pyth.network)
:::

### Redstone

[RedStone](https://www.redstone.finance/) is the fastest-growing modular oracle, specializing in yield-bearing collateral for lending markets, such as LSTs, LRTs and BTCFi.

To get started, visit the [Redstone documentation](https://docs.redstone.finance/docs/introduction).

Supported Networks:
- Monad Testnet


### Stork

[Stork](https://stork.network/) is an oracle protocol that enables ultra low latency connections between data providers and both on and off-chain applications. The most common use-case for Stork is pulling and consuming market data in the form of real time price feeds for DeFi.

Stork is implemented as a [pull oracle](https://docs.stork.network/introduction/core-concepts#docs-internal-guid-4b312e7b-7fff-1147-c04b-bbaadec1a82a). Stork continuously aggregates, verifies, and audits data from trusted publishers, and makes that aggregated data available at sub-second latency and frequency. This data can then be pulled into any on or off-chain application as often as needed.

To learn more about how Stork works, visit [Core Concepts](https://docs.stork.network/introduction/core-concepts) and [How It Works](https://docs.stork.network/introduction/how-it-works).

Supported Networks:
- Monad Testnet

### Supra

[Supra](https://supra.com) provides VRF and decentralized oracle price feeds (push and pull based) that can be used for onchain and offchain use-cases such as spot and perpetual DEXes, lending protocols, and payments protocols. 

To get started, visit the [Supra documentation](https://docs.supra.com)

Supported Networks:
- Monad Testnet

### Switchboard

[Switchboard](https://switchboard.xyz/) is a customizable oracle network and oracle aggregator.

To get started, visit the [Switchboard documentation](https://docs.switchboard.xyz/introduction).

Supported Networks:
- Monad Testnet
---

# RPC Providers

A number of RPC providers are supporting Monad testnet. 

See also: [API reference](/reference/json-rpc)

## Alchemy

[Alchemy](https://www.alchemy.com/) is a popular API provider and developer platform. Its robust, free tier offers access to JSON-RPC APIs, and hosted testnet nodes for Monad Testnet.

Supported Networks 
- Monad Testnet

## Blockdaemon

[Blockdaemon](https://www.blockdaemon.com/) provides enterprise-grade web3 infrastructure, including dedicated nodes, APIs, staking, liquid staking, MPC wallets, and more.

To get started, visit the [Blockdaemon documentation](https://docs.blockdaemon.com/).

Supported Networks
- Monad Testnet

## dRPC

[dRPC](https://drpc.org/) is an off-chain routing protocol for delivering reliable API infrastructure leveraging a distributed network of nodes.

To get started, visit the [dRPC documentation](https://drpc.org/docs).

Supported Networks
- Monad Testnet

## Envio

[Envio](https://envio.dev) has a free read only RPC that supports a subset of data intensive [methods](https://docs.envio.dev/docs/HyperSync/overview-hyperrpc#supported-methods), Envio's purpose built rust node supports historical data allowing you to query past 10,000 blocks into the past.

To get started, visit the [Envio documentation](https://docs.envio.dev/docs/HyperSync/overview-hyperrpc)

Supported Networks
- Monad Testnet

## QuickNode

[QuickNode](https://www.quicknode.com/) offers access to their [Core RPC API](https://www.quicknode.com/core-api) for Monad Testnet.

:::tip
QuickNode is offering discounts for projects building on Monad, more details [here](https://quicknode.notion.site/QuickNode-Benefits-for-Monad-Developers-18215a82e84c80e6a322d2174d6a1a26)!
:::

Supported Networks
- Monad Testnet


## Thirdweb

[thirdweb](https://thirdweb.com/)'s [RPC Edge](https://portal.thirdweb.com/infrastructure/rpc-edge/overview) provides an RPC endpoint for developers building on Monad.

To get started, visit the [Thirdweb RPC Edge documentation](https://portal.thirdweb.com/infrastructure/rpc-edge/overview).

Supported Networks
- Monad Testnet


## Triton One

[Triton One](https://triton.one/triton-rpc/) provides a robust RPC service optimized for front-end GUIs across multiple blockchains including Monad. Triton RPC weeds out the bots so your users can actually use your product reliably, at scale.

Supported Networks
- Monad Testnet
---

# Toolkits

Developers often find it helpful to build their project in the context of a broader framework that organizes external dependencies (i.e. package management), organizes unit and integration tests, defines a deployment procedure (against local nodes, testnet, and mainnet), records gas costs, etc.

Here are the two most popular toolkits for Solidity development:

-   [Foundry](https://book.getfoundry.sh/) is a Solidity framework for both development and testing. Foundry manages your dependencies, compiles your project, runs tests, deploys, and lets you interact with the chain from the command-line and via Solidity scripts. Foundry users typically write their smart contracts and tests in the Solidity language.
-   [Hardhat](https://hardhat.org/docs) is a Solidity development framework paired with a JavaScript testing framework. It allows for similar functionality as Foundry, and was the dominant toolchain for EVM developers prior to Foundry.
---

# Wallets

A number of wallets are compatible with Monad testnet.

## Provider Summary

<table>
  <thead>
    <tr>
      <th>Wallet</th>
      <th>Available on</th>
      <th>Blinks Support</th>
      <th>Autodetect Tokens and NFTs</th>
      <th>AA</th>
      <th>Other Features</th>
    </tr>
  </thead>
  <tbody>
    
    <tr>
      <td>[Phantom](https://phantom.com)</td>
      <td>Desktop, Mobile</td>
      <td>✅</td>
      <td>Tokens, NFTs</td>
      <td>❌</td>
      <td>NFT support, DApp browser, Token swaps, cross-chain swaps, staking options</td>
    </tr>
    
    <tr>
      <td>[Backpack](https://backpack.app)</td>
      <td>Desktop, Mobile</td>
      <td>✅</td>
      <td>Tokens, NFTs</td>
      <td>❌</td>
      <td>DApp browser, Built-in exchange, Futures trading, Portfolio tracking</td>
    </tr>
    
    <tr>
      <td>[Bitget Wallet](https://web3.bitget.com/en?source=bitget)</td>
      <td>Desktop, Mobile</td>
      <td>❌</td>
      <td>❌</td>
      <td>❌</td>
      <td>DApp browser, NFT Market, DApp Browser, and Launchpad</td>
    </tr>
    
    <tr>
      <td>[HaHa](https://www.haha.me)</td>
      <td>Desktop, Mobile</td>
      <td>❌</td>
      <td>❌</td>
      <td>✅</td>
      <td>DeFi integrations, AA, Monad Native, trading, hardware wallet support</td>
    </tr>
    
    <tr>
      <td>[Leap](https://www.leapwallet.io)</td>
      <td>Desktop, Mobile</td>
      <td>❌</td>
      <td>NFTs</td>
      <td>❌</td>
      <td>Portfolio tracking, Open source, Cross-chain swaps, Staking</td>
    </tr>
    
    <tr>
      <td>[Nomas](https://nomaswallet.com)</td>
      <td>Mobile</td>
      <td>❌</td>
      <td>❌</td>
      <td>❌</td>
      <td>AI features, Gas fee optimization</td>
    </tr>
    
    <tr>
      <td>[MetaMask](https://metamask.io)</td>
      <td>Desktop, Mobile</td>
      <td>❌</td>
      <td>NFTs</td>
      <td>❌</td>
      <td>NFT support, DApp browser, Open source, Token swaps, portfolio tracking</td>
    </tr>
    
    <tr>
      <td>[OKX Wallet](https://www.okx.com/en-us/help/section/faq-web3-wallet)</td>
      <td>Desktop, Mobile</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>DApp browser, Portfolio tracking, Cross-chain swaps, Biometric Login</td>
    </tr>
  </tbody>
</table>

## Provider Details

### Phantom

[Phantom](https://phantom.com) is a secure and easy-to-use wallet for the Monad Testnet.

To get started, download the phantom wallet [here](https://phantom.com/download) or visit the [documentation](https://phantom.com/learn/developers).

Supported Networks
- Monad Testnet

### Backpack

[Backpack](https://backpack.app) is a next-level wallet and exchange. Buy tokens, trade futures, and explore on-chain apps—seamlessly and securely. 🎒

To get started, download the Backpack wallet [here](https://backpack.app) or visit the [documentation](https://docs.backpack.app/).

Supported Networks
- Monad Testnet

### Bitget Wallet

[Bitget Wallet](https://web3.bitget.com/en?source=bitget) is a non-custodial wallet with advanced multi-chain capabilities and powerful swap function

To get started, download the Bitget wallet [here](https://web3.bitget.com/en/wallet-download?type=2) or visit the [documentation](https://web3.bitget.com/en/docs/).

Supported Networks
- Monad Testnet

### HaHa Wallet

[HaHa](https://www.haha.me) is a next-gen smart wallet with DeFi capabilities.

To get started, download the HaHa wallet [here](https://www.haha.me).

Supported Networks
- Monad Testnet

### Leap Wallet

[Leap](https://www.leapwallet.io) is a multi-chain wallet spanning across Cosmos, EVM & Bitcoin.

To get started, download the Leap wallet [here](https://www.leapwallet.io/download) or visit the [documentation](https://docs.leapwallet.io).

Supported Networks
- Monad Testnet

### Nomas Wallet

[Nomas](https://nomaswallet.com) is a Web3 Wallet Evolution Powered by AI.

Get started with the Nomas wallet [here](https://t.me/nomas_wallet_dev_bot).

Supported Networks
- Monad Testnet

### MetaMask

[MetaMask](https://metamask.io) is a secure and easy-to-use wallet for the Monad Testnet.

To get started, download the MetaMask wallet [here](https://metamask.io/download) or visit the [documentation](https://docs.metamask.io).

Supported Networks
- Monad Testnet

### OKX Wallet

[OKX Wallet](https://www.okx.com/en-us/help/section/faq-web3-wallet) is your all-in-one gateway to the Web3 world.

To get started, download the OKX Wallet [here](https://chromewebstore.google.com/detail/okx-wallet/mcohilncbfahbmgdjkbpemcciiolgcge) or visit the [documentation](https://www.okx.com/web3/build/docs/sdks/okx-wallet-integration-introduction).

Supported Networks
- Monad Testnet