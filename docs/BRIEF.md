# Project Brief: Ethereum Fine-Tuning Cookbook

## Vision
Create an open-source cookbook that teaches developers how to fine-tune language models on blockchain data by building models that translate raw Ethereum transactions into multiple output formats, serving as both a practical transaction analysis tool and an educational resource for QLoRA/LoRA fine-tuning techniques.

## User Personas
### Primary User: Blockchain Developer
- **Role:** Developer building dApps, block explorers, or analytics tools
- **Needs:** Automated transaction interpretation for their applications and practical fine-tuning examples using real blockchain data
- **Pain Points:** Raw transaction data requires manual decoding, debugging failed transactions is time-consuming, and no clear examples exist for fine-tuning on blockchain data
- **Success:** Integrates transaction explanations into their products and successfully fine-tunes models for their specific use cases

### Secondary User: ML Engineer
- **Role:** Machine learning engineer with blockchain data or seeking fine-tuning experience
- **Needs:** Structured approach to domain-specific data preparation and reproducible fine-tuning workflow
- **Pain Points:** Unclear how to leverage abundant blockchain data for ML and lack of production-ready fine-tuning examples
- **Success:** Can prepare specialized datasets and apply QLoRA techniques to achieve measurable improvements

## Core Requirements
- [MVP] The system should extract structured intents from Ethereum transactions (action, assets, protocol, outcome)
- [MVP] The system should decode ETH transfers, ERC-20 operations, and Uniswap V2/V3 swaps
- [MVP] The system should provide extensible recipes for extracting training data from transaction receipts and logs
- [MVP] The system should include complete fine-tuning pipeline runnable on 16GB consumer GPUs
- [Post MVP] The system should support multiple output formats including semantic summaries, narratives, and flow descriptions
- [Post MVP] The system should explain multi-step DeFi operations including flash loans and arbitrage
- [Post MVP] The system should diagnose transaction failure reasons from revert data
- [Post MVP] The system should trace value flow through internal transfers and multicalls

## Success Metrics
1. Model achieves 90% accuracy on transaction amounts, addresses, and protocol identification
2. Generated descriptions maintain 60+ Flesch Reading Ease score
3. Complete fine-tuning process executes in under 4 hours on RTX 3060 (12GB VRAM)