# Ethereum Fine-Tuning Project Guidelines

## General Coding Standards

- Follow PEP 8 style guide for Python code
- Use type hints for all function parameters and return values
- Write clear and concise docstrings for all functions and classes
- Maximum line length of 120 characters
- Use Black formatter for code formatting

## Documentation and Context

Always use Context7 to retrieve current documentation when working with frameworks, libraries, or APIs rather than relying on training data. This applies to:
- Answering questions about APIs, frameworks, or libraries
- Implementing integrations with external APIs or services
- Writing code that uses third-party packages or SDKs
- Debugging or updating existing integrations

Automatically invoke the Context7 MCP tools without being asked to ensure you're using the most up-to-date documentation and best practices.

## Project-Specific Guidelines

- This project focuses on Ethereum transaction analysis and LLM fine-tuning
- When working with blockchain data, always consider gas optimization
- When working with dataset preparation, ensure data validation and quality checks
- Test all transaction decoders with sample data before deployment
- Use descriptive variable names that reflect Ethereum/blockchain concepts
