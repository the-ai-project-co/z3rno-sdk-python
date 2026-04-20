# Z3rno Python SDK Examples

Working code examples demonstrating how to use the Z3rno Python SDK.

## Examples

| File | Description |
|------|-------------|
| [basic_usage.py](basic_usage.py) | Simple store, recall, update, and forget workflow. Covers all core SDK operations including batch store and memory history. |
| [langchain_memory.py](langchain_memory.py) | LangChain integration: a custom `ChatMessageHistory` backed by Z3rno. Shows how to persist and retrieve conversation messages using Z3rno as the storage layer. |
| [crewai_memory.py](crewai_memory.py) | CrewAI integration: a shared memory backend for multi-agent crews. Demonstrates how multiple agents can store and recall context from a common Z3rno memory pool. |

## Setup

1. Install the SDK:

   ```bash
   pip install z3rno
   ```

2. Set environment variables:

   ```bash
   export Z3RNO_BASE_URL="http://localhost:8000"
   export Z3RNO_API_KEY="z3rno_sk_..."
   ```

3. For framework examples, install the additional dependency:

   ```bash
   # LangChain example
   pip install langchain-core

   # CrewAI example
   pip install crewai
   ```

4. Run an example:

   ```bash
   python examples/basic_usage.py
   ```
