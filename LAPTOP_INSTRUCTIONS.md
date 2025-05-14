# INSTRUCTIONS: Using Your Local Laptop for a Full Production-like Experience

If you opt for the full experience, you need to have a Linux/MacOS based operating system. You should have already installed the software components in the pre-workshop instructions and you can find those [Software Prerequisites](#software-prerequisites) below.

This may take a long time depending on the (conference) WiFi conneciton. You may option to use the [COLAB_INSTRUCTIONS.md](https://github.com/dvonthenen/2025-odsc-east-workshop/tree/main/COLAB_INSTRUCTIONS.md) instead.

## Software Prerequisites

- A Linux or Mac-based Developer’s Laptop with enough memory to run a database (like Neo4j or Milvus) plus Llama 3.3 8B (below).
  - Windows Users should use a VM or Cloud Instance
- Python Installed: version 3.10 or higher
- (Recommended) Using a miniconda or venv virtual environment
- Docker (Linux or MacOS) Installed: for running a local Neo4j instance
- Basic familiarity with shell operations

Docker images you should pre-pull in your environment:

- `docker image pull neo4j:5.26`

### LLM to pre-download:

This is the official one used in today's workshop:

- Intel's [neural-chat-7B-v3-3-GGUF](https://huggingface.co/TheBloke/neural-chat-7B-v3-3-GGUF)

OR

- Huggingface [bartowski/Meta-Llama-3-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf)

OR

- Alternatively, using [ollama](https://ollama.com/)
  - Llama 3B: [https://ollama.com/library/llama3:8b](https://ollama.com/library/llama3:8b)

## First Lesson

Once you have installed all the Prerequisite Software, move on to the first lesson: [Building a Graph-Based RAG Agent with Neo4j and LLM-generated Cyphers](./LAPTOP/STEP1_LLM_GENERATED.md).
