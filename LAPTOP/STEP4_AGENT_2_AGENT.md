# Section 4: Hands-On Lab: Building Multi-Agent RAG Workflows with Agent2Agent

Agent2Agent (A2A) is an open, JSON-RPC-based protocol that treats every AI service - whether a heavyweight LLM or the tiniest Flask microservice - as an "agent" on an equal playing field. Instead of hard-coding HTTP endpoints or bespoke SDKs, agents advertise a single `/tasks` method and agree on well-typed request/response envelopes. The Coordinator in this demo abuses that elegance to juggle tasks between its peers without caring whether the peer is local, in a container, or sitting halfway across the planet.

Why does that matter for Retrieval-Augmented Generation (RAG)? In a classic RAG stack, the retrieval component, the reasoning LLM, and any post-processing code are usually glued together inside one chunky process. By shifting to an agentic mindset, each of those responsibilities becomes an autonomous service that can scale, fail, or evolve independently. Your vector search can run on a GPU, while your summarizer lives on a lightweight CPU box, and nothing breaks as long as both speak A2A.

Most importantly, A2A pushes the "function-calling" idea all the way out to the network. When the Coordinator's LLM decides it needs a summary or a direct answer, it **calls another agent** exactly as it would have called a local function. That alignment between LLM planning ("tool calls") and infrastructure ("A2A calls") keeps the control flow ergonomic for developers and completely transparent to the model.

## Step 1: Prerequisites

Before diving in, make sure the following prerequisites are satisfied:

* **!!!IMPORTANT!!!** **Python 3.12+** installed and active in a virtual environment.
* Local/Manual `pip install` of the `github.com/google/A2A` repo: `pip install -e .`
* An **OpenAI API key** exported as `OPENAI_API_KEY=<your_key>`.
* **Requirements installed**:

  ```bash
  pip install -r requirements.txt
  ```

  (The lab's `requirements.txt` pins Flask, Pydantic v2, OpenAI v1.x, and `requests`.);
* Four terminals (or tmux panes) available to run each microservice separately, or Docker if you prefer containerizing.
* Network ports **5000 - 5002** free on localhost (Coordinator on 5000, Summarizer 5001, Q&A 5002).

## Step 2: High-Level Demo Walk-Through

> **IMPORTANT:** All of the source code for this section can be found here:  
[https://github.com/davidvonthenen/2025-odsc-east-workshop/tree/main/LAPTOP/workshop/4_agent_2_agent](https://github.com/davidvonthenen/2025-odsc-east-workshop/tree/main/LAPTOP/workshop/4_agent_2_agent)

From the **client's perspective**, you simply supply two inputs - a text-based **context** and a related **question** - and receive back:

1. A concise summary of your provided context (if deemed necessary by the LLM).
2. A direct, targeted answer to your question, based strictly on that summarized context.

Behind the scenes, four modular Python-based agents coordinate seamlessly using the Agent2Agent (A2A) protocol, interacting over HTTP with strongly typed JSON-RPC requests. Let's dive deeper into their technical roles and interactions:

| Script                     | Technical Responsibilities & Function                                                                                                                                                                                                                                                                                                                                                                                                |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`qna_agent.py`**         | Flask service exposing `/tasks` (port 5002). Accepts a summarized text and question. Internally, it issues a targeted prompt to OpenAI's Chat Completion API, returning a precise response in JSON-RPC format via A2A.                                                                                                                                                                                                               |
| **`summarizer_agent.py`**  | Flask service exposing `/tasks` (port 5001). Receives lengthy context input, sends it as a summarization prompt to OpenAI, and returns a succinct, 2-3 sentence summary wrapped in JSON-RPC via A2A.                                                                                                                                                                                                                                 |
| **`coordinator_agent.py`** | Central orchestration service (port 5000). It employs OpenAI's client-side function-calling capability to dynamically choose whether to invoke the Summarizer Agent, the Q&A Agent, or both, by making HTTP requests formatted according to the Agent2Agent protocol. It does not parse or generate textual responses itself; instead, the Coordinator leverages the LLM's function-calling logic to precisely manage the workflow. |
| **`client.py`**            | A simple Python CLI that constructs a JSON-RPC request (context and question) and posts it directly to the Coordinator's `/tasks` endpoint. Upon receiving a JSON-RPC response, it neatly prints out both the summary (if returned) and the direct Q&A response.                                                                                                                                                                    |

The technical elegance here lies in the **Coordinator Agent**'s implementation of **client-side function calling**. The Coordinator delegates decision-making to an LLM using OpenAI's `tools` API, where each external agent is represented as a callable function. The LLM autonomously determines if summarization is required or if it should directly invoke the Q&A service, encoding these decisions as structured JSON-RPC calls dispatched via the A2A protocol. Consequently, complex orchestration logic is succinctly expressed through simple function definitions, greatly simplifying maintainability and scalability.

This modular approach ensures any future agents - such as specialized retrieval, verification, or multimodal rendering agents - can seamlessly integrate into the same LLM-driven orchestration loop simply by defining additional client-side functions. The combination of client-side function calling and the A2A protocol offers powerful flexibility and extensibility in constructing scalable agent-driven architectures.

## Conclusion

Decomposing RAG into A2A-compliant micro-agents unlocks horizontal scale and graceful degradation. Want to run eight summarizer replicas behind a load balancer while keeping a single heavyweight reasoning LLM? Go for it. Need to hot-patch the retrieval logic without redeploying the planner? Ship a new agent at a new port, update an environment variable, and the Coordinator won't miss a beat.

More broadly, this pattern lays the foundation for **agent swarms** where each capability - vector search, ranking, multimodal rendering, compliance checking - evolves independently yet plugs into the same planning loop. Architect once, swap in new brains forever. That's the power of Agent2Agent.
