#!/usr/bin/env python3

import os
import sys
import time
from pathlib import Path

from langchain_community.graphs import Neo4jGraph
from langchain_community.llms import LlamaCpp
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate

# ────────────────────────────── Neo4j Settings ───────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "neo4jneo4j")

# ───────────────────────────── llama-cpp Settings ────────────────────────────
MODEL_PATH = str(Path.home() / "models" / "neural-chat-7b-v3-3.Q4_K_M.gguf")

# ─────────────────────────────────────────────────────────────────────────────
# Prompt template for generating Cypher queries only. We are instructing the LLM
# to return a single valid Cypher query.
# ─────────────────────────────────────────────────────────────────────────────
CYTHER_ONLY_PROMPT = PromptTemplate(
    input_variables=["schema", "query"],
    template=(
        "You are an expert in Neo4j Cypher.\n"
        "Graph schema:\n{schema}\n\n"
        "Given a natural-language question, return ONE valid Cypher query "
        "that answers it.\n"
        "Output **only** the Cypher query—no explanation, no labels, no "
        "markdown fences.\n\n"
        "{query}"
    ),
)

def wait_for_neo4j(uri: str, user: str, pwd: str, tries: int = 10, delay: int = 3):
    """
    Ping the DB until it responds to 'RETURN 1'. This ensures the DB is up
    before the script attempts to run queries.
    """
    for i in range(1, tries + 1):
        try:
            graph = Neo4jGraph(url=uri, username=user, password=pwd)
            graph.query("RETURN 1")
            print(f"✓ Neo4j is ready on {uri}")
            return
        except Exception as e:
            print(f"[{i}/{tries}] Neo4j not reachable ({e}); retrying in {delay}s")
            time.sleep(delay)

    sys.exit("Neo4j never came online. Exiting.")


def main():
    # 1) Ensure the Neo4j database is reachable
    wait_for_neo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASS)

    # 2) Load the graph (this uses the Neo4jGraph class from langchain_community)
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASS,
        # enhanced_schema=True tries to infer node labels/relationships automatically
        enhanced_schema=True,
    )
    print("Detected schema:\n", graph.schema, "\n")

    # 3) Load local llama-cpp model
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=32768,
        n_threads=8,
        temperature=0.2,
        top_p=0.95,
        repeat_penalty=1.2,
        verbose=False,
    )

    # 4) Build the Cypher-aware QA chain
    chain = GraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        cypher_prompt=CYTHER_ONLY_PROMPT,    # Our specialized prompt
        validate_cypher=True,                # Validate the query
        verbose=False,
        allow_dangerous_requests=True,       # Allows MERGE if needed
    )

    # 5) Example questions for demonstration
    questions = [
        "What are the top 5 most mentioned entities in these articles?",
    ]

    # 6) Perform queries and display results
    for q in questions:
        print(f"\nQ: {q}")
        result = chain.invoke({"query": q})["result"]  # key must be "query"
        print("A:", result)


# run the RAG query
main()
print("\n\n")
print("RAG Complete!")
