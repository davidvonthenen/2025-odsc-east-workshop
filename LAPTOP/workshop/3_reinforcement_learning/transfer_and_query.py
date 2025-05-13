#!/usr/bin/env python3
"""
transfer_and_query_demo.py
Unified short-/long-term memory with a single :Document label.
Adds a third option (“expire”) to force-expire short-term MENTIONS.
"""

import time
from pathlib import Path

import spacy
from neo4j import GraphDatabase
from llama_cpp import Llama

# ─── Configuration ─────────────────────────────────────────────────────────────
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "neo4jneo4j"

# Path to your GGUF model
MODEL_PATH = str(Path.home() / "models" / "neural-chat-7b-v3-3.Q4_K_M.gguf")

TECH_CHECK = [
    "How much did OpenAI pay for Windsurf?",
    "What is the status of the Apple Vision Pro?",
    "What is the revenue share agreement between OpenAI and Microsoft?",
    "What is Perplexity's new fund?",
    "What is the significance of DeepSeek-R2?"
]
# ───────────────────────────────────────────────────────────────────────────────


# ─── Neo4j Helpers ─────────────────────────────────────────────────────────────
def connect_neo4j():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def find_documents_with_unexpired_mentions(session):
    """
    Return documents whose :MENTIONS relationships still have a future expiration.
    """
    now = time.time()
    query = """
    MATCH (d:Document)-[m:MENTIONS]->(e:Entity)
    WHERE m.expiration IS NOT NULL AND m.expiration > $now
    WITH d, collect(DISTINCT e.name) AS entities
    RETURN d.doc_uuid AS uuid, d.content AS content, entities
    ORDER BY d.timestamp ASC
    """
    return [
        {"uuid": r["uuid"], "content": r["content"], "entities": r["entities"]}
        for r in session.run(query, now=now)
    ]


def promote_to_long_term(session, doc_uuid):
    """Remove expiration ⇒ promote to long-term."""
    session.run(
        """
        MATCH (d:Document {doc_uuid:$uuid})-[m:MENTIONS]->()
        REMOVE m.expiration
        """,
        uuid=doc_uuid,
    )
    print(f"Promoted {doc_uuid} to long-term (expiration removed).")


def force_expire(session, doc_uuid, seconds_ago=2 * 24 * 60 * 60):
    """Force-expire by setting expiration to NOW - 2 days (default)."""
    past = time.time() - seconds_ago
    session.run(
        """
        MATCH (d:Document {doc_uuid:$uuid})-[m:MENTIONS]->()
        SET m.expiration = $past
        """,
        uuid=doc_uuid,
        past=past,
    )
    print(f"Forced expiration on {doc_uuid} (set to 2 days ago).")


def fetch_documents_by_entities(session, entity_texts, top_k=5):
    """
    Retrieve docs where MENTIONS are unexpired or permanent.
    """
    if not entity_texts:
        return []

    now = time.time()
    entity_list = [t.lower() for t in entity_texts]

    query = """
    MATCH (d:Document)-[m:MENTIONS]->(e:Entity)
    WHERE toLower(e.name) IN $entity_list
      AND (m.expiration IS NULL OR m.expiration > $now)
    WITH d, count(e) AS matches
    ORDER BY matches DESC
    LIMIT $topK
    RETURN d.doc_uuid AS uuid, d.content AS content, matches
    """

    return [
        {"uuid": r["uuid"], "content": r["content"], "matches": r["matches"]}
        for r in session.run(query, entity_list=entity_list, now=now, topK=top_k)
    ]


# ─── LLM / NLP Helpers ─────────────────────────────────────────────────────────
def extract_entities(text, nlp):
    doc = nlp(text)
    return [ent.text.strip() for ent in doc.ents if len(ent.text.strip()) >= 3]


def generate_answer(llm, question, context):
    prompt = f"""You are given the following context from multiple documents:
{context}

Question: {question}

Answer:"""
    res = llm(prompt, max_tokens=2048, temperature=0.2, stop=["Answer:"])
    return res["choices"][0]["text"].strip()


# ─── Main Workflow ─────────────────────────────────────────────────────────────
def main():
    print("=== Transfer & Query Demo (single memory with 'expire' option) ===")

    # Load NLP & LLM
    nlp = spacy.load("en_core_web_sm")
    print("Loading local LLaMA model...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=32768,
        n_threads=8,
        temperature=0.2,
        top_p=0.95,
        repeat_penalty=1.2,
    )

    driver = connect_neo4j()
    with driver.session() as session:
        # 1. Review unexpired short-term docs
        docs = find_documents_with_unexpired_mentions(session)
        if not docs:
            print("No unexpired short-term documents found.")
        else:
            for d in docs:
                print(f"\nDocUUID: {d['uuid']}")
                print(f"Content: {d['content']}")
                print(f"Entities: {d['entities']}")
                choice = input(
                    "Remove expiration (promote to long-term)? "
                    "(yes/no/expire): "
                ).strip().lower()

                if choice == "yes":
                    promote_to_long_term(session, d["uuid"])
                elif choice == "expire":
                    force_expire(session, d["uuid"])
                else:
                    print("Leaving document unchanged.")

        # 2. Run RAG queries for the TECH_CHECK questions
        for idx, fact in enumerate(TECH_CHECK, start=1):
            print(f"\n=== RAG Query Test for Fact #{idx} ===")
            question = f"What do we know related to: \"{fact}\"?"

            entity_texts = extract_entities(question, nlp)
            docs = fetch_documents_by_entities(session, entity_texts, top_k=5)
            if not docs:
                print("No documents found for this query.")
                continue

            # Build context
            combined_context = ""
            for doc in docs:
                snippet = doc["content"][:200].replace("\n", " ")
                combined_context += (
                    f"\n---\nDocUUID: {doc['uuid']}\nSnippet: {snippet}...\n"
                )

            answer = generate_answer(llm, question, combined_context)
            print(f"Question: {question}")
            print(f"Answer: {answer}")

    driver.close()
    print("=== Demo Complete ===")


if __name__ == "__main__":
    main()
