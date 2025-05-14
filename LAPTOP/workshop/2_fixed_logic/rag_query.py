#!/usr/bin/env python3
"""
RAG pipeline (Neo4j + spaCy + Llama-cpp)
Model tested with TheBloke/neural-chat-7B-v3-3.Q4_K_M.gguf
"""

import os
from pathlib import Path
from functools import lru_cache

import spacy
from neo4j import GraphDatabase
from llama_cpp import Llama   # pip install llama-cpp-python

##############################################################################
# Neo4j connection details
##############################################################################

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4jneo4j")

##############################################################################
# Llama-cpp configuration
##############################################################################

MODEL_PATH = str(Path.home() / "models" / "neural-chat-7b-v3-3.Q4_K_M.gguf")

@lru_cache(maxsize=1)
def load_llm() -> Llama:
    """
    Load the GGUF model once, cache, and reuse.
    """
    print(f"Loading model from {MODEL_PATH} …")
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=32768,
        n_threads=8,
        temperature=0.2,
        top_p=0.95,
        repeat_penalty=1.2,
        verbose=False,
        chat_format="chatml",  # Neural-Chat uses the ChatML template
    )


##############################################################################
# Neo4j helpers
##############################################################################

def connect_neo4j():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

##############################################################################
# spaCy Named-Entity Recognition
##############################################################################

def extract_entities_spacy(text, nlp):
    doc = nlp(text)
    return [(ent.text.strip(), ent.label_) for ent in doc.ents if len(ent.text.strip()) >= 3]

##############################################################################
# Graph query – fetch docs mentioning entities
##############################################################################

def fetch_documents_by_entities(session, entity_texts, top_k=5):
    if not entity_texts:
        return []

    query = """
    MATCH (d:Document)-[:MENTIONS]->(e:Entity)
    WHERE toLower(e.name) IN $entity_list
    WITH d, count(e) as matchingEntities
    ORDER BY matchingEntities DESC
    LIMIT $topK
    RETURN d.title AS title, d.content AS content,
           d.category AS category, matchingEntities
    """
    entity_list_lower = [txt.lower() for txt in entity_texts]

    results = session.run(query,
                          entity_list=entity_list_lower,
                          topK=top_k)

    docs = []
    for r in results:
        docs.append({
            "title":  r["title"],
            "content": r["content"],
            "category": r["category"],
            "match_count": r["matchingEntities"]
        })
    return docs

##############################################################################
# LLM-based answer generation
##############################################################################

def generate_answer(question: str, context: str) -> str:
    llm = load_llm()

    system_msg = "You are an expert assistant answering questions using the given context."
    user_prompt = (
        f"You are given the following context from multiple documents:\n"
        f"{context}\n\nQuestion: {question}\n\nProvide a concise answer."
    )

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        top_p=0.95,
        max_tokens=32768,
    )
    return response["choices"][0]["message"]["content"].strip()

##############################################################################
# Main
##############################################################################

if __name__ == "__main__":
    user_query = "What do these articles say about Ernie Wise?"
    print(f"User Query: {user_query}")

    # Load spaCy model once
    nlp = spacy.load("en_core_web_sm")

    # NER over user query
    recognized_entities = extract_entities_spacy(user_query, nlp)
    entity_texts = [ent[0] for ent in recognized_entities]
    print("Recognized entities:", recognized_entities)

    # Neo4j — fetch docs
    driver = connect_neo4j()
    with driver.session() as session:
        docs = fetch_documents_by_entities(session, entity_texts, top_k=5)

    # Build context
    combined_context = ""
    for doc in docs:
        snippet = doc["content"][:300].replace("\n", " ")
        combined_context += (
            f"\n---\nTitle: {doc['title']} | Category: {doc['category']}\n"
            f"Snippet: {snippet}...\n"
        )

    # Ask the model
    final_answer = generate_answer(user_query, combined_context)
    print("\nRAG-based Answer:\n", final_answer)
