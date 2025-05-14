# Section 2: Building a Graph-Based RAG Agent with Neo4j and Fixed Path Cyphers

In this section, we will build a **Retrieval-Augmented Generation (RAG)** pipeline that uses a **knowledge graph** (Neo4j) instead of a traditional vector database for retrieval. We'll ingest a corpus of BBC news articles into Neo4j as a graph of connected concepts, then use **Fixed Path Cyphers** as determined by the data schema; note, these Cyphers are created `manually`. Using these specific Cyphers, we will retrieve articles of interest based on the Named Entity Recognition (NER) and then using these articles as the source of truth, use an LLM to find the answer within these articles.

By the end of this lab, you will have a working environment with Neo4j and a local LLM, a Neo4j graph populated with documents, categories, and key concepts (with relationships like `BELONGS_TO` and `MENTIONS`), and examples of querying the graph using natural language questions.

> **IMPORTANT:** All of the source code for this section can be found here:  
[https://github.com/davidvonthenen/2025-odsc-east-workshop/tree/main/LAPTOP/workshop/2_fixed_logic](https://github.com/davidvonthenen/2025-odsc-east-workshop/tree/main/LAPTOP/workshop/2_fixed_logic)

## Step 1: Environment Setup

> **IMPORTANT:** This environment step is exactly the same as in [Section 1: Building a Graph-Based RAG Agent with Neo4j and LLM-generated Cyphers](./STEP1_LLM_GENERATED.md). If you already have your `Neo4j` database running from the previous step, skip to [Step 3: Querying Using Fixed Cypher paths](#step-3-querying-using-fixed-cypher-paths).

## Step 2: Data Ingestion: From Raw Text to a Queryable Graph

If you already have your data ingested on an instance of `Neo4j`, continue on to [Step 3](#step-3-querying-using-fixed-cypher-paths). Otherwise, you can go back to [Section 1: Building a Graph-Based RAG Agent with Neo4j and LLM-generated Cyphers](./STEP1_LLM_GENERATED.md).

## Step 3: Querying Using Fixed Cypher Paths

Now, let's dive into a powerful and efficient alternative approach: using predefined, or **fixed Cypher paths**, to interact with your Neo4j graph database within a Retrieval-Augmented Generation (RAG) agent. Rather than dynamically generating queries on-the-fly with an LLM, this method involves setting up specific, fixed Cypher queries optimized for common retrieval tasks. These carefully crafted queries ensure consistent performance, predictable results, and streamlined interactions with the database.

In this section, you'll learn how to:

* Define and utilize **fixed Cypher queries** designed for frequent and high-impact retrieval scenarios.
* Connect these fixed paths directly into your RAG Agent workflow, enabling rapid responses to user questions without the overhead of dynamic query generation.

By leveraging fixed Cypher paths, your RAG agent maintains efficiency, reduces complexity, and delivers reliable, lightning-fast answers tailored precisely to your application's needs.

### 3.1 Using Precise/Fixed Cypher Paths Has MANY Benefits

This code will perform the RAG Query using our Graph-based implementation on Neo4j. Based on the data ingested, we will ask the RAG Agent the following questions:

- What do these articles say about Ernie Wise?

This code can be found and executed from this location: [./workshop/2_fixed_logic/rag_query.py](https://github.com/davidvonthenen/2025-odsc-east-workshop/tree/main/LAPTOP/workshop/2_fixed_logic/rag_query.py).

```python
#!/usr/bin/env python3
"""
RAG pipeline (Neo4j + spaCy + Llama-cpp)
Model tested with TheBloke/neural-chat-7B-v3-3.Q4_K_M.gguf
"""

import os
from functools import lru_cache

import spacy
from neo4j import GraphDatabase
from llama_cpp import Llama

##############################################################################
# Neo4j connection details
##############################################################################

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4jneo4j")

##############################################################################
# Llama-cpp configuration
##############################################################################

# TODO: fix the home dir path
MODEL_PATH      = os.getenv("MODEL_PATH", "/Users/vonthd/models/neural-chat-7b-v3-3.Q4_K_M.gguf")
N_CTX           = int(os.getenv("LLAMA_CTX",        "32768"))
N_THREADS       = int(os.getenv("LLAMA_THREADS",    str(os.cpu_count() or 8)))
N_GPU_LAYERS    = int(os.getenv("LLAMA_GPU_LAYERS", "0"))   # 0 = CPU only
TEMPERATURE     = float(os.getenv("LLAMA_TEMP",     "0.2"))
TOP_P           = float(os.getenv("LLAMA_TOP_P",    "0.95"))
MAX_NEW_TOKENS  = int(os.getenv("LLAMA_MAX_TOKENS", "512"))


@lru_cache(maxsize=1)
def load_llm() -> Llama:
    """
    Load the GGUF model once, cache, and reuse.
    """
    print(f"Loading model from {MODEL_PATH} …")
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
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
# Graph query - fetch docs mentioning entities
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
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
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

    # Neo4j - fetch docs
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
```

Here are four key considerations when using and extending this RAG pipeline:

1. **Context Window & Token Limits**

   * You must stay within the your LLMs context length (`n_ctx`) and the `max_tokens` you request. If your combined document snippets exceed this window, the model may truncate earlier context, reducing answer quality. The larger the context window, the better!
   * Consider summarizing or chunking very long documents before passing them in.

2. **Secure, Efficient Cypher Queries**

   * Always use parameterized Cypher (as shown) to prevent injection attacks and leverage Neo4j's query planning.
   * If you add an `expiration` filter on `[:MENTIONS]`, embed timestamps as parameters so Neo4j can cache and reuse the query plan.

3. **NER Accuracy & Coverage**

   * The small spaCy model (`en_core_web_sm`) is fast but may miss or mislabel domain-specific entities. For technical content, consider a larger or custom-trained NER model.
   * Always normalize (e.g. `toLower()`) both your stored entity names and extracted entities to improve matching.

#### 3.2 Understanding the Fixed Cypher Path Queries

In contrast to LLM-generated queries, **we explicitly define every Cypher statement** up front, mapping each question pattern to a predetermined graph traversal. This "fixed path" approach highlights a different but equally powerful pattern:

* The **knowledge graph** still holds all facts and relationships explicitly (documents, entities, categories, etc.).
* The **developer** now acts as the translator, hand-crafting precise Cypher templates that retrieve exactly the intended nodes and relationships.

For example, consider a fixed query to fetch all documents mentioning a given entity:

```cypher
MATCH (d:Document)-[:MENTIONS]->(e:Entity {name: $entityName})
RETURN
  d.title      AS title,
  d.category   AS category,
  substring(d.content, 0, 200) AS snippet
LIMIT $limit
```

When executed in Python:

```python
fixed_query = """
MATCH (d:Document)-[:MENTIONS]->(e:Entity {name: $entityName})
RETURN d.title AS title, d.category AS category,
       substring(d.content,0,200) AS snippet
LIMIT $limit
"""
params = {"entityName": "quantum computing", "limit": 5}
results = session.run(fixed_query, **params)
for r in results:
    print(f'Title: "{r["title"]}" | Category: "{r["category"]}"\nSnippet: {r["snippet"]}…')
```

You might see output such as:

```
Title: "Quantum Leap in Computing"   | Category: "tech"
Snippet: "Researchers at Google have unveiled a new 72-qubit quantum processor…"…

Title: "Advances in Quantum Hardware" | Category: "science"
Snippet: "A team at IBM demonstrated error correction on a superconducting quantum…"…
```

This fixed-path method offers clear benefits:

* **Predictable Performance**: Neo4j can cache and optimize these well-known queries.
* **Deterministic Results**: Each question corresponds to a single, auditable Cypher template.
* **Easier Maintenance & Security**: Queries are defined in code, making it simpler to review, test, and secure against injection.

By combining fixed Cypher paths with RAG-style context assembly (or even alongside LLM-generated queries), you gain full control over both performance and flexibility in your graph-powered Q&A system.

## Section 3: Understanding Reinforcement Learning

Now that we understand how LLM Generated and Fixed Cypher Paths work when it comes to document retrieval, let's tackle the concept of Reinforcement Learning.

→ [Next Up: Understanding Reinforcement Learning](./STEP3_REINFORCEMENT_LEARNING.md) 
