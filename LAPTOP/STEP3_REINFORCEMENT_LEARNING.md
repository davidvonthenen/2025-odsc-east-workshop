# Section 3: Implementing Reinforcement Learning Using Graph-based RAG

In this section, we will introduce the concept of **reinforcement learning** and explore how it enhances the capabilities of Retrieval-Augmented Generation (RAG) systems. Reinforcement learning involves training models to improve their performance based on feedback from interactions, enabling them to adapt and refine their behavior over time. Here, we will leverage fixed Cypher paths to streamline interactions with a Neo4j knowledge graph, ensuring consistency and efficiency in retrieving data.

By combining reinforcement learning with Graph-based RAG, you'll learn how to build adaptive and intelligent agents capable of delivering precise and timely insights.

> **IMPORTANT:** All of the source code for this section can be found here:  
[https://github.com/davidvonthenen/2025-odsc-east-workshop/tree/main/LAPTOP/workshop/3_reinforcement_learning](https://github.com/davidvonthenen/2025-odsc-east-workshop/tree/main/LAPTOP/workshop/3_reinforcement_learning)

## Step 1: Environment Setup

> **IMPORTANT:** This environment step is exactly the same as in [Section 1: Building a Graph-Based RAG Agent with Neo4j and LLM-generated Cyphers](./STEP1_LLM_GENERATED.md). If you already have your `Neo4j` database running from the previous step, skip to [Step 3: Implementation of Reinforcement Using Graph-based RAG](#step-3-implementation-of-reinforcement-using-graph-based-rag).

## Step 2: Data Ingestion: From Raw Text to a Queryable Graph

> **IMPORTANT:** If you already have your data ingested on an instance of `Neo4j`, continue on to [Step 3](#step-3-implementation-of-reinforcement-using-graph-rag). Otherwise, you can go back to [Section 1: Building a Graph-Based RAG Agent with Neo4j and LLM-generated Cyphers](./STEP1_LLM_GENERATED.md).

## Step 3: Implementation of Reinforcement Using Graph-based RAG

This reinforcement learning demonstration presents a series of five technology "facts" to the user and treats each "yes" response as a positive reward signal, storing the approved fact as a Neo4j `:Document` node and linking it to its named entities via `:MENTIONS` relationships that carry a `24-hour expiration timestamp` (short-term memory). Facts the user declines are simply skipped. Immediately afterward, the script runs a RAG-style retrieval: it extracts entities from a question derived from each fact, fetches all `:Document` nodes whose `:MENTIONS` relationships are either still unexpired or have never expired (long-term), concatenates document snippets into a context, and feeds that context plus question to a locally loaded LLaMA model (via llama-cpp-python) to generate an answer. Through this loop of interactive feedback and retrieval-augmented inference, the system "learns" which facts to retain and demonstrates how those preferences influence downstream Q&A.

### 3.1 Ingesting New Facts in Short-Term Memory

New facts are ingested as Document nodes linked to recognized Entity nodes via MENTIONS relationships stamped with a 24-hour expiration to represent short-term memory.

This code can be found and executed from this location: [./workshop/3_reinforcement_learning/reinforcement_learning.py](https://github.com/davidvonthenen/2025-odsc-east-workshop/tree/main/LAPTOP/workshop/3_reinforcement_learning/reinforcement_learning.py).

```python
#!/usr/bin/env python3

import time
import uuid
import spacy
from neo4j import GraphDatabase
from llama_cpp import Llama

# Neo4j connection settings
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4jneo4j"

# Path to your local LLaMA model file. Example: "models/ggml-model-q4_0.bin"
# TODO: fix the home dir path
LLAMA_MODEL_PATH = "/Users/vonthd/models/neural-chat-7b-v3-3.Q4_K_M.gguf"

# Five BBC-style technology facts
TECH_FACTS = [
    ...
]
TECH_CHECK =[
    "How much did OpenAI pay for Windsurf?",
    "What is the status of the Apple Vision Pro?",
    "What is the revenue share agreement between OpenAI and Microsoft?",
    "What is Perplexity's new fund?",
    "What is the significance of DeepSeek-R2?"
]

def connect_neo4j():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return driver

def setup_neo4j_schema(session):
    """
    Optional: Clear old documents and relationships if desired,
    for a fresh run. This will delete all :Document nodes and Entities.
    Comment out if you want to preserve prior data.
    """
    query = """
    MATCH (d:Document)
    DETACH DELETE d
    """
    session.run(query)
    
    query = """
    MATCH (e:Entity)
    DETACH DELETE e
    """
    session.run(query)

def insert_fact_with_expiration(session, fact_text, nlp, expiration_window_seconds=24*60*60):
    """
    Insert the fact as a :Document node. For each recognized entity, create
    a :MENTIONS relationship with an expiration time (now + expiration_window_seconds).
    """
    doc_uuid = str(uuid.uuid4())
    create_doc_query = """
    MERGE (d:Document {doc_uuid: $doc_uuid})
    ON CREATE SET
        d.content = $content,
        d.timestamp = timestamp()
    RETURN d
    """
    session.run(create_doc_query, doc_uuid=doc_uuid, content=fact_text)

    # Named Entity Recognition
    doc_spacy = nlp(fact_text)
    expiration_time = time.time() + expiration_window_seconds

    for ent in doc_spacy.ents:
        if len(ent.text.strip()) < 3:
            continue

        entity_uuid = str(uuid.uuid4())

        merge_entity_query = """
        MERGE (e:Entity {name: $name, label: $label})
        ON CREATE SET e.ent_uuid = $ent_uuid
        RETURN e
        """
        session.run(
            merge_entity_query,
            name=ent.text.strip(),
            label=ent.label_,
            ent_uuid=entity_uuid
        )

        # Create a short-term mention relationship with an expiration
        mentions_query = """
        MATCH (d:Document {doc_uuid: $docId})
        MATCH (e:Entity {ent_uuid: $entId})
        MERGE (d)-[m:MENTIONS]->(e)
        ON CREATE SET m.expiration = $expiration
        """
        session.run(
            mentions_query,
            docId=doc_uuid,
            entId=entity_uuid,
            expiration=expiration_time
        )

    return doc_uuid

def extract_entities_spacy(text, nlp):
    doc = nlp(text)
    return [(ent.text.strip(), ent.label_) for ent in doc.ents if len(ent.text.strip()) >= 3]

def fetch_documents_by_entities(session, entity_texts, top_k=5):
    """
    Fetch documents for which there is a :MENTIONS relationship *not expired*
    or having no expiration property. That is:
      - m.expiration IS NULL (long-term) OR m.expiration > now (unexpired short-term)
    Return up to top_k docs sorted by the count of matched entities.
    """
    if not entity_texts:
        return []

    entity_list_lower = [txt.lower() for txt in entity_texts]
    current_time = time.time()

    query = """
    MATCH (d:Document)-[m:MENTIONS]->(e:Entity)
    WHERE toLower(e.name) IN $entity_list
      AND (m.expiration IS NULL OR m.expiration > $current_time)
    WITH d, count(e) AS matchingEntities
    ORDER BY matchingEntities DESC
    LIMIT $topK
    RETURN
        d.doc_uuid AS doc_uuid,
        d.content AS content,
        matchingEntities
    """
    results = session.run(query, entity_list=entity_list_lower, current_time=current_time, topK=top_k)
    
    docs = []
    for record in results:
        docs.append({
            "doc_uuid": record["doc_uuid"],
            "content": record["content"],
            "match_count": record["matchingEntities"]
        })
    return docs

def generate_answer(llm, question, context):
    """
    Generates an answer using llama-cpp-python.
    """
    prompt = f"""You are given the following context from multiple documents:
{context}

Question: {question}

Please provide a concise answer.
Answer:
"""
    output = llm(
        prompt,
        max_tokens=1024,
        temperature=0.2,
        stop=["Answer:"]
    )
    return output["choices"][0]["text"].strip()

def main():
    print("=== Reinforcement Learning Demo (Single Mechanism for Memory) ===")

    # Load spaCy
    nlp = spacy.load("en_core_web_sm")

    # Load LLaMA model
    print("Loading local LLaMA model; please wait...")
    llm = Llama(
        model_path=LLAMA_MODEL_PATH,
        n_ctx=32768,
        n_threads=8,
        temperature=0.2,
        top_p=0.95,
        repeat_penalty=1.2,
    )

    driver = connect_neo4j()
    with driver.session() as session:
        # Optional: Clear existing data
        setup_neo4j_schema(session)

        # Store or skip each fact
        stored_fact_uuids = []
        for fact in TECH_FACTS:
            print("\nNew Fact Detected:")
            print(f" -> {fact}")
            decision = input("Store this fact for 24 hours? (yes/no): ").strip().lower()

            if decision == "yes":
                doc_uuid = insert_fact_with_expiration(session, fact, nlp)
                stored_fact_uuids.append(doc_uuid)
                print(f"Stored with doc_uuid {doc_uuid}\n")
            else:
                print("Skipped storing fact.\n")

        # Now let's do a RAG query test for each fact
        for idx, fact in enumerate(TECH_CHECK, start=1):
            print(f"\n=== RAG Query Test for Fact #{idx} ===")
            question = f"What do we know related to: \"{fact}\"?"
            recognized_entities = extract_entities_spacy(question, nlp)
            entity_texts = [ent[0] for ent in recognized_entities]

            docs = fetch_documents_by_entities(session, entity_texts, top_k=5)
            if not docs:
                print("No documents found for this query.")
                continue

            # Build context
            combined_context = ""
            for doc in docs:
                snippet = doc["content"][:200].replace("\n", " ")
                combined_context += f"\n---\nDocUUID: {doc['doc_uuid']}\nSnippet: {snippet}...\n"

            final_answer = generate_answer(llm, question, combined_context)
            print(f"Question: {question}")
            print(f"Answer: {final_answer}")

    driver.close()
    print("=== Demo Complete ===")

if __name__ == "__main__":
    main()
```

Below are the key considerations you'll want to keep in mind when running, extending, or hardening this RAG-style reinforcement demo:

1. **Memory & Expiration Logic**

   * **24-hour TTL**: By default `expiration = now + 24h`. You can parameterize `expiration_window_seconds` for shorter or longer short-term memory.
   * **Automatic pruning**: Relationships past their `expiration` won't be returned by your RAG query - but they still live in the graph for auditing. If you prefer automatic cleanup, consider Neo4j's `apoc.periodic` jobs or [TTL Procedures](https://neo4j.com/labs/apoc/4.4/temporal/ttl/) to delete or archive expired edges.

2. **NER Quality & Granularity**

   * **Model choice**: `en_core_web_sm` is lightweight but misses many fine-grained entities. For technical facts you may want `en_core_web_trf` or a custom fine-tuned model.
   * **Filtering**: You skip entities shorter than three characters - but you may also want to filter out numeric entities or overly generic terms.

3. **Auditability & Data Retention**

   * Every `:Document` and `:Entity` persists forever - only `m.expiration` changes. If privacy or storage is a concern, plan a downstream archival process.
   * You might also add a `createdBy` or `source` property to track provenance of each fact.

Keep these points in mind as you test and evolve the code. They'll help you maintain performance, accuracy, and a clear audit trail as your RAG agent navigates between ephemeral and enduring knowledge.

## Step 4: Short-Term and Long-Term Memory

Short-term memory temporarily records new facts with an expiration timestamp, whereas long-term memory endures by removing that expiration flag to preserve knowledge permanently.

### 4.1 Promoting Data from Short-Term to Long-Term Memory Through Reinforcement Learning

This code can be found and executed from this location: [./workshop/3_reinforcement_learning/transfer_and_query.py](https://github.com/davidvonthenen/2025-odsc-east-workshop/tree/main/LAPTOP/workshop/3_reinforcement_learning/transfer_and_query.py).

```python
#!/usr/bin/env python3
"""
transfer_and_query_demo.py
Unified short-/long-term memory with a single :Document label.
Adds a third option ("expire") to force-expire short-term MENTIONS.
"""

import time
import spacy
from neo4j import GraphDatabase
from llama_cpp import Llama

# ─── Configuration ─────────────────────────────────────────────────────────────
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "neo4jneo4j"

# Path to your GGUF model
# TODO: fix the home dir path
LLAMA_MODEL_PATH = "/Users/vonthd/models/neural-chat-7b-v3-3.Q4_K_M.gguf"

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
        model_path=LLAMA_MODEL_PATH,
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
```

Here's a curated checklist of critical considerations to keep this reinforcement-learning + RAG pipeline robust, performant, and maintainable:

1. **Expiration & Time Synchronization**

   * **Clock Skew**: All expiration logic relies on the host's system time. If your Neo4j server and application server clocks drift, short-term facts could expire prematurely (or hang around indefinitely). Consider NTP synchronization.
   * **Expiration Granularity**: Using `time.time()` (float seconds) vs. Neo4j's `timestamp()` (milliseconds) demands careful unit conversions - mismatches may filter out docs incorrectly.

2. **NER Coverage & Noise**

   * **Short Texts**: Very brief facts may yield no entities, so your RAG queries return empty context. You might want a fallback (e.g. keyword search) for "no-entity" cases.
   * **Entity Normalization**: spaCy may extract overlapping or partial entities ("OpenAI" vs. "OpenAI Inc."). Merging on `name` alone can fragment your graph. Consider lowercasing, stripping punctuation, or using a dedicated alias table.

3. **Session & Transaction Management**

   * **Batching vs. Per-Relationship Commits**: Each MERGE/CREATE currently runs in its own transaction. For bulk ingest or high throughput, wrap multiple operations in a single transaction to reduce overhead.
   * **Error Handling**: Uncaught exceptions (e.g., network blips, write conflicts) will crash the script. Consider try/except around sessions, with retries or circuit-breakers.

4. **Auditability & Data Retention**

   * **Historical Facts**: Once the 24h expiration lapses, mentions vanish from RAG queries. If you need to debug or audit, consider logging or archiving expired relationships rather than letting them slip away.
   * **Promotions**: If you ever remove expiration (promote to long-term), record that event (e.g., with a timestamp or provenance field) so you know why a document persisted.

5. **Edge Cases & Emergency Overrides**

    * **"No Entities Found"**: If a user question yields no entities, handle gracefully - perhaps by returning a canned message or falling back to "search all unexpired docs."
    * **Forced Expiration**: Be mindful that setting expiration two days in the past effectively hides docs forever. If that happens by accident, you'll need a manual override to restore them.

By keeping these points in view, you'll avoid the classic "it worked on my laptop" pitfalls and ensure your unified memory + RAG system remains reliable, performant, and secure in production.

### 4.2: Understanding How Reinforcement Learning Works: Vector vs Graph

Imagine teaching an AI system what's "important" and what's "forgettable," not unlike deciding which jokes survive at your next stand-up set. In RAG (Retrieval-Augmented Generation) pipelines, "reinforcement learning" isn't just about reward functions - it can also describe how our system **reinforces** or **expires** knowledge. Two dominant paradigms emerge:

1. **Vector-based memory** (think embedding indexes in FAISS or Pinecone).
2. **Graph-based memory** (our Neo4j + spaCy solution with expiring `MENTIONS` edges).

Let's dive into how each handles the "yes/no" decision process, and why the graph approach offers fine-grained control over your AI's short-term and long-term facts.

#### Vector-Based Memory: The Quick and Dirty Cache

* **Mechanism**: Every new fact is converted into a fixed-length embedding, then appended to a vector index.
* **Reinforcement Analogy**: Accept a fact? Push its vector. Reject it? Don't.
* **Expiry**: Often you must delete vectors manually or retrain your index to "forget." There's no native timestamp on individual embeddings - you're juggling IDs, rebuilds, and hoping you tracked them properly.
* **Pros**: Blazing-fast similarity searches; turnkey solutions in FAISS, Annoy, Pinecone.
* **Cons**: Coarse control - deleting or promoting facts is a blunt operation. No built-in audit trail.

> **Real-world quip**: It's like tossing important receipts into a shredder when they expire - you can do it, but you have to remember which bin you used.

#### Graph-Based Memory: Precision and Auditability

Our Neo4j + spaCy + llama-cpp setup brings structure to the party:

1. **Document Nodes** (`:Document`): Every stored fact - be it short-term or long-term - lives here.

2. **Entity Nodes** (`:Entity`): Extracted by spaCy NER, these anchor the "who," "what," and "where."

3. **MENTIONS Relationships** (`:Document`→`Entity`):

   * **`expiration` property** (Unix timestamp):

     * **Short-term**: `expiration = now + 24h`.
     * **Long-term**: `expiration = NULL`.

   * **Promote to Long-Term**: simply **remove** the `expiration` field.
   * **Force-Expire**: set `expiration` to two days ago, effectively hiding it from any future queries.

4. **RAG Queries**:

   * Match only `MENTIONS` where `expiration IS NULL` **or** `expiration > now`.
   * Build context snippets and feed them to LLaMA via llama-cpp.

#### Why This Matters

* **Granular Control**: Pin-point which facts you want to expire or promote without touching the nodes themselves.
* **Audit Trail**: Documents and entities remain intact - perfect for compliance or later analysis.
* **Unified Retrieval**: Single Cypher query handles both short- and long-term facts seamlessly.

## BONUS Section: Investigating Agent2Agent Protocol

Now that we understand how Reinforcement Learning works in relation to Graph-based RAG implementations, you are finished with this workshop. If you are interested in learning more about [Google's Agent2Agent Protocol](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/), proceed to the next section.

→ [BONUS SECTION: Agent2Agent Protocol](./STEP4_AGENT_2_AGENT.md) 
