# Section 1: Building a Graph-Based RAG Agent with Neo4j and LLM-generated Cyphers

In this section, we will build a **Retrieval-Augmented Generation (RAG)** pipeline that uses a **knowledge graph** (Neo4j) instead of a traditional vector database for retrieval. We'll ingest a corpus of BBC news articles into Neo4j as a graph of connected entities, then use a local large language model (LLM) to translate natural language questions into Cypher queries with `LangChain`'s `GraphCypherQAChain`. The LLM will execute those queries on the graph and generate answers based on the results.

By the end of this lab, you will have a working environment with Neo4j and a local LLM, a Neo4j graph populated with documents, categories, and key entities (with relationships like `BELONGS_TO` and `MENTIONS`), and examples of querying the graph using natural language questions.

> **IMPORTANT:** All of the source code for this section can be found here:  
[https://github.com/davidvonthenen/2025-odsc-east-workshop/tree/main/LAPTOP/workshop/1_llm_cypher](https://github.com/davidvonthenen/2025-odsc-east-workshop/tree/main/LAPTOP/workshop/1_llm_cypher)

## Step 1: Environment Setup

To get started, we need to set up two main components of our environment: a Neo4j graph database and a local LLM for question-answering. We'll use **Docker** to run Neo4j, you will need to download an LLM (we'll provide some recommendations) and set up a Python environment for our code.

### 1.1 Launch Neo4j with Docker

First, spin up a Neo4j instance using Docker. We can use the official Neo4j image and expose the default ports (7474 for HTTP interface, 7687 for Bolt protocol). Below is a **Docker command** that starts a Neo4j instance:

```bash
docker run \
  -d \
  --publish=7474:7474 --publish=7687:7687 \
  --env NEO4J_AUTH=neo4j/neo4jneo4j \
  -v $HOME/neo4j/data:/data \
  -v $HOME/neo4j/logs:/logs \
  -v $HOME/neo4j/import:/var/lib/neo4j/import \
  -v $HOME/neo4j/plugins:/plugins \
  -e NEO4JLABS_PLUGINS='["apoc"]' \
  -e NEO4J_apoc_export_file_enabled=true \
  -e NEO4J_apoc_import_file_enabled=true \
  neo4j:5.26
```

This will download the Neo4j image (if not already) and start a Neo4j server in the background. The Neo4j database will be empty initially. You can verify it's running by opening the Neo4j Browser at **[http://localhost:7474](http://localhost:7474)** in your browser. Log in with the username `neo4j` and password `neo4jneo4j`. If you see the Neo4j UI, your database is up and ready.

> **IMPORTANT:** After first login, you will need to change your password. Change the password to `neo4jneo4j` as all of the Python application in this workshop use this new default password.

### 1.2 Python Environment and Dependencies

With Neo4j running and the model file ready, set up a Python environment for running the ingestion and querying code. You should have Python 3.10+ available. It's recommended to use a virtual environment or Conda environment for the lab.

Install the required Python libraries using pip a convenient `requirements.txt` file has been provided for you. 

```bash
cd workshop/1_llm_cypher
pip install -r requirements.txt
```

After installing spaCy, download the small English model for NER:

```bash
python -m spacy download en_core_web_sm
```

### 1.3 Set Up the Local LLM (using llama.cpp)

Next, we need a local LLM to answer questions.

In this lab, we'll use a 7B parameter model called [neural-chat-7B-v3-3-GGUF](https://huggingface.co/TheBloke/neural-chat-7B-v3-3-GGUF) (a quantized GGUF file). This is the model that will be used in the lab, so for maximum "it just works", stick with this model.

However, this could easy be [bartowski/Meta-Llama-3-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf) or an ollama model. Both are a medium-sized model that can be run on CPU using [llama.cpp](https://github.com/ggerganov/llama.cpp) and its Python bindings.

## Step 2: Data Ingestion: From Raw Text to a Queryable Graph

With the environment ready, we'll proceed to prepare our data (BBC articles) and build the knowledge graph in Neo4j as graph nodes and relationships.

Our knowledge source is a collection of BBC news articles in text format which can be found in the zip file [bbc-lite.zip](./workshop/1_llm_cypher/bbc-lite.zip). This zip file ontains a subset of 300 BBC news articles from the 2225 articles in the [BBC Full Text Document Classification](https://bit.ly/4hBKNjp) dataset. After unzipping the archive, the directory structure will look like:

```
bbc/
├── tech/
    ├── 001.txt
    ├── 002.txt
    ├── 003.txt
    ├── 004.txt
    ├── 005.txt
    └── ...
```

Each file is a news article relating to technology in the world today.

You may need to unzip the `bbc-lite.zip` file which you can do by running this script:

```bash
unzip bbc-lite.zip
```

### 2.1 What We're Really Building

Forget vector stores for a moment. We're creating **two node labels** and **one relationship type**-all you need for entity-centric retrieval:

| Node Label       | Key Properties           | Purpose                                    |
| ---------------- | ------------------------ | ------------------------------------------ |
| `:Document`      | `id`, `title`, `content` | Holds the full article text                |
| `:Entity`       | `name`                   | Unique named entities (people, orgs, etc.) |
| **Relationship** | **Direction**            | **Meaning**                                |
| `[:MENTIONS]`    | `(Document) → (Entity)` | "This article talks about that entity."    |

No vectors... just raw NER-driven connections that keep the graph clean and demo-ready.

### 2.2 Ingestion Script

Now we will construct the knowledge graph in Neo4j by creating nodes for **documents** and **entities**, and defining relationships among them. Our graph schema will be:

* **Document** nodes: each article is a document node with properties like `title` (we'll use filename as title) and `content` (the full text).
* **Entity** nodes: significant entities mentioned in the articles (we'll extract these via NER).

Relationships:

* `(:Document)-[:MENTIONS]->(:Entity)` - links a document to an entity it mentions.

We'll use **spaCy** to identify named entities in each article as our "areas of interest." SpaCy's small English model can recognize entities like PERSON, ORG (organization), GPE (location), etc. We'll treat each unique entity text as a Entity node (with an optional property for its type/label).

This code can be found and executed from this location: [./workshop/1_llm_cypher/ingest.py](https://github.com/davidvonthenen/2025-odsc-east-workshop/tree/main/LAPTOP/workshop/1_llm_cypher/ingest.py).

```python
#!/usr/bin/env python3

import os
import uuid
import spacy
from neo4j import GraphDatabase

# Neo4j connection settings
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4jneo4j"

# Path to the unzipped BBC dataset folder (with subfolders like 'tech')
DATASET_PATH = "./bbc"

def ingest_bbc_documents_with_ner():
    """
    Ingest BBC documents from the 'technology' subset (or other categories if desired)
    and store them in Neo4j with Document and Entity nodes. The code uses spaCy for NER
    and links documents to extracted entities using MENTIONS relationships.
    """
    # Load spaCy's small English model for Named Entity Recognition
    nlp = spacy.load("en_core_web_sm")

    # Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Perform ingestion in a session
    with driver.session() as session:
        # Optional: clear old data
        print("Clearing old data from Neo4j...")
        session.run("MATCH (n) DETACH DELETE n")
        print("Old data removed.\n")

        # Walk through each category folder
        for category in os.listdir(DATASET_PATH):
            category_path = os.path.join(DATASET_PATH, category)
            if not os.path.isdir(category_path):
                continue  # Skip non-directories

            print(f"Ingesting documents in category '{category}'...")
            for filename in os.listdir(category_path):
                if filename.endswith(".txt"):
                    filepath = os.path.join(category_path, filename)

                    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                        text_content = f.read()

                    # Generate a UUID for each document
                    doc_uuid = str(uuid.uuid4())

                    # Create or MERGE the Document node
                    create_doc_query = """
                    MERGE (d:Document {doc_uuid: $doc_uuid})
                    ON CREATE SET
                        d.title = $title,
                        d.content = $content,
                        d.category = $category
                    RETURN d
                    """
                    session.run(
                        create_doc_query,
                        doc_uuid=doc_uuid,
                        title=filename,
                        content=text_content,
                        category=category
                    )

                    # Named Entity Recognition
                    doc_spacy = nlp(text_content)

                    # For each recognized entity, MERGE on (name + label)
                    # Then create a relationship from the Document to the Entity.
                    for ent in doc_spacy.ents:
                        # Skip very short or numeric-only entities
                        if len(ent.text.strip()) < 3:
                            continue

                        # Generate a unique ID for new entities
                        entity_uuid = str(uuid.uuid4())

                        merge_entity_query = """
                        MERGE (e:Entity {name: $name, label: $label})
                        ON CREATE SET e.ent_uuid = $ent_uuid
                        RETURN e.ent_uuid as eUUID
                        """
                        record = session.run(
                            merge_entity_query,
                            name=ent.text.strip(),
                            label=ent.label_,
                            ent_uuid=entity_uuid
                        ).single()

                        ent_id = record["eUUID"]

                        # Now create relationship by matching on doc_uuid & ent_uuid
                        rel_query = """
                        MATCH (d:Document { doc_uuid: $docId })
                        MATCH (e:Entity { ent_uuid: $entId })
                        MERGE (d)-[:MENTIONS]->(e)
                        """
                        session.run(
                            rel_query,
                            docId=doc_uuid,
                            entId=ent_id
                        )

            print(f"Finished ingesting category '{category}'.\n")

    driver.close()
    print("Ingestion with NER complete!")

if __name__ == "__main__":
    ingest_bbc_documents_with_ner()
```

After the script runs, fire up the Neo4j Browser and sanity-check the first few articles by running:

```cypher
MATCH (d:Document)-[:MENTIONS]->(c:Entity)
RETURN d.title AS article, collect(c.name)[..5] AS sampleEntity
LIMIT 3;
```

![TODO: IMAGE]()

At this point, we have a rich knowledge graph: documents categorized, and connected to the key entities they mention. This graph can answer more complex questions than a pure vector search - for example, we can traverse from categories to entities to documents, etc., to find multi-hop relationships. We'll leverage this graph for querying in the next step.

## Step 3: Querying with LangChain's GraphCypherQAChain

Now comes the exciting part: using an LLM (the one we set up in Step 1) to query the Neo4j graph with natural language. LangChain provides a chain specifically for this purpose called **GraphCypherQAChain**. This chain integrates an LLM with a graph by having the **LLM generate Cypher queries** in response to user questions, retrieving data from the graph, and then formulating a final answer. In other words, the LLM acts as a translator from English to Cypher and then uses the query results to compose an answer.

We will configure LangChain's GraphCypherQAChain to use:

* Our **local LLM** as the language model that will do the reasoning and query generation.
* A **Neo4j graph** connection (pointing to our populated database) for executing the Cypher queries.

### 3.1 Hook Up LangChain to Neo4j and the Local LLM

This code will perform the RAG Query using our Graph-based implementation on Neo4j. Based on the data ingested, we will ask the RAG Agent the following questions:

- What are the top 5 most mentioned entities in these articles?

This code can be found and executed from this location: [./workshop/1_llm_cypher/rag_query.py](https://github.com/davidvonthenen/2025-odsc-east-workshop/tree/main/LAPTOP/workshop/1_llm_cypher/rag_query.py).

```python
#!/usr/bin/env python3

import os
import sys
import time
from langchain_community.graphs import Neo4jGraph
from langchain_community.llms import Llama
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate

# ────────────────────────────── Neo4j Settings ───────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "neo4jneo4j")

# ───────────────────────────── llama-cpp Settings ────────────────────────────

# TODO: fix the home dir path
MODEL_PATH = "./models/neural-chat-7b-v3-3.Q4_K_M.gguf"  # Adjust to your local model

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
        "Output **only** the Cypher query-no explanation, no labels, no "
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
    llm = Llama(
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
        verbose=True,
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

if __name__ == "__main__":
    main()
```

A couple of notes:

* `Neo4jGraph` is a LangChain wrapper that uses the Neo4j Python driver under the hood. It can optionally introspect the database to understand the schema. If `enhanced_schema=True`, it will sample some data to list example property values which can help the LLM understand the domain (this can be useful, though not strictly required).
* We printed `graph.schema` to see what it detected. It should list labels like **Document**, **Entity** and their properties (e.g., `Document` has properties `id`, `title`, `content`, etc.) and relationships like **MENTIONS**. This schema info will be provided to the LLM in its prompt context so that it knows how to form the Cypher queries.
* We then load the LLM. We use `Llama` from LangChain, giving the path to our .gguf model file. We also specify `n_ctx=32768` to set a context window (adjust based on model capability). If you have GPU acceleration, you could pass parameters like `n_gpu_layers` or use a different BLAS, but CPU should work for a 7B model (albeit slowly). The `verbose=False` just suppresses internal logging from the LLM.

#### Understanding the LLM-Generated Cypher Queries

It's worth noting that **we did not manually program any Cypher queries** for our Q&A - the LLM generated them on the fly based on the question and the graph schema. This demonstrates a powerful pattern:

* The **knowledge graph** stores facts and relationships explicitly (documents, their topics, categories, etc.).
* The **LLM** acts as a reasoner and translator, mapping a natural question to the right graph query, and then interpreting the results.

LangChain's GraphCypherQAChain provided the scaffolding to make this happen easily. If you check the `verbose` output, you might see something like:

```
> Entering new GraphCypherQAChain chain...
Generated Cypher:
MATCH (d:Document)-[:MENTIONS]->(p:Entity)
RETURN p.name, count(*) ORDER BY count(*) DESC LIMIT 5
Full Context:
[{'p.name': 'government', 'count(*)': 12}, {'p.name': 'Prime Minister', 'count(*)': 8}, ...]
> Finished chain.
```

> **IMPORTANT:** It is also worth noting that this method is **EXTREMELY** brittle. Try changing the questions by rephrasing. How did it fair? Not good I am guessing.

## Section 2: Understanding Fixed Logic

Sticking strictly to `Document → MENTIONS → Entity` proves that even a minimal knowledge graph outshines flat vector search when your questions hinge on entities rather than fuzzy semantics. Now that we have taken a look at using LLM-Generated Cypher Queries, let's take a look at using Fixed Logic for Cypher paths.

→ [Next Up: Fixed Logic for Cypher paths](./STEP2_FIXED_LOGIC.md) 
