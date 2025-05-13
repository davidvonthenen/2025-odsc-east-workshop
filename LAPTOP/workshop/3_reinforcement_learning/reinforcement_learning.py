#!/usr/bin/env python3

import time
import uuid
from pathlib import Path

import spacy
from neo4j import GraphDatabase
from llama_cpp import Llama

# Neo4j connection settings
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4jneo4j"

# Path to your local LLaMA model file. Example: "models/ggml-model-q4_0.bin"
MODEL_PATH = str(Path.home() / "models" / "neural-chat-7b-v3-3.Q4_K_M.gguf")

# Five BBC-style technology facts
TECH_FACTS = [
    """
    OpenAI has agreed to buy artificial intelligence-assisted coding tool Windsurf for about $3 billion, Bloomberg News reported on Monday, citing people familiar with the matter.
    The deal has not yet closed, the report added.

    OpenAI declined to comment, while Windsurf did not immediately respond to Reuters' requests for comment.

    Windsurf, formerly known as Codeium, had recently been in talks with investors, including General Catalyst and Kleiner Perkins, to raise funding at a $3 billion valuation, according to Bloomberg News.
    
    It was valued at $1.25 billion last August following a $150 million funding round led by venture capital firm General Catalyst. Other investors in the company include Kleiner Perkins and Greenoaks.
    
    The deal, which would be OpenAI's largest acquisition to date, would complement ChatGPT's coding capabilities. The company has been rolling out improvements in coding with the release of each of its newer models, but the competition is heating up.
    
    OpenAI has made several purchases in recent years to boost different segments of its AI products. It bought search and database analytics startup Rockset in a nine-figure stock deal last year, to provide better infrastructure for its enterprise products.
    
    OpenAI's weekly active users surged past 400 million in February, jumping sharply from the 300 million weekly active users in December.
    """,
    """
    Will the Apple Vision Pro be discontinued? It's certainly starting to look that way. In the last couple of months, numerous reports have emerged suggesting that Apple is either slowing down or completely halting production of its flagship headset.

    So, what does that mean for Apple's future in the extended reality market?

    Apple has had a rough time with its Vision Pro headset. Despite incredibly hype leading up to the initial release, and the fact that preorders for the device sold out almost instantly, demand for headset has consistently dropped over the last year.

    In fact, sales have diminished to the point that rumors have been coming thick and fast. For a while now, industry analysts and tech enthusiasts believe Apple might give up on its XR journey entirely and return its focus to other types of tech (like smartphones).

    However, while Apple has failed to achieve its sales targets with the Vision Pro, I don't think they will abandon the XR market entirely. It seems more likely that Apple will view the initial Vision Pro as an experiment, using it to pave the way to new, more popular devices.

    Here's what we know about Apple's XR journey right now.
    """,
    """
    OpenAI sees itself paying a lower share of revenue to its investor and close partner Microsoft by 2030 than it currently does, The Information reported, citing financial documents.

    The news comes after OpenAI this week changed tack on a major restructuring plan to pursue a new plan that would see its for-profit arm becoming a public benefit corporation (PBC) but continue to be controlled by its nonprofit division.

    OpenAI currently has an agreement to share 20% of its top line with Microsoft, but the AI company has told investors it expects to share 10% of revenue with its business partners, including Microsoft, by the end of this decade, The Information reported.

    Microsoft has invested tens of billions in OpenAI, and the two companies currently have a contract until 2030 that includes revenue sharing from both sides. The deal also gives Microsoft rights to OpenAI IP within its AI products, as well as exclusivity on OpenAI's APIs on Azure.

    Microsoft has not yet approved OpenAI's proposed corporate structure, Bloomberg reported on Monday, as the bigger tech company reportedly wants to ensure the new structure protects its multi-billion-dollar investment.

    OpenAI and Microsoft did not immediately return requests for comment.
    """,
    """
    Perplexity, the developer of an AI-powered search engine, is raising a $50 million seed and pre-seed investment fund, CNBC reported. Although the majority of the capital is coming from limited partners, Perplexity is using some of the capital it raised for the company's growth to anchor the fund. Perplexity reportedly raised $500 million at a $9 billion valuation in December.

    Perplexity's fund is managed by general partners Kelly Graziadei and Joanna Lee Shevelenko, who in 2018 co-founded an early-stage venture firm, F7 Ventures, according to PitchBook data. F7 has invested in startups like women's health company Midi. It's not clear if Graziadei and Shevelenko will continue to run F7 or if they will focus all their energies on Perplexity's venture fund.

    OpenAI also manages an investment fund known as the OpenAI Startup Fund. However, unlike Perplexity, OpenAI claims it does not use its own capital for these investments.
    """,
    """
    DeepSeek-R2 is the upcoming AI model from Chinese startup DeepSeek, promising major advancements in multilingual reasoning, code generation, and multimodal capabilities. Scheduled for early 2025, DeepSeek-R2 combines innovative training techniques with efficient resource usage, positioning itself as a serious global competitor to Silicon Valley's top AI technologies.

    In the rapidly evolving landscape of artificial intelligence, a new contender is emerging from China that promises to reshape global AI dynamics. DeepSeek, a relatively young AI startup, is making waves with its forthcoming DeepSeek-R2 model—a bold step in China's ambition to lead the global AI race.

    As Western tech giants like OpenAI, Anthropic, and Google dominate headlines, DeepSeek's R2 model represents a significant milestone in AI development from the East. With its unique approach to training, multilingual capabilities, and resource efficiency, DeepSeek-R2 isn't just another language model—it's potentially a game-changer for how we think about AI development globally.

    What is DeepSeek-R2?
    DeepSeek-R2 is a next-generation large language model that builds upon the foundation laid by DeepSeek-R1. According to reports from Reuters, DeepSeek may be accelerating its launch timeline, potentially bringing this advanced AI system to market earlier than the original May 2025 target.

    What sets DeepSeek-R2 apart is not just its improved performance metrics but its underlying architecture and training methodology. While R1 established DeepSeek as a serious competitor with strong multilingual and coding capabilities, R2 aims to push these boundaries significantly further while introducing new capabilities that could challenge the dominance of models like GPT-4 and Claude.

    DeepSeek-R2 represents China's growing confidence and technical capability in developing frontier AI technologies. The model has been designed from the ground up to be more efficient with computational resources—a critical advantage in the resource-intensive field of large language model development.
    """
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
        model_path=MODEL_PATH,
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
