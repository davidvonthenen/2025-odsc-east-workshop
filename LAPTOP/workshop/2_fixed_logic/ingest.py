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
