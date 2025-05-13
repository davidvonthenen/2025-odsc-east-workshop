from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4jneo4j"

def connect_neo4j():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def find_articles_for_entity(entity_name):
    """
    Returns a list of documents that mention the given entity by name.
    """
    query = """
    MATCH (d:Document)-[:MENTIONS]->(e:Entity)
    WHERE toLower(e.name) = toLower($entityName)
    RETURN d.title AS title, d.category AS category, d.content AS content
    """
    driver = connect_neo4j()
    results_list = []

    with driver.session() as session:
        results = session.run(query, entityName=entity_name)
        for record in results:
            results_list.append({
                "title": record["title"],
                "category": record["category"],
                "content": record["content"]
            })
    driver.close()
    return results_list

if __name__ == "__main__":
    entity = "Ernie Wise"
    docs_mentioning_ernie = find_articles_for_entity(entity)
    
    if docs_mentioning_ernie:
        print(f"Documents mentioning '{entity}':")
        for doc in docs_mentioning_ernie:
            print(f"  Title: {doc['title']}")
            print(f"  Category: {doc['category']}")
            snippet = doc['content'][:300].replace('\n', ' ')
            print(f"  Snippet: {snippet}...")
            print("  ---")
    else:
        print(f"No documents found mentioning '{entity}'.")
