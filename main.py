import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. INITIALIZATION & SETUP
# ==========================================
load_dotenv()  # reads .env file in project root and loads vars into os.environ

# Initialize Pinecone Client
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "realtime-trend-engine"

# Create the index if it doesn't exist (384 dims for all-MiniLM-L6-v2)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

    # Wait until the index is actually ready to accept requests
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# Load lightweight embedding model (runs quickly on a standard laptop CPU)
model = SentenceTransformer('all-MiniLM-L6-v2')

# ==========================================
# 2. DATA SOURCE & EMBEDDING PIPELINE
# ==========================================
# Simulated daily trending feed from a marketplace API
mock_trending_feed = [
    {"id": "prod_1", "title": "Wireless Noise Cancelling Headphones", "category": "Electronics", "trend_score": 9.5},
    {"id": "prod_2", "title": "Ergonomic Mechanical Gaming Keyboard", "category": "Electronics", "trend_score": 8.2},
    {"id": "prod_3", "title": "Stainless Steel Vacuum Insulated Water Bottle", "category": "Fitness",
     "trend_score": 9.1},
    {"id": "prod_4", "title": "High-Density Yoga Mat with Carrying Strap", "category": "Fitness", "trend_score": 7.4},
    {"id": "prod_5", "title": "4K Ultra HD Streaming Media Player", "category": "Electronics", "trend_score": 8.8}
]


def pipeline_feed_to_vector_db(feed_data):
    """Converts product data to vectors and uploads to Pinecone."""
    vectors_to_upsert = []

    for item in feed_data:
        # Generate mathematical vector representation of the product text
        embedding = model.encode(item["title"]).tolist()

        # Structure metadata so we can filter/rank by trends later
        metadata = {
            "title": item["title"],
            "category": item["category"],
            "trend_score": item["trend_score"]
        }

        vectors_to_upsert.append((item["id"], embedding, metadata))

    # Batch upload to Pinecone
    index.upsert(vectors=vectors_to_upsert)
    print(f"Successfully vectorized and stored {len(vectors_to_upsert)} products.")


# Run the ingestion pipeline
pipeline_feed_to_vector_db(mock_trending_feed)


# ==========================================
# 3. HYBRID RECOMMENDATION SEARCH ENGINE
# ==========================================
def get_hybrid_recommendations(user_search_query, alpha=0.5, top_k=3):
    """
    Finds products matching user intent while boosting trending items.
    alpha: Controls balance (1.0 = Pure Semantic Search, 0.0 = Pure Trend Score)
    """
    # Vectorize user search string
    query_vector = model.encode(user_search_query).tolist()

    # Query vector database for semantic matches
    results = index.query(
        vector=query_vector,
        top_k=top_k * 2,  # Fetch extra candidates to recalculate trend weight
        include_metadata=True
    )

    hybrid_ranked_results = []
    for match in results["matches"]:
        semantic_score = match["score"]  # Range: ~0.0 to 1.0 (Cosine)
        trend_score = match["metadata"]["trend_score"] / 10.0  # Normalize 0-10 score to 0-1

        # Calculate Hybrid Score
        final_score = (alpha * semantic_score) + ((1 - alpha) * trend_score)

        hybrid_ranked_results.append({
            "title": match["metadata"]["title"],
            "category": match["metadata"]["category"],
            "semantic_score": semantic_score,
            "trend_score": trend_score,
            "match_confidence": final_score
        })

    # Sort candidates by the final hybrid math calculation
    hybrid_ranked_results.sort(key=lambda x: x["match_confidence"], reverse=True)
    return hybrid_ranked_results[:top_k]


test = "gym"
alpha = .7
# Test the recommendation engine
print(f"\n--- Testing Recommendations for user searching '{test}' ---\nAlpha: {alpha}")
recommendations = get_hybrid_recommendations(test, alpha)
for idx, rec in enumerate(recommendations, 1):
    print(f"{idx}. {rec['title']} [{rec['category']}] - "
          f"Hybrid: {rec['match_confidence']:.2f} "
          f"(semantic: {rec['semantic_score']:.2f}, trend: {rec['trend_score']:.2f})")