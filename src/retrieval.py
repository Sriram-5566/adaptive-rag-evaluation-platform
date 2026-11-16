
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticRetriever:

    def __init__(self):
        print("Loading semantic embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, question, chunks, top_k=3):

        if not chunks:
            return []

        # Convert question and chunks into embeddings
        question_embedding = self.model.encode([question])
        chunk_embeddings = self.model.encode(chunks)

        # Calculate similarity scores
        scores = cosine_similarity(
            question_embedding,
            chunk_embeddings
        )[0]

        # Select the highest-scoring chunks
        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:
            results.append({
                "chunk_id": int(index) + 1,
                "score": float(scores[index]),
                "text": chunks[index]
            })

        return results
