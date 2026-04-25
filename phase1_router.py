import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


BOT_PERSONAS = {
    "bot_a": {
        "name": "Tech Maximalist",
        "persona": (
            "I believe AI and crypto will solve all human problems. I am highly optimistic "
            "about technology, Elon Musk, and space exploration. I dismiss regulatory concerns."
        ),
    },
    "bot_b": {
        "name": "Doomer / Skeptic",
        "persona": (
            "I believe late-stage capitalism and tech monopolies are destroying society. "
            "I am highly critical of AI, social media, and billionaires. I value privacy and nature."
        ),
    },
    "bot_c": {
        "name": "Finance Bro",
        "persona": (
            "I strictly care about markets, interest rates, trading algorithms, and making money. "
            "I speak in finance jargon and view everything through the lens of ROI."
        ),
    },
}


class PersonaRouter:
    """
    Loads bot personas into an in-memory ChromaDB collection, then routes
    incoming posts to whichever bots would actually care about that topic.
    """

    def __init__(self):
        # all-MiniLM-L6-v2 is small, fast, and good enough for this kind of semantic matching
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.create_collection(
            name="bot_personas",
            metadata={"hnsw:space": "cosine"},
        )
        self._index_personas()

    def _index_personas(self):
        ids, docs, embeddings = [], [], []
        for bot_id, data in BOT_PERSONAS.items():
            ids.append(bot_id)
            docs.append(data["persona"])
            embeddings.append(self.encoder.encode(data["persona"]).tolist())
        self.collection.add(documents=docs, embeddings=embeddings, ids=ids)
        print(f"[Router] Indexed {len(ids)} personas into vector store.\n")

    def route_post_to_bots(self, post_content: str, threshold: float = 0.30) -> list:
        """
        Returns a list of bots whose persona is semantically close enough to the post.

        Threshold note: 0.85 is the right ballpark for OpenAI's text-embedding-ada-002.
        sentence-transformers (all-MiniLM-L6-v2) produces lower absolute values — realistic
        similarity for related content lands around 0.25–0.55, so the default here is 0.30.
        Crank it up if you swap in a stronger embedding model.
        """
        post_vec = self.encoder.encode(post_content).tolist()

        results = self.collection.query(
            query_embeddings=[post_vec],
            n_results=len(BOT_PERSONAS),
            include=["documents", "distances"],
        )

        matched = []
        for bot_id, distance in zip(results["ids"][0], results["distances"][0]):
            # ChromaDB stores cosine distance (0 = identical, 2 = opposite), so flip it
            similarity = 1.0 - distance
            print(f"  {bot_id} ({BOT_PERSONAS[bot_id]['name']}): similarity = {similarity:.4f}")
            if similarity >= threshold:
                matched.append({
                    "bot_id": bot_id,
                    "name": BOT_PERSONAS[bot_id]["name"],
                    "similarity": round(similarity, 4),
                })

        return matched
