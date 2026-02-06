import google.generativeai as genai
from app.core.clients import collection, embed_text, GENERATION_MODEL

class RAGService:
    def retrieve(self, query: str, user_id: str, k: int = 4):
        query_embedding = embed_text([query])[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where={
                "userId": user_id
            }
        )
        print('results: ', results)
        
        if not results["documents"]:
            return []
            
        return results["documents"][0]

    async def query(self, query: str, user_id: str) -> dict:
        retrieved_chunks = self.retrieve(query, user_id)

        print('retrieved chunks:', retrieved_chunks)
        
        current_context = "\n\n".join(retrieved_chunks)

        prompt = f"""
        Use the context below to answer the question.
        If the answer is not in the context, say you don't know nothin.

        CONTEXT:
        {current_context}

        QUESTION:
        {query}
        """

        model = genai.GenerativeModel(GENERATION_MODEL)
        response = model.generate_content(prompt)

        return {
            "answer": response.text,
            "sources": retrieved_chunks
        }

rag_service = RAGService()
