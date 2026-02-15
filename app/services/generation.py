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
        
        if not results["documents"]:
            return [], []
            
        return results["documents"][0], results["metadatas"][0]

    async def query(self, query: str, user_id: str) -> dict:
        retrieved_chunks, metadatas = self.retrieve(query, user_id)

        current_context = "\n\n".join(retrieved_chunks)

        prompt = f"""
        Use the context below to answer the question.
        Don't hallisunate the answer on your own.
        Based on the context develop a detail note 
        and understand it clearly to develop a summarization.
        priority is not hallisunating. 
        If there is no exact data in the context warn the user like "I don't have exact data in the context to answer this question"

        CONTEXT:
        {current_context}

        QUESTION:
        {query}
        """

        print('current_context: ', current_context)
        print('metadatas: ', metadatas)

        model = genai.GenerativeModel(GENERATION_MODEL)
        response = model.generate_content(prompt)

        file_ids = [meta.get("fileId") for meta in metadatas if meta.get("fileId")]

        print('file_ids: ', file_ids)
        return {
            "answer": response.text,
            "sources": retrieved_chunks,
            "file_ids": file_ids
        }

rag_service = RAGService()
