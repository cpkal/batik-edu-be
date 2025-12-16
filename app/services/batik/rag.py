class RAGService:  
    def __init__(self, index, gemini_client):
        self.index = index
        self.gemini_client = gemini_client

    def answer_query(self, query, embedding_model="gemini-embedding-001", top_k=5):
        """
        Retrieve context using Pinecone and generate an answer using Gemini LLM.

        Args:
            query (str): The user's question.
            embedding_model (str): Name of the Gemini embedding model.
            top_k (int): Number of top matches to retrieve from Pinecone.

        Returns:
            str: Generated answer.
        """

        query_embedding = self.gemini_client.models.embed_content(
            model=embedding_model,
            contents=[query]
        ).embeddings[0].values

        result = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        retrieved_texts = [match['metadata']['text'] for match in result['matches']]
        context = " ".join(retrieved_texts)

        prompt = f"""
            Answer based on this context:
            {context}
            
            Question: {query}
            
            if the question is out of context, just say I don't know
            """

        response = self.gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text