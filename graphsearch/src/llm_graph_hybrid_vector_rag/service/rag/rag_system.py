from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.chat_models import BaseChatModel

class Neo4jVectorSearch:
    def __init__(self, vector_store: VectorStore, search_type="hybrid"):
        self.vector_store = vector_store
        self.search_type = search_type

    def similarity_search_with_score(self, query: str, k: int = 3):
        """Perform similarity search with scores."""
        if not self.vector_store:
            raise ValueError("Vector index not created. Call create_vector_index() first.")
        
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def as_retriever(self, search_kwargs: dict = None):
        """Get retriever for RAG chain."""
        if not self.vector_store:
            raise ValueError("Vector index not created. Call create_vector_index() first.")
        
        if search_kwargs is None:
            search_kwargs = {"k": 8}
        
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

class CompanyRAGChain:
    def __init__(self, llm: BaseChatModel, retriever: BaseRetriever):
        self.llm = llm
        self.retriever = retriever
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant. Answer ONLY using the provided context. "
             "If the answer is not in the context, say you don't know."),
            ("human",
             "Context:\n{context}\n\nQuestion:\n{question}")
        ])
        
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def invoke(self, question: str) -> str:
        """Ask a question and get an answer."""
        return self.rag_chain.invoke(question)
    
    def search_and_display_results(self, question: str, k: int = 3):
        """Search and display results with scores."""
        if hasattr(self.retriever, 'vectorstore'):
            results = self.retriever.vectorstore.similarity_search_with_score(question, k=k)
            for doc, score in results:
                print(f"Score: {score}")
                print(f"Metadata: {doc.metadata}")
                print(f"Content: {doc.page_content[:200]}...")
                print("----")
        else:
            print("Direct search not available with this retriever type.")
