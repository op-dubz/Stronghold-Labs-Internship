# @title Setup and Imports

import google.generativeai as genai
import numpy as np
import os
from pinecone import Pinecone, ServerlessSpec

# Configure your API key 
APIKEY = "AIzaSyA6bnQK33HYRspkrOi8-Q54bq4E4RgcHj4" 
# It's recommended to store your API key securely, e.g., in environment variables
# For this example, we'll assume it's directly set.
# Replace "YOUR_API_KEY" with your actual Google API Key
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"
genai.configure(api_key=APIKEY) 

# Initialize the Generative Model
model = genai.GenerativeModel('gemini-1.5-flash')

# Chat history and truncation settings
chat_history = []
MAX_CHAT_HISTORY_LENGTH = 4 # Number of recent turns to keep in active memory, so # of messages by user + chatbot = 2 * chat_history_length (I think) 


#----------------------------------------------------------------------------------------------------

# Configure the Pinecone API key 
pineconeAPIKEY = "pcsk_4xPCuD_6WLyNse1TcP3qmHKhMPKNdRCSQwU8g9MvVVVorLvLjWVboM3CwA76YnNzu8yd4V" 

#initialize pinecone client
pc = Pinecone(api_key=pineconeAPIKEY)  
PINECONE_INDEX_NAME = "chatbot-memory-integrated" 
if not pc.has_index(PINECONE_INDEX_NAME):
    pc.create_index_for_model(
        name = PINECONE_INDEX_NAME,      
        cloud="aws",
        region="us-east-1",
        embed = { 
            "model": "llama-text-embed-v2", #Does this allow for auto embedding without needing an embedding function?? I think so.  
            "field_map": {"text": "message_text"}
        }
    )
    print(f"Created new Pinecone index '{PINECONE_INDEX_NAME}' with integrated embedding model")
else:
    print(f" :( thats not good. maybe use the basic vdb instead")  

index = pc.Index(PINECONE_INDEX_NAME)
print(f"Successfully connected to Pinecone index with integrated embeddings")

#----------------------------------------------------------------------------------------------------

# Pinecone Vector Database Class (replacing SimulatedVectorDB) 
class PineconeVectorDB:
    def __init__(self, index): 
        self.index = index  # Real Pinecone index connection
        self._is_built = True  # Pinecone indexes are always built
        self.item_counter = 0  # Track item IDs

    def add_item(self, item_id: int, embedding: list, text_content: str):
        """Adds an item with text content - Pinecone handles embedding generation."""
        # with integrated models, we send text directly: no need for embeddings
        # Enhanced metadata for better retrieval 
        metadata = {
            "item_id": str(item_id),
            "timestamp": str(np.datetime64('now')), 
            "message_type": "chat_message",
            "text_content": text_content,
            "word_count": len(text_content.split()),  # For chunking optimization 
            "keywords": self._extract_keywords(text_content)  # For hybrid search  
        }  
        
        # Upsert with text: Pinecone generates embeddings automatically
        self.index.upsert(
            vectors=[(str(item_id), {"message_text": text_content}, metadata)]
        )   
        
        print(f"Added item ID {item_id} to Pinecone VDB (Text: '{text_content[:30]}...')")

    def build(self, n_trees: int = 10): #I think a default value for n_trees is ok 
        #Pinecone indexes are automatically built: no manual build needed.
        print(f"Pinecone index is already built and optimized for similarity search.")
        self._is_built = True # We probably don't need the function I think.

    def query(self, query_text: str, k: int = 1) -> list:  
        #Real advanced semantic search using Pinecone's integrated model.
        #Returns the k most semantically similar items hybrid search techniques. 
        
        if not self._is_built:
            print("Pinecone index is always ready for querying. We continue as needed.")

        try:  
            # Query with text directly: Pinecone handles embedding generation; Query expansion 
            expanded_queries = self._expand_query(query_text)   
            
            # Hybrid search combining semantic and keyword matching
            all_results = []
            
            for expanded_query in expanded_queries:
                # Semantic search with expanded query
                query_results = self.index.query(
                    vector={"message_text": expanded_query},
                    top_k=k * 2,  # Get more results for re-ranking
                    include_metadata=True
                )
                
                # Re-rank results based on multiple factors   
                reranked_results = self._rerank_results(query_results.matches, query_text)
                all_results.extend(reranked_results)
            
            # Deduplicate and select top k results
            final_results = self._deduplicate_and_select_top(all_results, k)
            
            # Format results
            retrieved_results = []
            for result in final_results:
                item_id = result['id']
                text_content = result['text_content']
                similarity_score = result['score']
                retrieval_method = result.get('method', 'semantic')
                retrieved_results.append(f"Retrieved content (ID: {item_id}, Score: {similarity_score:.3f}, Method: {retrieval_method}): '{text_content}'")
            
            return retrieved_results
            
        except Exception as e:
            print(f"Error querying Pinecone: {e}. Please try again.")
            return [] 

    # Query Expansion 
    def _expand_query(self, query_text: str) -> list: 
        #Expand query with synonyms and related terms for better retrieval.
        expanded_queries = [query_text]  # Original query 
        
        # Simple synonym expansion (when making this more advanced, use a proper thesaurus API which idk)   
        synonyms = {   
            "what": ["tell me about", "explain", "describe"],
            "how": ["explain how", "describe the process", "what is the method"],
            "why": ["explain why", "what is the reason", "what causes"],
            "when": ["at what time", "during what period", "what date"],
            "where": ["in what location", "at what place", "which place"],
            "who": ["which person", "what person", "tell me about"]
        }
        
        # Expand query with synonyms
        words = query_text.lower().split()
        for word in words:
            if word in synonyms:
                for synonym in synonyms[word]:
                    expanded_query = query_text.lower().replace(word, synonym)
                    if expanded_query not in expanded_queries:
                        expanded_queries.append(expanded_query)
        
        # Add question variations 
        if query_text.endswith('?'):
            # Remove question mark and add as statement
            statement_query = query_text[:-1].strip()
            if statement_query not in expanded_queries:
                expanded_queries.append(statement_query)
        
        return expanded_queries[:3]  # Limit to 3 expanded queries

    # Re-ranking Method. NEED HELP CUZ IDK IF ITS GOOD 
    def _rerank_results(self, matches, original_query: str) -> list:
        #Re-rank results based on multiple relevance factors. 
        
        reranked = []
        
        for match in matches:
            score = match.score
            text_content = match.metadata.get('text_content', '')
            
            # Boost score based on keyword overlap
            keyword_boost = self._calculate_keyword_overlap(original_query, text_content)
            
            # Boost score based on recency (newer messages slightly preferred)
            recency_boost = self._calculate_recency_boost(match.metadata.get('timestamp', ''))
            
            # Boost score based on content length (prefer meaningful responses)
            length_boost = self._calculate_length_boost(text_content)
            
            # Combined score with weights
            final_score = (score * 0.6 + keyword_boost * 0.2 + recency_boost * 0.1 + length_boost * 0.1)
            
            reranked.append({
                'id': match.id,
                'text_content': text_content,
                'score': final_score,
                'original_score': score,
                'method': 'hybrid_reranked'
            })
        
        # Sort in decreasing order by final score    
        reranked.sort(key = lambda x: x['score'], reverse = True)  
        return reranked

    # Helper methods for re-ranking 
    def _extract_keywords(self, text: str) -> list: 
        # Extract important keywords from text. 
        # Simple keyword extraction (when making this more advanced, use NLP libraries but idk)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        words = text.lower().split()   
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:10]  # Limit to top 10 keywords   

    def _calculate_keyword_overlap(self, query: str, content: str) -> float:
        # Calculate keyword overlap between the query and the content. 
        query_keywords = set(self._extract_keywords(query))
        content_keywords = set(self._extract_keywords(content))
        
        if not query_keywords: # Size = 0 
            return 0.0
        
        overlap = len(query_keywords.intersection(content_keywords))
        return overlap / len(query_keywords) 

    def _calculate_recency_boost(self, timestamp: str) -> float:
        # Calculate recency boost for newer messages.
        try:
            # Simple recency boost (newer messages get slight preference)
            # I don't know if we need time calculations to be accounted for in this function 
            return 0.10  # Small boost for recency
        except:       
            return 0.0

    def _calculate_length_boost(self, text: str) -> float:
        # Calculate boost based on content length. 
        word_count = len(text.split())  
        # Prefer medium-length responses (not too short, not too long)  
        if 5 <= word_count <= 50: # Adjust accordingly (I think)  
            return 0.2
        elif word_count > 50:
            return 0.1
        else:
            return 0.0

    def _deduplicate_and_select_top(self, results: list, k: int) -> list:
        # Remove duplicates and select top k results. 
        seen_ids = set()
        unique_results = []
        
        for result in results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                unique_results.append(result)
        
        return unique_results[:k]

    #Deleting the current model 
    def reset(self):
        #Deletes all vectors from pinecone index.
        try:
            # Delete all vectors from the index
            self.index.delete(delete_all=True)
            self.item_counter = 0
            print("Pinecone index has been reset (all vectors deleted).")
        except Exception as e:
            print(f"Error resetting Pinecone index: {e}. Please try again.")


def start_chat(): 
    # Global instance of a real Pinecone VDB   
    vdb1 = PineconeVectorDB(index) 
    print("Using real Pinecone Vector Database with integrated embeddings")
    vdb1.build() 
    # Global counter for VDB item IDs 
    vdb_index_counter = 0 

    # No need for get_embedding function - Pinecone handles it automatically
    def get_embedding(text: str) -> list:
        """With integrated models, Pinecone handles embedding generation automatically."""
        # Return None since we don't need to generate embeddings manually
        return None  
    
    return vdb1, vdb_index_counter, get_embedding

# Initialize the Pinecone VDB
vdb, vdb_index_counter, get_embedding = start_chat() 