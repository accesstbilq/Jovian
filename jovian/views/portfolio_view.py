# jovian/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
import json
from typing import Dict, Any, List
from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHROMA_DB_PATH = BASE_DIR / "jovian/chroma_db"
COLLECTION_NAME = "project_portfolio"

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

class VectorStoreAPIView(APIView):
    """API to retrieve ALL vector store data with filtering"""
    
    def get(self, request):
        """GET all vectors with optional filtering"""
        try:
            # Load vectorstore
            vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=str(CHROMA_DB_PATH),
            )
            
            # Get ALL documents
            all_docs = vectorstore.get()
            
            # Transform for frontend
            vectors_data = []
            for i, (doc_id, doc_content, doc_metadata) in enumerate(zip(
                all_docs['ids'], 
                all_docs['documents'], 
                all_docs['metadatas']
            )):
                vectors_data.append({
                    "id": doc_id,
                    "index": i,
                    "content": doc_content[:500] + "..." if len(doc_content) > 500 else doc_content,
                    "full_content": doc_content,
                    "metadata": {
                        "category": doc_metadata.get("category", "unknown"),
                        "sub_type": doc_metadata.get("sub_type", ""),
                        "keywords": doc_metadata.get("keywords", ""),
                        "project_ref": doc_metadata.get("project_ref", ""),
                        "evidence_type": doc_metadata.get("evidence_type", ""),
                        "taxonomy_score": doc_metadata.get("taxonomy_score", 1.0),
                        "chunk_id": doc_metadata.get("chunk_id", i),
                        "source": doc_metadata.get("source", ""),
                        "created_at": doc_metadata.get("created_at", ""),
                    }
                })
            
            # Apply filters if provided
            category_filter = request.query_params.get('category')
            search_query = request.query_params.get('search', '').lower()
            
            filtered_data = vectors_data
            
            if category_filter:
                filtered_data = [d for d in filtered_data if d["metadata"]["category"] == category_filter]
            
            if search_query:
                filtered_data = [
                    d for d in filtered_data 
                    if (search_query in d["content"].lower() or 
                        search_query in d["metadata"]["keywords"].lower() or
                        search_query in d["metadata"]["project_ref"].lower())
                ]
            
            # Get unique categories for filter dropdown
            categories = list(set(d["metadata"]["category"] for d in vectors_data))
            
            return Response({
                "success": True,
                "total_vectors": len(vectors_data),
                "filtered_count": len(filtered_data),
                "categories": sorted(categories),
                "data": filtered_data,
                "vectorstore_stats": {
                    "collection": COLLECTION_NAME,
                    "total_docs": len(all_docs['ids']),
                    "dimensions": len(all_docs['embeddings'][0]) if all_docs['embeddings'] else 3072
                }
            })
            
        except Exception as e:
            return Response({
                "success": False,
                "error": str(e),
                "message": "Failed to retrieve vector data"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class VectorSearchAPIView(APIView):
    """API for semantic search"""
    
    def get(self, request):
        """Semantic search endpoint"""
        try:
            query = request.query_params.get('q', '')
            k = int(request.query_params.get('k', 5))
            
            if not query:
                return Response({"error": "Query parameter 'q' required"}, status=400)
            
            vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=str(CHROMA_DB_PATH),
            )
            
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke(query)
            
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": 0.95  # Add actual scoring later
                })
            
            return Response({
                "success": True,
                "query": query,
                "results": results
            })
            
        except Exception as e:
            return Response({"error": str(e)}, status=500)

# Simple JSON endpoint for quick testing
def vector_data_json(request):
    """Quick JSON endpoint for testing"""
    try:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DB_PATH),
        )
        all_docs = vectorstore.get()
        
        data = {
            "total": len(all_docs['ids']),
            "categories": list(set(m.get("category") for m in all_docs['metadatas'])),
            "sample": all_docs['documents'][:3]
        }
        return JsonResponse(data)
    except:
        return JsonResponse({"error": "Vectorstore not found"}, status=500)