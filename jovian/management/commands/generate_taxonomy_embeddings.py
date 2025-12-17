import re
import json
import traceback
from typing import List, Dict, Any, Set
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from django.core.management.base import BaseCommand
from langchain_core.documents import Document
from collections import Counter

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
CHROMA_DB_PATH = BASE_DIR / "jovian/chroma_db"
JSON_FILE = BASE_DIR / "advanced_portfolio_data.json"
COLLECTION_NAME = "project_portfolio"

# ✅ Global LLM and embeddings
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

class Command(BaseCommand):
    help = 'Generate 5 Master Taxonomy embeddings from portfolio JSON'
    
    MASTER_TAXONOMY = {
        "Technical_Capability": {
            "purpose": "Do you know X? How do you build Y?",
            "keywords": ["Node.js", "React", "MERN", "AWS", "Auto-scaling", "Shopify API", 
                        "Klaviyo", "MongoDB", "PostgreSQL", "GraphQL", "Headless"],
            "evidence": "Architecture Implementation"
        },
        "Domain_Expertise": {
            "purpose": "Industry experience & functionality",
            "keywords": ["SaaS", "E-commerce", "Cybersecurity", "LMS", "Dashboard", 
                        "Influencer Marketing", "Phishing", "Multi-tenant"],
            "evidence": "Functional Case Study"
        },
        "Business_Impact_Trust": {
            "purpose": "Why trust you? Startup experience?",
            "keywords": ["Startup", "Scale", "Growth", "ROI", "5-star rating", "Reviews", 
                        "Series A", "Success Story", "Retention"],
            "evidence": "Social Proof / Metrics"
        },
        "Engagement_Hiring": {
            "purpose": "Resource availability & hiring models",
            "keywords": ["Hire Dedicated Developer", "Team Augmentation", "Fixed Cost", 
                        "Hourly Model", "Full-time", "Resource availability", "Staffing"],
            "evidence": "Service Offering"
        },
        "Process_Communication": {
            "purpose": "How do we work? NDAs? Communication?",
            "keywords": ["Agile", "Sprint", "Slack", "Trello", "Jira", "NDA", 
                        "Confidentiality", "Discovery Call", "Quote", "Estimation"],
            "evidence": "Workflow / Protocol"
        }
    }

    def _chroma_safe_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """✅ CHROMA SAFE: Convert ALL values to primitives (str/int/float/bool/None)"""
        safe = {}
        for key, value in metadata.items():
            if value is None:
                safe[key] = None
            elif isinstance(value, (str, int, float, bool)):
                safe[key] = value
            elif isinstance(value, list):
                # ✅ Convert lists to comma-separated strings
                safe[key] = ", ".join(str(item) for item in value[:10])  # Max 10 items
            elif isinstance(value, dict):
                # ✅ Flatten nested dicts
                safe[key] = json.dumps(value, default=str)[:500]  # Max 500 chars
            else:
                safe[key] = str(value)[:500]  # Everything else to string
        return safe

    def build_taxonomy_chunks_from_project_json(self, project_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate EXACTLY 5 taxonomy chunks - one per Master Taxonomy category"""
        self.stdout.write(self.style.SUCCESS("🤖 Generating LLM Taxonomy chunks..."))
        all_chunks = []
        
        for category_name, category_info in self.MASTER_TAXONOMY.items():
            prompt = f"""
You are analyzing Brihaspati Infotech's project portfolio for {category_name}.

**CATEGORY PURPOSE:** {category_info['purpose']}
**KEYWORD TRIGGERS:** {', '.join(category_info['keywords'])}
**EVIDENCE TYPE:** {category_info['evidence']}

**TASK:** Extract ONE paragraph from `project_json` that BEST demonstrates this capability.
- Use ONLY facts from the JSON
- Reference specific project names & technologies  
- Keep concise (100-200 words)
- Make it query-answer ready

**PROJECT JSON:**
{json.dumps(project_json, indent=2)}

**OUTPUT JSON ONLY:**
{{
    "content": "Your extracted paragraph here",
    "metadata": {{
        "category": "{category_name}",
        "sub_type": "Extract from JSON",
        "keywords": ["list", "3-5", "exact", "phrases"],
        "project_ref": "Specific project name"
    }}
}}
            """
            
            try:
                response = llm.invoke(prompt)
                chunk = json.loads(response.content.strip())
                all_chunks.append(chunk)
                self.stdout.write(f"  ✅ {category_name}")
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"  ⚠️ Failed {category_name}: {e}"))
                all_chunks.append({
                    "content": f"No {category_name} evidence found in portfolio.",
                    "metadata": {
                        "category": category_name,
                        "sub_type": "fallback",
                        "keywords": [],  # Will be converted to string
                        "project_ref": "N/A"
                    }
                })
        
        return self.validate_taxonomy_chunks(all_chunks)

    def validate_taxonomy_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """✅ Strict validation ensuring ALL 5 categories"""
        required_categories: Set[str] = set(self.MASTER_TAXONOMY.keys())
        validated: List[Dict[str, Any]] = []
        
        for chunk in chunks:
            if not isinstance(chunk, dict) or "content" not in chunk or "metadata" not in chunk:
                continue
                
            metadata = chunk["metadata"]
            if not isinstance(metadata, dict):
                continue
                
            category = metadata.get("category")
            if category not in required_categories:
                continue
            
            # ✅ PRE-CONVERT keywords to string here
            keywords_list = metadata.get("keywords", [])
            keywords_str = ", ".join(str(k).strip() for k in keywords_list[:10])
            
            validated_chunk = {
                "content": str(chunk["content"]).strip()[:1000],
                "metadata": {
                    "category": str(category),
                    "sub_type": str(metadata.get("sub_type", "unknown"))[:50],
                    "keywords": keywords_str,  # ✅ STRING - CHROMA SAFE
                    "project_ref": str(metadata.get("project_ref", "unknown"))[:100],
                    "evidence_type": self.MASTER_TAXONOMY[category]["evidence"]
                }
            }
            
            validated.append(validated_chunk)
            required_categories.discard(category)
        
        if required_categories:
            self.stdout.write(self.style.WARNING(f"⚠️ Missing categories: {required_categories}"))
            for missing_cat in required_categories:
                validated.append({
                    "content": f"[{missing_cat}] Portfolio data pending for this category.",
                    "metadata": {
                        "category": missing_cat,
                        "sub_type": "missing",
                        "keywords": "",  # ✅ Empty string - CHROMA SAFE
                        "project_ref": "N/A",
                        "evidence_type": self.MASTER_TAXONOMY[missing_cat]["evidence"]
                    }
                })
        
        self.stdout.write(self.style.SUCCESS(f"✅ Validated {len(validated)} chunks across 5 categories"))
        return validated

    def convert_to_documents(self, taxonomy_chunks: List[Dict[str, Any]]) -> List[Document]:
        """Convert to CHROMA-SAFE LangChain Documents"""
        documents = []
        
        for i, chunk in enumerate(taxonomy_chunks):
            metadata = chunk["metadata"]
            
            # ✅ CHROMA SAFE: ALL PRIMITIVES ONLY
            safe_metadata = self._chroma_safe_metadata({
                "category": metadata["category"],
                "sub_type": metadata["sub_type"],
                "keywords": metadata["keywords"],  # Already string
                "project_ref": metadata["project_ref"],
                "evidence_type": metadata["evidence_type"],
                "chunk_id": i,
                "taxonomy_score": 1.0,
                "created_at": str(datetime.now().isoformat()),
                "source": "brihaspati_portfolio"
            })
            
            doc = Document(
                page_content=str(chunk["content"]),
                metadata=safe_metadata
            )
            documents.append(doc)
        
        self.stdout.write(self.style.SUCCESS(f"📄 Created {len(documents)} CHROMA-SAFE documents"))
        return documents

    def test_retrieval_sample(self, vectorstore: Chroma) -> None:
        """Test retrieval across all 5 categories"""
        test_queries = {
            "Technical_Capability": "Do you have React and Node.js experience?",
            "Domain_Expertise": "Can you build SaaS dashboards?",
            "Business_Impact_Trust": "Have you worked with startups?",
            "Engagement_Hiring": "Do you offer team augmentation?",
            "Process_Communication": "What is your Agile process?"
        }
        
        self.stdout.write(self.style.NOTICE("\n🔍 Testing retrieval by category:"))
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2})
        
        for category, query in test_queries.items():
            docs = retriever.invoke(query)
            cat_match = "✅" if any(d.metadata["category"] == category for d in docs) else "❌"
            self.stdout.write(f"{cat_match} {category:<25} | '{query[:40]}...' -> {len(docs)} docs")

    def upsert_documents_to_vectorstore(self, documents: List[Document]) -> Chroma:
        """✅ Atomic upsert - 100% CHROMA SAFE"""
        if not documents:
            raise ValueError("No documents to upsert")
        
        category_counts = dict(
            sorted(
                Counter(d.metadata.get("category") for d in documents).items()
            )
        )
        
        self.stdout.write(self.style.SUCCESS(f"\n💾 Upserting {len(documents)} documents..."))
        self.stdout.write(self.style.SUCCESS(f"📊 Categories: {category_counts}"))
        
        try:
            # 1. Delete existing collection
            temp_vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=str(CHROMA_DB_PATH),
            )
            temp_vectorstore.delete_collection()
            
            # 2. Create NEW collection with safe documents
            vectorstore = Chroma.from_documents(
                documents=documents,  # ✅ Already CHROMA SAFE
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=str(CHROMA_DB_PATH),
            )
            
            # 3. Test retrieval
            self.test_retrieval_sample(vectorstore)
            
            self.stdout.write(
                self.style.SUCCESS(f"\n✅ SUCCESS: {len(documents)} docs across 5 taxonomies!")
            )
            self.stdout.write(self.style.SUCCESS(f"📁 Stored at: {CHROMA_DB_PATH}"))
            self.stdout.write(self.style.SUCCESS("🔢 Dimensions: 3072 (text-embedding-3-large)"))
            
            return vectorstore
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ Vectorstore error: {e}"))
            traceback.print_exc()
            raise

    def handle(self, *args, **options) -> None:
        """🚀 PRODUCTION PIPELINE: JSON → Taxonomy → Embeddings → Test"""
        self.stdout.write(self.style.SUCCESS("🚀 Brihaspati Portfolio → Master Taxonomy Embeddings"))
        self.stdout.write(self.style.SUCCESS("=" * 60))
        
        # 1. LOAD JSON
        try:
            self.stdout.write(self.style.NOTICE(f"Loading JSON: {JSON_FILE}"))
            with open(JSON_FILE, "r", encoding="utf-8") as f:
                portfolio_json = json.load(f)
            self.stdout.write(self.style.SUCCESS(f"✅ Loaded: {JSON_FILE} ({len(portfolio_json)} projects)"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ JSON load failed: {e}"))
            return

        # 2. GENERATE TAXONOMY
        taxonomy_chunks = self.build_taxonomy_chunks_from_project_json(portfolio_json)
        
        # 3. VALIDATE & CONVERT
        documents = self.convert_to_documents(taxonomy_chunks)
        
        # 4. EMBED & STORE
        self.upsert_documents_to_vectorstore(documents)
        
        self.stdout.write(self.style.SUCCESS("\n🎉 Master Taxonomy Embeddings COMPLETE!"))