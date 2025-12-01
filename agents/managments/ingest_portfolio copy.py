import re
import json
import requests

from typing import List, Dict, Optional, Set
from datetime import datetime
from bs4 import BeautifulSoup, NavigableString, Tag

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHROMA_DB_PATH = BASE_DIR / "agents/chroma_db"

llm = ChatOpenAI(model="gpt-4o-mini")  # or your model


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " "],
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = Chroma(
    collection_name="tbi_portfolio",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DB_PATH),
)

class AdvancedPortfolioScraper:
    """
    Robust scraper that dynamically detects sections and formats data 
    specifically for RAG (Retrieval Augmented Generation) systems.
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Expanded tech list for better recall
        self.tech_keywords = {
            "Frontend": ["React", "React.js", "Vue", "Vue.js", "Angular", "Next.js", "Nuxt", "Tailwind", "Bootstrap", "Redux"],
            "Backend": ["Node.js", "Node", "Express", "Python", "Django", "Flask", "PHP", "Laravel", "Java", "Spring"],
            "Database": ["MongoDB", "PostgreSQL", "MySQL", "Redis", "Firebase", "Supabase"],
            "Infrastructure": ["AWS", "Docker", "Kubernetes", "Azure", "GCP", "Nginx", "CI/CD"],
            "CMS/Ecom": ["Shopify", "WordPress", "Magento", "WooCommerce", "BigCommerce", "Liquid"],
            "API": ["GraphQL", "REST", "Stripe", "PayPal", "Google Maps API", "Twilio"]
        }

    def load_page(self, url: str) -> Optional[BeautifulSoup]:
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            print(f"❌ Error loading {url}: {e}")
            return None

    def clean_text(self, text: str) -> str:
        """Removes extra whitespace and newlines."""
        return re.sub(r'\s+', ' ', text).strip()

    def extract_technologies(self, full_text: str) -> Dict[str, List[str]]:
        """Scans the full page text for known tech keywords."""
        found_tech = {}
        full_text_lower = full_text.lower()
        
        for category, terms in self.tech_keywords.items():
            found_in_category = []
            for term in terms:
                # Use word boundary \b so "Go" doesn't match "Google"
                if re.search(r'\b' + re.escape(term.lower()) + r'\b', full_text_lower):
                    found_in_category.append(term)
            
            if found_in_category:
                found_tech[category] = list(set(found_in_category))
                
        return found_tech

    def extract_dynamic_sections(self, soup: BeautifulSoup) -> List[Dict]:
        """
        The Core Intelligence:
        Iterates through the document flow to find Headers and their associated Content.
        Does not rely on hardcoded IDs.
        """
        sections = []
        
        # 1. Identify the main content container to avoid footer/header noise
        # This is specific to the site structure but falls back to body
        main_content = soup.find('div', class_='portfolio-single') or soup.find('main') or soup.body
        
        if not main_content:
            return []

        # 2. Find all potential headers
        # We look for h1-h6 OR paragraphs with specific class names used as headers
        headers = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'span'])
        
        current_section = {"heading": "Introduction", "content": []}
        
        for element in headers:
            # Check if this element is acting as a header
            is_header = False
            
            # Criteria for being a header in this specific website
            if element.name in ['h1', 'h2', 'h3', 'h4']:
                is_header = True
            elif element.name == 'p' and ('casestudy-phead' in element.get('class', [])):
                is_header = True
            elif element.name == 'span' and ('main-list' in element.get('class', [])):
                is_header = True

            if is_header:
                # Save previous section if it has content
                if current_section["content"]:
                    sections.append({
                        "heading": self.clean_text(current_section["heading"]),
                        "content": " ".join(current_section["content"])
                    })
                
                # Start new section
                current_section = {
                    "heading": self.clean_text(element.get_text()), 
                    "content": []
                }
                
                # Capture the content immediately following this header
                # We iterate siblings until we hit the next header-like element
                for sibling in element.next_siblings:
                    if isinstance(sibling, Tag):
                        # Stop if sibling is another header
                        if sibling.name in ['h1', 'h2', 'h3', 'h4'] or \
                           ('casestudy-phead' in sibling.get('class', [])):
                            break
                        
                        # Extract text from p, ul, div
                        text = self.clean_text(sibling.get_text())
                        if text:
                            current_section["content"].append(text)
                            
        # Append the final section
        if current_section["content"]:
            sections.append({
                "heading": self.clean_text(current_section["heading"]),
                "content": " ".join(current_section["content"])
            })

        return sections

    def scrape_portfolio(self, url: str) -> Optional[Dict]:
        print(f"🕷️ Scraping: {url}")
        soup = self.load_page(url)
        
        if not soup:
            return None

        # 1. Basic Metadata
        title_tag = soup.find('h1') or soup.find(class_='title')
        title = self.clean_text(title_tag.get_text()) if title_tag else "Unknown Project"
        
        # 2. Extract Generic Sections (The "Safe" Way)
        sections = self.extract_dynamic_sections(soup)
        
        # 3. Extract Full Text for Tech Scanning
        full_text = " ".join([s['content'] for s in sections]) + " " + title
        tech_stack = self.extract_technologies(full_text)

        # 4. Construct RAG-Optimized Document
        # We create a single markdown string that is perfect for embedding.
        rag_context = f"# Project: {title}\n\n"
        rag_context += f"## Tech Stack Detected\n{json.dumps(tech_stack, indent=2)}\n\n"
        
        for section in sections:
            # Skip empty or redundant sections
            if len(section['content']) < 10: continue
            rag_context += f"## {section['heading']}\n{section['content']}\n\n"

        return {
            "url": url,
            "title": title,
            "tech_stack": tech_stack,
            "sections": sections,
            "rag_context": rag_context, # <--- THIS IS WHAT YOU EMBED
            "timestamp": datetime.now().isoformat()
        }

    def save_to_json(self, data: List[Dict], filename: str):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Data saved to {filename}")

def section_to_chunk_docs(section_doc: Document) -> list[Document]:
    # for item in section_doc:
        chunks = text_splitter.split_text(section_doc.page_content)
        chunk_docs = []

        for i, chunk in enumerate(chunks):
            md = section_doc.metadata.copy()
            md.update({
                "chunk_index": i,
                "chunk_id": f"{md['project_title']}::chunk-{i}",
            })

            chunk_docs.append(
                Document(
                    page_content=chunk,
                    metadata=md,
                )
            )
        return chunk_docs


def summarize_section(section_doc: Document) -> Document:
    prompt = f"""
    Summarize the following project section in 2–3 sentences, focusing on
    what was built and why it matters to the client.

    Section title: {section_doc.metadata.get('section_heading')}
    Content:
    {section_doc.page_content}
    """
    summary = llm.invoke(prompt).content.strip()

    md = section_doc.metadata.copy()
    md.update({
        "is_summary": True,
        "chunk_index": 0,
        # "chunk_id": f"{md['section_id']}::summary",
    })

    return Document(page_content=summary, metadata=md)


def upsert_section(section_doc: Document):
    # Make chunks
    chunk_docs = section_to_chunk_docs(section_doc)
    summary_doc = summarize_section(section_doc)

    all_docs = chunk_docs + [summary_doc]

    # ids = [d.metadata["chunk_id"] for d in all_docs]

    print("ALL DOCUMENT", all_docs)
    print("ALL DOCUMENT", len(all_docs))

    # vectorstore.add_documents(
    #     documents=all_docs,
    #     ids=ids,
    # )
    # vectorstore.persist()


def load_and_process_data():
    urls = [
        "https://www.brihaspatitech.com/portfolio/saas-platform-for-corporate-phishing-training-simulation/"
    ]

    scraper = AdvancedPortfolioScraper()
    results = []

    print(f"\n🚀 Starting Advanced Extraction...")
    for url in urls:
        data = scraper.scrape_portfolio(url)
        if data:
            results.append(data)

    processed_docs = []

    for project in results:
        # 1. Extract Global Project Metadata
        # We attach this to EVERY chunk so we can filter later (e.g., "Show me only React projects")
        
        # Flatten the tech stack dict into a single list for metadata filtering
        # From: {"Frontend": ["React"], "Backend": ["Node"]} 
        # To: "React, Node"
        all_tech = []
        if isinstance(project.get("tech_stack"), dict):
            for category, items in project["tech_stack"].items():
                all_tech.extend(items)
        flat_tech_string = ", ".join(all_tech)

        project_title = project.get("title", "Unknown")
        project_url = project.get("url", "")

        # 2. STRATEGY: Semantic Section Chunking
        # Instead of one big doc, we create a specific doc for EACH section.
        
        for section in project.get("sections", []):
            section_heading = section.get("heading", "")
            section_content = section.get("content", "")

            # Skip noise (empty sections)
            if len(section_content) < 15:
                continue

            # 3. CRITICAL: Context Injection
            # A chunk saying "We used AWS Lambda" is useless without knowing WHICH project it was.
            # We prepend the Project Title to the content.
            
            clean_content = re.sub(r'\s+', ' ', section_content).strip()

            # 2. Skip "Call to Action" sections (Noise Filtering)
            # You don't want the AI to index "Ready to Discuss?" or "Contact Us" 
            # because it dilutes real technical answers.
            noise_headers = ["Ready to Discuss?", "Get Started", "Contact Us", "Newsletter"]
            if any(noise in section_heading for noise in noise_headers):
                continue 

            enhanced_content = f"""
            Project: {project_title}
            Topic: {section_heading}
            Tech: {flat_tech_string}
            -----------------------
            {clean_content}
            """

            project_slug = project_url.split("/")[-2]

            # 4. Create the Document
            doc = Document(
                page_content=enhanced_content, # The AI reads this
                 metadata={
                    # Document-level identifiers
                    "doc_id": f"portfolio::{project_slug}",           # e.g. "portfolio::saas-phishing-training"
                    "source": project_url,
                    "project_title": project_title,
                    "project_slug": project_slug,                     # slugified title
                    "section_heading": section_heading,                 # human-readable

                    # Domain-specific filters
                    # "tech_stack": tech_stack_list,                    # list of techs, not just flat string
                    # "industry": industry,
                    # "client_name": client_name,
                    # "tags": tags_list,                                # e.g. ["ReactJs", "NodeJs", "Cybersecurity"]

                    # Control fields
                    "is_summary": False,
                    "source_type": "portfolio",
                    "version": 1,
                }
            )
            processed_docs.append(doc)

    upsert_section(processed_docs[0])
    return processed_docs


load_and_process_data()