# rag_utils.py
import os
import glob
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import logging

logger = logging.getLogger(__name__)

# ==== GLOBAL CONFIGURATION ====
PROJECT_ROOT = r"C:/Users/vishn/Workspace/designathon"
CHROMA_PERSIST_PATH = os.path.join(PROJECT_ROOT, "chroma_db")
EXTENSIONS = ["*.java", "*.kt", "*.js", "*.jsx", "*.ts", "*.tsx"]

app_dirs = [
    os.path.join(PROJECT_ROOT, "banking-app"),
    os.path.join(PROJECT_ROOT, "core-bank-operations"), 
    os.path.join(PROJECT_ROOT, "reporting-services"),
    os.path.join(PROJECT_ROOT, "security-operations"),
]

# Technology classification
FRONTEND_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx"}
BACKEND_EXTENSIONS = {".java", ".kt"}

class RAGSystem:
    def __init__(self):
        self.is_initialized = False
        self.embed_model = None
        self.chroma_client = None
        self.chroma_collection = None
        
    def initialize(self):
        """Initialize RAG system components once during application startup"""
        if self.is_initialized:
            logger.info("RAG system already initialized")
            return
            
        try:
            logger.info("Initializing RAG system...")
            
            # Initialize embedding model
            self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✓ Embedding model loaded")
            
            # Initialize persistent ChromaDB client
            os.makedirs(CHROMA_PERSIST_PATH, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_PATH,
                settings=Settings(allow_reset=False)
            )
            logger.info("✓ ChromaDB client connected")
            
            # Always use get_or_create to avoid conflicts
            self.chroma_collection = self.chroma_client.get_or_create_collection("repo_chunks")
            
            # Check if collection is empty and needs initial indexing
            if self.chroma_collection.count() == 0:
                logger.info("✓ Empty collection - starting initial indexing")
                self._index_apps()
            else:
                logger.info("✓ Existing collection loaded")
            
            self.is_initialized = True
            logger.info("RAG system initialization complete!")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def update_index(self, force_reindex=False):
        """Update RAG index with latest code changes"""
        if not self.is_initialized:
            raise RuntimeError("RAG system not initialized")
            
        try:
            logger.info("Starting RAG index update...")
            
            if force_reindex:
                # Always use get_or_create, then clear contents
                self.chroma_collection = self.chroma_client.get_or_create_collection("repo_chunks")
                
                # Clear all existing documents
                try:
                    all_docs = self.chroma_collection.get(include=[])
                    if all_docs['ids']:
                        self.chroma_collection.delete(ids=all_docs['ids'])
                        logger.info(f"✓ Cleared {len(all_docs['ids'])} existing documents")
                    else:
                        logger.info("✓ Collection was already empty")
                except Exception as e:
                    logger.warning(f"Could not clear collection: {e}")
            
            self._index_apps()
            logger.info("RAG index update complete!")
            
        except Exception as e:
            logger.error(f"Failed to update RAG index: {e}")
            raise
    
    def _index_apps(self):
        """Enhanced indexing with better content filtering"""
        indexed_count = 0
        for app_dir in app_dirs:
            for ext in EXTENSIONS:
                files = glob.glob(os.path.join(app_dir, "**", ext), recursive=True)
                for file_path in files:
                    try:
                        # Skip certain directories
                        if any(skip_dir in file_path for skip_dir in ['node_modules', 'target', 'build', '.git', 'chroma_db']):
                            continue
                        
                        with open(file_path, "r", encoding="utf-8") as f:
                            code = f.read()
                        
                        if not code.strip() or len(code) < 50:  # Skip very small files
                            continue
                        
                        # Generate embedding with progress suppressed
                        embedding = self.embed_model.encode([code], show_progress_bar=False)[0]
                        relative_path = os.path.relpath(file_path, PROJECT_ROOT)
                        
                        # Enhanced metadata
                        file_ext = Path(file_path).suffix
                        path_parts = Path(file_path).parts
                        
                        metadata = {
                            "path": relative_path,
                            "file_extension": file_ext,
                            "technology": self._classify_technology(file_ext),
                            "app_name": path_parts[-4] if len(path_parts) > 3 else "unknown",
                            "file_size": len(code)
                        }
                        
                        self.chroma_collection.upsert(
                            embeddings=[embedding],
                            metadatas=[metadata],
                            documents=[code],
                            ids=[relative_path]
                        )
                        indexed_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Skipped: {file_path} ({e})")
        
        logger.info(f"✓ Indexed {indexed_count} files")
    
    def _classify_technology(self, file_extension):
        """Classify file by technology stack"""
        if file_extension in FRONTEND_EXTENSIONS:
            return "frontend"
        elif file_extension in BACKEND_EXTENSIONS:
            return "backend"
        else:
            return "general"
    
    def _detect_intent_from_prompt(self, user_prompt):
        """Enhanced technology detection from user prompt"""
        prompt_lower = user_prompt.lower()
        
        # More specific frontend keywords
        frontend_keywords = [
            "react", "jsx", "tsx", "frontend", "ui", "component", "input", "form", 
            "validation", "quick pay", "receiver", "upi", "interface", "client",
            "button", "field", "label", "submit", "onclick", "onchange", "state",
            "props", "render", "return", "const", "function", "hook", "usestate",
            "bootstrap", "css", "style", "class", "div", "span"
        ]
        
        # More specific backend keywords  
        backend_keywords = [
            "api", "endpoint", "service", "java", "spring", "controller", 
            "repository", "database", "server", "rest", "microservice",
            "@requestmapping", "@postmapping", "@getmapping", "@restcontroller",
            "@service", "@repository", "@autowired", "entity", "model",
            "hibernate", "jpa", "sql", "query", "transaction"
        ]
        
        # Count matches with weights
        frontend_score = 0
        backend_score = 0
        
        for keyword in frontend_keywords:
            if keyword in prompt_lower:
                # Give higher weight to UI-specific terms
                weight = 3 if keyword in ["ui", "component", "input", "form", "validation", "quick pay"] else 1
                frontend_score += weight
        
        for keyword in backend_keywords:
            if keyword in prompt_lower:
                # Give higher weight to API-specific terms
                weight = 3 if keyword in ["api", "endpoint", "service", "controller"] else 1
                backend_score += weight
        
        logger.info(f"Intent detection - Frontend score: {frontend_score}, Backend score: {backend_score}")
        
        if frontend_score > backend_score:
            return "frontend"
        elif backend_score > frontend_score:
            return "backend"
        else:
            return "general"
    
    def retrieve_context(self, user_prompt, top_k=5, file_filter=None):
        """Enhanced retrieval with better filtering and fallback"""
        if not self.is_initialized:
            raise RuntimeError("RAG system not initialized")
            
        # Auto-detect technology if not specified
        if file_filter is None:
            file_filter = self._detect_intent_from_prompt(user_prompt)
            logger.info(f"Auto-detected technology: {file_filter}")
        
        prompt_embedding = self.embed_model.encode([user_prompt], show_progress_bar=False)[0]
        
        # Try specific technology filter first
        where_clause = None
        if file_filter and file_filter != "general":
            where_clause = {"technology": {"$eq": file_filter}}
        
        results = self.chroma_collection.query(
            query_embeddings=[prompt_embedding],
            n_results=top_k,
            where=where_clause,
            include=['documents', 'metadatas', 'distances']
        )
        
        # If no results with filter, try without filter
        if not results['documents'] or not results['documents'][0]:
            logger.info("No results with technology filter, trying without filter")
            results = self.chroma_collection.query(
                query_embeddings=[prompt_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
        
        # Log retrieved files for debugging
        if results['metadatas'] and results['metadatas'][0]:
            retrieved_info = []
            for i, meta in enumerate(results['metadatas'][0]):
                distance = results['distances'][0][i] if results['distances'] else 0
                retrieved_info.append(f"{meta['path']} (tech: {meta.get('technology', 'unknown')}, distance: {distance:.3f})")
            
            logger.info(f"Retrieved {len(retrieved_info)} files:")
            for info in retrieved_info:
                logger.info(f"  - {info}")
        
        return results['documents'][0] if results['documents'] else []
    
    def retrieve_context_with_metadata(self, user_prompt, top_k=5, file_filter=None):
        """Retrieve context with metadata for debugging purposes"""
        if not self.is_initialized:
            raise RuntimeError("RAG system not initialized")
            
        # Auto-detect technology if not specified
        if file_filter is None:
            file_filter = self._detect_intent_from_prompt(user_prompt)
        
        prompt_embedding = self.embed_model.encode([user_prompt], show_progress_bar=False)[0]
        
        # Build where clause for filtering
        where_clause = None
        if file_filter and file_filter != "general":
            where_clause = {"technology": {"$eq": file_filter}}
        
        results = self.chroma_collection.query(
            query_embeddings=[prompt_embedding],
            n_results=top_k,
            where=where_clause,
            include=['documents', 'metadatas', 'distances']
        )
        
        return {
            "documents": results['documents'][0] if results['documents'] else [],
            "metadatas": results['metadatas'][0] if results['metadatas'] else [],
            "distances": results['distances'][0] if results['distances'] else [],
            "detected_technology": file_filter
        }
    
    def craft_prompt(self, user_prompt, retrieved_contexts, technology_hint=""):
        """Generate final prompt with context and technology-specific instructions"""
        context_snippet = "\n\n".join(retrieved_contexts)
        
        # Technology-specific instructions
        tech_instructions = ""
        if "frontend" in technology_hint.lower():
            tech_instructions = (
                "Focus on React/frontend components, JSX, state management, and UI validation.\n"
                "Use modern React patterns with functional components and hooks.\n"
            )
        elif "backend" in technology_hint.lower():
            tech_instructions = (
                "Focus on Java/Spring Boot implementation, REST APIs, services, and data validation.\n"
                "Use proper Spring annotations and follow Java best practices.\n"
            )
        
        return (
            "You are a senior software engineer AI assistant.\n\n" +
            tech_instructions +
            "Given the following Jira Story:\n" +
            f"{user_prompt}\n\n" +
            "And the following relevant code from the project:\n" +
            f"{context_snippet}\n\n" +
            "Your task is to suggest exact code changes to implement the requirement.\n" +
            "Your response must include:\n" +
            "- File path (relative if possible)\n" +
            "- Line number or a clear placement location (e.g., 'after method XYZ')\n" +
            "- A short description of what the change does\n" +
            "- The exact code to add, modify, or replace, enclosed in proper code blocks\n\n" +
            "```\n" +
            "---\n\n" +
            "Only include relevant and required code. Do not generate extra text or assumptions beyond the scope of the requirement.\n"
        )
    
    def get_stats(self):
        """Get RAG system statistics with technology breakdown"""
        if not self.is_initialized:
            return {"initialized": False}
            
        try:
            collection_count = self.chroma_collection.count()
            
            # Get technology breakdown
            all_docs = self.chroma_collection.get(include=['metadatas'])
            tech_counts = {}
            if all_docs['metadatas']:
                for meta in all_docs['metadatas']:
                    tech = meta.get('technology', 'unknown')
                    tech_counts[tech] = tech_counts.get(tech, 0) + 1
            
            return {
                "initialized": True,
                "total_documents": collection_count,
                "persist_path": CHROMA_PERSIST_PATH,
                "technology_breakdown": tech_counts
            }
        except Exception as e:
            return {"initialized": True, "error": str(e)}


# Global RAG system instance
rag_system = RAGSystem()

# Backward compatibility functions
def retrieve_context(user_prompt, top_k=3):
    return rag_system.retrieve_context(user_prompt, top_k)

def craft_prompt(user_prompt, retrieved_contexts):
    return rag_system.craft_prompt(user_prompt, retrieved_contexts)
