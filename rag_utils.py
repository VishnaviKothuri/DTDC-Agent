# rag_utils.py
"""
RAG helper utilities with enhanced debugging and multi-repository support
––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
• Clones / updates any number of Git repositories to a temp folder
• Builds a Chroma vector store (persistent) with embeddings of all
  source-code files that match the EXTENSIONS list
• Adds rich metadata (repo_name, technology, file path …)
• Enhanced debugging and per-repository logging
• Exposes:
      rag_system.initialize()
      rag_system.update_from_git(force_reindex=False)
      rag_system.retrieve_context(...)
      rag_system.get_stats()
      rag_system.verify_repositories()
      retrieve_context()  – back-compat wrapper
      craft_prompt()      – back-compat wrapper
"""

import os, glob, shutil, tempfile, logging
from pathlib import Path
import numpy as np                  # Required for embedding conversion
import git                          # pip install GitPython
import chromadb                     # pip install chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer   # pip install sentence-transformers

# ---------------------------------------------------------------------------
#  USER CONFIGURATION - UPDATE THESE WITH YOUR REPOSITORIES
# ---------------------------------------------------------------------------

# 1️⃣  Add / edit repositories here.
#     Replace URLs with your actual GitHub repositories
REPOSITORIES = [
    {
        "name": "banking-ui",
        "url": "https://github.com/VishnaviKothuri/banking-ui.git",
        "branch": "master",
        "app_dirs": [""]
    },
    {
        "name": "core-bank-operations",
        "url": "https://github.com/VishnaviKothuri/core-bank-operations.git",
        "branch": "main",
        "app_dirs": [""]  # Empty string means entire repo
    },
    {
        "name": "reporting-services",
        "url": "https://github.com/VishnaviKothuri/reporting-services.git",
        "branch": "main",
        "app_dirs": [""]
    },
    {
        "name": "security-operations",
        "url": "https://github.com/VishnaviKothuri/security-operations.git",
        "branch": "main",
        "app_dirs": [""]
    },
]

# 2️⃣  File extensions that should be embedded
EXTENSIONS = ["*.java", "*.kt", "*.js", "*.jsx", "*.ts", "*.tsx", "*.py", "*.sql"]

# 3️⃣  Where to keep things
TEMP_CLONE_BASE_DIR = os.path.join(tempfile.gettempdir(), "rag_multi_repo")
CHROMA_PERSIST_PATH = os.path.join(os.getcwd(), "chroma_db")

# ---------------------------------------------------------------------------

FRONTEND_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx"}
BACKEND_EXTENSIONS  = {".java", ".kt", ".py"}

logging.basicConfig(
    level=os.getenv("RAG_LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self):
        self.is_initialized    = False
        self.embed_model       = None
        self.chroma_client     = None
        self.chroma_collection = None

    # ---------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------
    def initialize(self):
        """Call once at application start-up."""
        if self.is_initialized:
            logger.info("RAG system already initialized")
            return

        logger.info("Initializing RAG system...")
        
        # Embedding model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("✓ SentenceTransformer loaded")

        # Chroma persistent DB
        os.makedirs(CHROMA_PERSIST_PATH, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_PATH,
            settings=Settings(allow_reset=False)
        )
        self.chroma_collection = self.chroma_client.get_or_create_collection("repo_chunks")
        logger.info("✓ ChromaDB collection ready (docs=%s)", self.chroma_collection.count())

        # First-time indexing if collection is empty
        if self.chroma_collection.count() == 0:
            logger.info("Empty collection – performing initial Git indexing")
            self._index_apps()

        self.is_initialized = True
        logger.info("RAG initialization complete")
    

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def debug_embedding_step_by_step(self):
        """Debug the exact embedding conversion process"""
        test_code = "public class Test { public static void main(String[] args) { } }"
        
        print("=== DEBUGGING EMBEDDING CONVERSION ===")
        
        # Step 1: Generate embedding
        emb = self.embed_model.encode([test_code], show_progress_bar=False)
        print(f"1. Raw embedding type: {type(emb)}")
        print(f"   Raw embedding shape: {emb.shape if hasattr(emb, 'shape') else 'No shape'}")
        print(f"   Raw embedding sample: {emb}")
        
        # Step 2: Convert based on type
        if isinstance(emb, np.ndarray):
            print(f"2. Processing numpy array with shape: {emb.shape}")
            if len(emb.shape) == 2:
                emb_list = emb[0].tolist()
                print(f"   Took first row: shape {emb[0].shape}")
            elif len(emb.shape) == 1:
                emb_list = emb.tolist()
                print(f"   Direct conversion")
            else:
                print(f"   ERROR: Unexpected shape")
                return
        else:
            print(f"2. ERROR: Not a numpy array: {type(emb)}")
            return
        
        # Step 3: Validate conversion
        print(f"3. Converted type: {type(emb_list)}")
        print(f"   Converted length: {len(emb_list) if isinstance(emb_list, list) else 'Not a list'}")
        print(f"   First 5 elements: {emb_list[:5] if isinstance(emb_list, list) else 'Cannot slice'}")
        print(f"   Element types: {[type(x) for x in emb_list[:3]] if isinstance(emb_list, list) else 'Cannot check'}")
        
        # Step 4: Force float conversion
        try:
            final_emb = [float(x) for x in emb_list]
            print(f"4. Final conversion successful")
            print(f"   Final type: {type(final_emb)}")
            print(f"   Final length: {len(final_emb)}")
            print(f"   Final sample: {final_emb[:5]}")
            print(f"   All floats? {all(isinstance(x, float) for x in final_emb[:10])}")
            
            # Step 5: Test ChromaDB compatibility
            try:
                # Try to create a temporary test document
                test_collection = self.chroma_client.get_or_create_collection("debug_test")
                test_collection.upsert(
                    ids=["test"],
                    embeddings=[final_emb],
                    documents=["test code"],
                    metadatas=[{"test": "true"}]
                )
                print("5. ✅ ChromaDB upsert successful!")
                
                # Clean up
                self.chroma_client.delete_collection("debug_test")
                
            except Exception as e:
                print(f"5. ❌ ChromaDB upsert failed: {e}")
                
        except Exception as e:
            print(f"4. ❌ Float conversion failed: {e}")
            
        print("=== END DEBUG ===")
        return emb


    def validate_embeddings(self):
        """Validation that accepts both lists and numpy arrays"""
        if not self.is_initialized:
            raise RuntimeError("RAG system not initialized")
        
        all_data = self.chroma_collection.get(include=['embeddings'])
        all_ids = all_data.get('ids', [])
        all_embs = all_data.get('embeddings', [])
        
        bad_ids = []
        for doc_id, emb in zip(all_ids, all_embs):
            # Accept both lists and numpy arrays
            if isinstance(emb, np.ndarray):
                # Numpy arrays are valid - convert to list for validation
                try:
                    emb_as_list = emb.tolist()
                    if not all(isinstance(x, (int, float)) for x in emb_as_list[:5]):
                        bad_ids.append(f"{doc_id} - numpy array with non-numeric values")
                except Exception as e:
                    bad_ids.append(f"{doc_id} - numpy array conversion failed: {e}")
            elif isinstance(emb, list):
                if len(emb) == 0:
                    bad_ids.append(f"{doc_id} - empty list")
                elif not all(isinstance(x, (int, float)) for x in emb[:5]):
                    types_found = [type(x) for x in emb[:5]]
                    bad_ids.append(f"{doc_id} - bad types: {types_found}")
            else:
                bad_ids.append(f"{doc_id} - invalid type: {type(emb)}")
        
        return {
            'total_documents': len(all_ids),
            'invalid_embeddings': len(bad_ids),
            'invalid_details': bad_ids[:10]
        }



    def update_from_git(self, force_reindex: bool = False):
        """
        Pull latest commits from every repo and update the vector store.
        • force_reindex=True  → wipes the collection before re-adding everything
        """
        if not self.is_initialized:
            raise RuntimeError("RAG system not initialized")

        logger.info("Updating RAG index from Git (force_reindex=%s)…", force_reindex)

        if force_reindex:
            ids = self.chroma_collection.get(include=[]).get("ids", [])
            if ids:
                self.chroma_collection.delete(ids=ids)
                logger.info("✓ cleared %s existing documents", len(ids))

        self._index_apps()
        logger.info("Update complete")

    def retrieve_context(
        self,
        user_prompt: str,
        top_k: int = 5,
        file_filter: str | None = None,
        repo_filter: str | list[str] | None = None,
    ):
        """
        Retrieve relevant code snippets.
        - file_filter   → "frontend" | "backend" | "general"
        - repo_filter   → repo name str  or list[str]
        Returns a list[str] with the raw code documents.
        """
        if not self.is_initialized:
            raise RuntimeError("RAG system not initialized")
        

        if file_filter is None:
            file_filter = self._detect_intent_from_prompt(user_prompt)
            logger.info("Auto-detected technology: %s", file_filter)

        # Generate prompt embedding with proper conversion
        prompt_emb = self.embed_model.encode([user_prompt], show_progress_bar=False)
        
        # Convert numpy array to list for ChromaDB
        if isinstance(prompt_emb, np.ndarray):
            if len(prompt_emb.shape) == 2:
                prompt_emb = prompt_emb[0].tolist()
            else:
                prompt_emb = prompt_emb.tolist()

        # Build where-clause
        clauses = []
        if file_filter == "backend":
            clauses.append({"technology": {"$eq": "backend"}})
        elif file_filter and file_filter != "general":
            clauses.append({"technology": {"$eq": file_filter}})
        # clauses = []
        # if file_filter and file_filter != "general":
        #     clauses.append({"technology": {"$eq": file_filter}})
        # if repo_filter:
        #     if isinstance(repo_filter, str):
        #         clauses.append({"repo_name": {"$eq": repo_filter}})
        #     else:  # list
        #         clauses.append({"repo_name": {"$in": repo_filter}})

        where = None
        if len(clauses) == 1:
            where = clauses[0]
        elif clauses:
            where = {"$and": clauses}

        results = self.chroma_collection.query(
            query_embeddings=[prompt_emb],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            logger.info("No hits with filter – retrying without any filter")
            results = self.chroma_collection.query(
                query_embeddings=[prompt_emb],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

        # log retrieved items
        if results["metadatas"] and results["metadatas"][0]:
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                logger.info(
                    "→ %s/%s  (tech=%s  dist=%.3f)",
                    meta.get("repo_name"), meta.get("path"),
                    meta.get("technology"), dist,
                )

        return (results["documents"][0] if results["documents"] else [], file_filter)

    def retrieve_context_with_metadata(
        self, user_prompt: str, top_k: int = 5, **kw
    ):
        """Same as retrieve_context but returns the full Chroma payload."""
        prompt_emb = self.embed_model.encode([user_prompt], show_progress_bar=False)
        
        # Convert numpy array to list
        if isinstance(prompt_emb, np.ndarray):
            if len(prompt_emb.shape) == 2:
                prompt_emb = prompt_emb[0].tolist()
            else:
                prompt_emb = prompt_emb.tolist()
        
        res = self.chroma_collection.query(
            query_embeddings=[prompt_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return res

    def craft_prompt(
        self,
        user_prompt: str,
        retrieved_contexts: list[str],
        technology_hint: str | None = "",
    ):
        """Compose final LLM prompt."""
        context_snippet = "\n\n".join(retrieved_contexts[:5])

        tech_instr = ""
        if technology_hint and "frontend" in technology_hint.lower():
            tech_instr = (
                "Focus on React/frontend components, JSX, state management, and UI validation.\n"
            )
        elif technology_hint and "backend" in technology_hint.lower():
            tech_instr = (
                "Focus on Java/Spring Boot implementation, REST APIs, services, and data validation.\n"
            )

        return (
            "You are a senior software-engineer AI assistant.\n\n"
            + tech_instr
            + "Given the following Jira Story:\n"
            + user_prompt
            + "\n\nAnd the following relevant code from the project:\n"
            + context_snippet
            + "\n\nYour task is to suggest exact code changes to implement the requirement and provide information on which existing class(es) will be modified.\n"
            + "Your response must include:\n"
            + "- File path (relative)\n"
            + "- Line number or clear placement location\n"
            + "- Short description\n"
            + "- Exact code to add / modify / replace (in code blocks)\n\n"
            + "```"
            + "---\n\n"
            + "Only include relevant and required code. Do not generate extra text or assumptions beyond the scope of the requirement.\n"
        )

    def get_stats(self):
        """Return detailed collection, technology, and repository statistics."""
        if not self.is_initialized:
            return {"initialized": False}

        try:
            data = self.chroma_collection.get(include=["metadatas"])
            repo_cnt, tech_cnt, ext_cnt = {}, {}, {}

            for m in data.get("metadatas", []):
                # Repository breakdown
                repo = m.get("repo_name", "unknown")
                repo_cnt[repo] = repo_cnt.get(repo, 0) + 1
                
                # Technology breakdown
                tech = m.get("technology", "unknown")
                tech_cnt[tech] = tech_cnt.get(tech, 0) + 1
                
                # Extension breakdown
                ext = m.get("file_extension", "unknown")
                ext_cnt[ext] = ext_cnt.get(ext, 0) + 1

            return {
                "initialized": True,
                "total_documents": self.chroma_collection.count(),
                "repository_breakdown": repo_cnt,
                "technology_breakdown": tech_cnt,
                "extension_breakdown": ext_cnt,
                "persist_path": CHROMA_PERSIST_PATH,
                "configured_repositories": [cfg["name"] for cfg in REPOSITORIES]
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"initialized": True, "error": str(e)}

    # ---------------------------------------------------------------------
    # Debug and Verification Methods
    # ---------------------------------------------------------------------
    def verify_repositories(self):
        """Verify all repositories are accessible and show their structure"""
        repo_paths = self._clone_or_update_repos()
        
        verification_results = {}
        
        for cfg in REPOSITORIES:
            repo_name = cfg["name"]
            repo_url = cfg["url"]
            
            if repo_name not in repo_paths:
                verification_results[repo_name] = {
                    "status": "FAILED",
                    "error": "Could not clone/access repository",
                    "files_count": 0
                }
                logger.error(f"❌ {repo_name}: Repository not accessible")
                continue
            
            repo_root = repo_paths[repo_name]
            total_files = 0
            
            # Count files by extension
            extension_counts = {}
            
            for pattern in EXTENSIONS:
                files = list(glob.glob(os.path.join(repo_root, "**", pattern), recursive=True))
                # Filter out excluded directories
                files = [f for f in files if not any(x in f for x in (".git", "node_modules", "target", "build", "chroma_db", "__pycache__"))]
                extension_counts[pattern] = len(files)
                total_files += len(files)
            
            verification_results[repo_name] = {
                "status": "OK",
                "files_count": total_files,
                "extension_breakdown": extension_counts,
                "repo_path": repo_root
            }
            
            logger.info(f"✅ {repo_name}: {total_files} files found")
            for ext, count in extension_counts.items():
                if count > 0:
                    logger.info(f"   {ext}: {count} files")
        
        return verification_results

    def debug_single_file(self, file_path):
        """Debug processing of a single file"""
        try:
            if not os.path.exists(file_path):
                return f"File doesn't exist: {file_path}"
                
            code = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            logger.info(f"File: {file_path}")
            logger.info(f"Size: {len(code)} characters")
            logger.info(f"Lines: {len(code.splitlines())}")
            logger.info(f"First 200 chars: {repr(code[:200])}")
            
            if len(code.strip()) < 10:
                return f"File too small: {len(code)} chars"
                
            # Test embedding generation
            emb = self.embed_model.encode([code[:1000]], show_progress_bar=False)
            logger.info(f"Embedding shape: {emb.shape}")
            
            return "File processing successful"
            
        except Exception as e:
            logger.error(f"Debug error: {e}")
            return f"Error: {e}"

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    def _index_apps(self):
        """Clone / pull all repos and embed code with proper embedding conversion."""
        repo_paths = self._clone_or_update_repos()
        if not repo_paths:
            logger.error("No repositories available – skipping indexing")
            return

        total_indexed, total_skipped = 0, 0
        repo_stats = {}

        for cfg in REPOSITORIES:
            repo_name = cfg["name"]
            
            if repo_name not in repo_paths:
                logger.error("Skipping %s – clone / pull failed earlier", repo_name)
                continue
                
            repo_root = repo_paths[repo_name]
            repo_indexed = 0
            repo_skipped = 0
            
            logger.info(f"🔍 Processing repository: {repo_name}")

            for sub in cfg.get("app_dirs", [""]):
                if sub == "":
                    dir_abs = repo_root
                    logger.info(f"  📁 Scanning entire repository: {repo_name}")
                else:
                    dir_abs = os.path.join(repo_root, sub)
                    logger.info(f"  📁 Scanning subdirectory: {repo_name}/{sub}")
                
                if not os.path.isdir(dir_abs):
                    logger.warning("  ⚠️  Missing dir %s – skipped", dir_abs)
                    continue

                for pattern in EXTENSIONS:
                    files_found = list(glob.glob(os.path.join(dir_abs, "**", pattern), recursive=True))
                    if files_found:
                        logger.info(f"  🔎 Found {len(files_found)} {pattern} files in {repo_name}")
                    
                    for fp in files_found:
                        if any(x in fp for x in (".git", "node_modules", "target", "build", "chroma_db", "__pycache__")):
                            continue
                            
                        try:
                            if not os.path.exists(fp):
                                continue
                                
                            code = Path(fp).read_text(encoding="utf-8", errors="ignore")
                            
                            if len(code.strip()) < 10:
                                continue

                            # Generate embedding
                            emb = self.embed_model.encode([code], show_progress_bar=False)
                            
                            # 🔧 SIMPLIFIED FIXED CONVERSION
                            if isinstance(emb, np.ndarray):
                                if len(emb.shape) == 2:  # Shape: (1, embedding_dim)
                                    emb_list = emb[0].tolist()  # Take first row, convert to list
                                elif len(emb.shape) == 1:  # Shape: (embedding_dim,)
                                    emb_list = emb.tolist()  # Convert directly to list
                                else:
                                    logger.error(f"Unexpected embedding shape: {emb.shape}")
                                    continue
                            else:
                                logger.error(f"Expected numpy array, got: {type(emb)}")
                                continue

                            # 🔧 VALIDATE THE CONVERSION
                            if not isinstance(emb_list, list) or len(emb_list) == 0:
                                logger.error(f"Invalid embedding format: {type(emb_list)}, length: {len(emb_list) if isinstance(emb_list, list) else 'N/A'}")
                                continue

                            # 🔧 ENSURE ALL ELEMENTS ARE NUMBERS
                            try:
                                emb_list = [float(x) for x in emb_list]  # Force convert to floats
                            except (ValueError, TypeError) as ve:
                                logger.error(f"Cannot convert embedding elements to float: {ve}")
                                continue

                            # Now emb_list should be [0.123, -0.456, 0.789, ...]
                            if repo_indexed < 3:  # Debug first few files
                                logger.debug(f"Embedding sample for {os.path.basename(fp)}: {emb_list[:5]}...")
                                
                            rel_path = os.path.relpath(fp, repo_root)
                            uid = f"{repo_name}::{rel_path}"

                            meta = {
                                "repo_name": repo_name,
                                "path": rel_path,
                                "file_extension": Path(fp).suffix,
                                "technology": self._classify_technology(Path(fp).suffix),
                            }

                            self.chroma_collection.upsert(
                                ids=[uid],
                                embeddings=[emb_list],
                                documents=[code],
                                metadatas=[meta],
                            )
                            repo_indexed += 1
                            total_indexed += 1
                            
                            # Show first few files per repo
                            if repo_indexed <= 3:
                                logger.info(f"  ✅ Indexed: {rel_path}")
                                
                        except Exception as exc:
                            repo_skipped += 1
                            total_skipped += 1
                            if repo_skipped <= 3:
                                logger.error(f"  ❌ Failed: {os.path.basename(fp)} - {exc}")

            # Per-repository summary
            repo_stats[repo_name] = {"indexed": repo_indexed, "skipped": repo_skipped}
            logger.info(f"✅ {repo_name}: {repo_indexed} indexed, {repo_skipped} skipped")

        # Overall summary
        logger.info(f"🎯 TOTAL: {total_indexed} indexed, {total_skipped} skipped")
        logger.info("📊 Repository breakdown:")
        for repo_name, stats in repo_stats.items():
            logger.info(f"   {repo_name}: {stats['indexed']} files")



    def _clone_or_update_repos(self) -> dict[str, str]:
        """Return {repo_name: local_path} after ensuring all repos are up to date."""
        os.makedirs(TEMP_CLONE_BASE_DIR, exist_ok=True)
        repo_paths = {}

        for cfg in REPOSITORIES:
            name, url, branch = cfg["name"], cfg["url"], cfg.get("branch", "main")
            dst = os.path.join(TEMP_CLONE_BASE_DIR, name)

            try:
                if os.path.isdir(dst):
                    logger.info("Pulling latest for %s", name)
                    repo = git.Repo(dst)
                    repo.git.checkout(branch)
                    repo.remotes.origin.pull()
                else:
                    logger.info("Cloning %s → %s", url, dst)
                    git.Repo.clone_from(url, dst, branch=branch, depth=1)
                repo_paths[name] = dst
            except Exception as exc:
                logger.error("✗ could not clone/pull %s (%s)", name, exc)

        return repo_paths

    # ---------------------------------------------------------------------
    @staticmethod
    def _classify_technology(ext: str) -> str:
        if ext in FRONTEND_EXTENSIONS:
            return "frontend"
        if ext in BACKEND_EXTENSIONS:
            return "backend"
        return "general"

    # ---------------------------------------------------------------------
    # In rag_utils.py, replace your _detect_intent_from_prompt with:

    def _detect_intent_from_prompt(self, prompt: str) -> str:
        p = prompt.lower()

        frontend_terms = ["react","jsx","tsx","component","ui","form","input","label"]
        backend_terms  = [
            "java","spring","controller","service","repository",
            "endpoint","api","transaction","transfer","payment","fee"
        ]

        f_score = sum(term in p for term in frontend_terms)
        b_score = sum(term in p for term in backend_terms)

        if f_score == b_score == 0:
            return "general"
        return "backend" if b_score > f_score else "frontend"


# ---------------------------------------------------------------------------
# Singleton instance + back-compat wrappers
# ---------------------------------------------------------------------------
rag_system = RAGSystem()

def retrieve_context(user_prompt, top_k=3):
    return rag_system.retrieve_context(user_prompt, top_k)

def craft_prompt(
        self,
        user_prompt: str,
        retrieved_contexts: list[str],
        technology_hint: str = "general",
    ):
        """
        Build the final LLM prompt.
        technology_hint: 'frontend', 'backend' or 'general'
        """
        context = "\n\n".join(retrieved_contexts[:5])

        if technology_hint == "frontend":
            tech_header = (
                "Focus ONLY on React / Typescript / JSX files.\n"
                "Ignore Java, Spring, SQL and configuration classes.\n"
            )
        elif technology_hint == "backend":
            tech_header = (
                "Focus ONLY on Java / Spring-Boot files (controllers, services, repositories).\n"
                "Ignore React components, CSS and client-side validation.\n"
            )
        else:
            tech_header = ""

        return f"""You are a senior full-stack engineer AI assistant.

    {tech_header}
    Jira task:
    {user_prompt}

    Relevant code snippets:
    {context}

    TASK:
    Provide exact code changes.

    FORMAT:
    - File path
    - Line numbers or anchor comment
    - BEFORE and AFTER code blocks in ```language fences
    - One short explanation for each file

    Respond **only** with the required diffs, no extra text.
    """