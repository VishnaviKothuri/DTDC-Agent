import os
import glob
from sentence_transformers import SentenceTransformer
import chromadb

# ==== PATH SETUP ====
PROJECT_ROOT = r"C:/Users/vishn/Workspace/designathon"
app_dirs = [
    os.path.join(PROJECT_ROOT, "banking-app"),
    os.path.join(PROJECT_ROOT, "core-bank-operations"),
    os.path.join(PROJECT_ROOT, "reporting-services"),
    os.path.join(PROJECT_ROOT, "security-operations"),
]

# ==== SUPPORTED EXTENSIONS ====
EXTENSIONS = ["*.java", "*.kt", "*.js", "*.jsx", "*.ts", "*.tsx"]

# ==== INITIALIZE EMBEDDING MODEL & CHROMA ====
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("repo_chunks")
print
# ==== INDEXING FUNCTION ====
def index_apps(app_dirs):
    for app_dir in app_dirs:
        for ext in EXTENSIONS:
            files = glob.glob(os.path.join(app_dir, "**", ext), recursive=True)
            for file_path in files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        code = f.read()
                    embedding = embed_model.encode([code])[0]
                    chroma_collection.add(
                        embeddings=[embedding],
                        metadatas=[{"path": os.path.relpath(file_path, PROJECT_ROOT)}],
                        documents=[code]
                    )
                    print(f"Indexed: {file_path}")
                except Exception as e:
                    print(f"Skipped: {file_path} ({e})")

# ==== CONTEXT RETRIEVAL ====
def retrieve_context(user_prompt, top_k=3):
    prompt_embedding = embed_model.encode([user_prompt])[0]
    results = chroma_collection.query(
        query_embeddings=[prompt_embedding],
        n_results=top_k
    )
    return results['documents'][0]

# ==== PROMPT GENERATION ====
def craft_prompt(user_prompt, retrieved_contexts):
    context_snippet = "\n\n".join(retrieved_contexts)
    return (
        f"You are an expert developer. Given this Jira Story:\n"
        f"{user_prompt}\n\n"
        f"And the following relevant code from the project:\n"
        f"{context_snippet}\n\n"
        "Generate implementation code and corresponding unit tests."
    )

# ==== USAGE EXAMPLE ====
if __name__ == "__main__":
    # 1. Index all apps (run this when code changes)
    index_apps(app_dirs)

    # 2. Handle a user prompt (replace with your real Jira story)
    jira_story = "Implement user registration endpoint for the API."
    contexts = retrieve_context(jira_story, top_k=3)

    # 3. Craft a prompt for the language model
    final_prompt = craft_prompt(jira_story, contexts)
    print("--- PROMPT TO SEND TO LLM ---")
    print(final_prompt)
