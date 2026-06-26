import json
import os
import sys
import subprocess
import numpy as np

# Ensure necessary packages are installed
def install_dependencies():
    packages = ["sentence-transformers", "torch", "numpy", "tqdm"]
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

install_dependencies()

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

def build_candidate_text(cand):
    profile = cand.get("profile", {})
    skills = cand.get("skills", [])
    
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    skill_names = [s.get("name", "") for s in skills]
    skills_str = ", ".join(skill_names)
    
    # We build a structured text summary for embeddings matching
    text = f"Headline: {headline}. Summary: {summary}. Skills: {skills_str}."
    return text

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates_path = os.path.join(script_dir, "candidates.jsonl")
    if not os.path.exists(candidates_path):
        # Fallback to parent directory if running from inside Submission/
        candidates_path = os.path.join(os.path.dirname(script_dir), "candidates.jsonl")
        
    output_dir = os.path.join(script_dir, "vector_embeddings")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "candidate_embeddings.npz")
    
    if not os.path.exists(candidates_path):
        print(f"Error: {candidates_path} not found.")
        sys.exit(1)
        
    print("Loading candidate records...")
    candidates = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
                
    print(f"Loaded {len(candidates)} candidates.")
    
    # Pre-filtering candidates down to target technical profiles
    print("Pre-filtering candidates to optimize encoding time...")
    filtered_candidates = []
    for cand in candidates:
        profile = cand.get("profile", {})
        yoe = profile.get("years_of_experience", 0)
        if yoe < 3.0:
            continue
        title = profile.get("current_title", "").lower()
        non_tech = ["marketing", "sales", "accountant", "hr manager", "human resources", "ops manager", "operations manager", "customer support", "designer", "writer", "mechanical", "civil", "financial", "admin", "recruiter", "legal"]
        if any(nt in title for nt in non_tech):
            continue
        filtered_candidates.append(cand)
        
    print(f"Pre-filtered {len(candidates)} candidates down to {len(filtered_candidates)} high-potential candidates for encoding.")
    
    # Construct texts for filtered candidates
    texts = [build_candidate_text(c) for c in filtered_candidates]
    candidate_ids = [c["candidate_id"] for c in filtered_candidates]
    
    print("Initializing SentenceTransformer model (all-MiniLM-L6-v2)...")
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    
    # Save model weights locally
    model_weights_path = os.path.join(output_dir, "model_weights")
    os.makedirs(model_weights_path, exist_ok=True)
    model.save(model_weights_path)
    print(f"Saved model weights to {model_weights_path}")
    
    # Generate JD embedding
    jd_embedding_path = os.path.join(output_dir, "jd_embedding.npy")
    jd_text = ""
    txt_path = os.path.join(os.path.dirname(script_dir), "job_description.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            jd_text = f.read().strip()
    else:
        docx_path = os.path.join(os.path.dirname(script_dir), "job_description.docx")
        if os.path.exists(docx_path):
            try:
                import zipfile
                import xml.etree.ElementTree as ET
                with zipfile.ZipFile(docx_path) as docx:
                    xml_content = docx.read('word/document.xml')
                    root = ET.fromstring(xml_content)
                    paragraphs = []
                    for para in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                        texts_t = [node.text for node in para.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text]
                        if texts_t:
                            paragraphs.append("".join(texts_t))
                    jd_text = "\n".join(paragraphs).strip()
            except Exception as e:
                print(f"Error reading docx: {e}")
                
    if jd_text:
        print("Encoding Job Description...")
        jd_vector = model.encode(jd_text, convert_to_numpy=True)
        np.save(jd_embedding_path, jd_vector)
        print(f"Successfully saved JD embedding to {jd_embedding_path}")
    
    print("Encoding candidate texts in batches (this pre-computation is done offline)...")
    # Encode in batches
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"Saving embeddings of shape {embeddings.shape} to {output_path}...")
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        candidate_ids=np.array(candidate_ids)
    )
    print("Successfully saved pre-computed embeddings!")

if __name__ == "__main__":
    main()
