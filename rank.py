import random
import os
import sys
import argparse
import json
import csv
import re
from datetime import datetime
import numpy as np

# Inlined reasoning generator code
def get_candidate_numeric_id(cid):
    """Extracts the numeric part of CAND_XXXXXXX to use as a seed."""
    match = re.search(r'\d+', cid)
    if match:
        return int(match.group())
    return 42

def generate_reasoning(candidate, rank, score):
    cid = candidate["candidate_id"]
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    
    # Extract details
    name = profile.get("anonymized_name", "Candidate")
    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "Software Engineer")
    company = profile.get("current_company", "Product Company")
    location = profile.get("location", "India")
    
    # 1. Deterministic seeding based on candidate ID
    seed = get_candidate_numeric_id(cid)
    rng = random.Random(seed)
    
    # Identify key skills present in candidate's skills list
    skill_names = [s.get("name", "").lower() for s in skills]
    career_descs = [job.get("description", "").lower() for job in career]
    all_text = " ".join(skill_names) + " " + " ".join(career_descs)
    
    vector_dbs = []
    for db in ["pinecone", "weaviate", "qdrant", "milvus", "faiss", "elasticsearch", "opensearch"]:
        if db in all_text:
            vector_dbs.append(db.capitalize() if db != "faiss" else "FAISS")
            
    eval_metrics = []
    for m in ["ndcg", "mrr", "map"]:
        if m in all_text:
            eval_metrics.append(m.upper())
            
    has_embeddings = any(kw in all_text for kw in ["embedding", "sentence-transformers", "bge", "e5"])
    
    # 2. Build introduction sentence
    intro_templates = [
        f"{name} is a seasoned professional with {yoe} years of experience, currently working as a {title} at {company}.",
        f"With {yoe} years of experience in ML, {name} currently holds the title of {title} at {company}.",
        f"An experienced engineering profile with {yoe} years in the industry, currently serving as a {title} at {company}.",
        f"{name} brings {yoe} years of applied expertise, currently operating as a {title} at {company}."
    ]
    intro = rng.choice(intro_templates)
    
    # 3. Build technical fit sentence
    tech_fits = []
    
    # Embeddings and Vector DBs
    if vector_dbs and has_embeddings:
        dbs_str = " and ".join(vector_dbs[:2])
        tech_fits.append(rng.choice([
            f"demonstrated experience implementing dense embeddings and vector databases such as {dbs_str}",
            f"proven track record of building semantic search systems using embeddings and {dbs_str}",
            f"hands-on deployment of dense retrieval models integrated with {dbs_str}"
        ]))
    elif vector_dbs:
        dbs_str = " and ".join(vector_dbs[:2])
        tech_fits.append(rng.choice([
            f"production experience scaling search indexing with {dbs_str}",
            f"extensive familiarity with vector DBs including {dbs_str}"
        ]))
    elif has_embeddings:
        tech_fits.append(rng.choice([
            "applied ML experience with sentence embeddings and representation learning",
            "hands-on exposure to embeddings-based retrieval systems"
        ]))
        
    # Evaluation metrics
    if eval_metrics:
        m_str = " and ".join(eval_metrics[:2])
        tech_fits.append(rng.choice([
            f"strong proficiency in ranking evaluation metrics (such as {m_str})",
            f"designed offline benchmarks using search evaluation metrics like {m_str}",
            f"optimization of ranking layers guided by metrics like {m_str}"
        ]))
        
    # Combine tech fits
    if tech_fits:
        tech_focus = " They show " + ", combined with ".join(tech_fits) + "."
    else:
        tech_focus = rng.choice([
            " They possess a solid foundation in Python development and NLP systems.",
            " Their background covers strong Python systems engineering and applied ML."
        ])
        
    # 4. Build signals and honest concerns
    notice = signals.get("notice_period_days", 90)
    willing_reloc = signals.get("willing_to_relocate", False)
    resp_rate = signals.get("recruiter_response_rate", 0.5)
    gh_score = signals.get("github_activity_score", -1)
    
    signals_sentence = ""
    concerns = []
    
    # Check notice period concern
    if notice > 30:
        concerns.append(f"notice period of {notice} days requires a buyout")
    elif notice == 0:
        signals_sentence += " Immediately available (0-day notice)."
        
    # Check location concern
    is_local = any(city in location.lower() for city in ["noida", "pune", "delhi", "gurgaon"])
    if not is_local:
        if willing_reloc:
            concerns.append(f"relocation from {location} will be necessary")
        else:
            concerns.append(f"currently located in {location} and not willing to relocate")
            
    # Platform activity
    if resp_rate >= 0.75:
        signals_sentence += f" Highly responsive on the platform ({resp_rate:.0%} response rate)."
    elif resp_rate < 0.20:
        concerns.append("very low recruiter responsiveness")
        
    if gh_score >= 60:
        signals_sentence += f" Strong open-source engagement with a GitHub score of {gh_score}/100."
        
    # Combine concerns
    concerns_str = ""
    if concerns:
        if rank <= 10:
            concerns_str = f" Despite a minor concern regarding {', '.join(concerns)}, their exceptional technical fit makes them our top choice."
        elif rank <= 50:
            concerns_str = f" Acknowledging a concern with {', '.join(concerns)}, they remain a highly competitive candidate."
        else:
            concerns_str = f" Note: Candidate has gaps including {', '.join(concerns)}, which places them lower on the shortlist."
            
    # Combine everything
    reasoning = f"{intro}{tech_focus}{signals_sentence}{concerns_str}"
    reasoning = reasoning.replace("  ", " ").strip()
    return reasoning

CONSULTING_COMPANIES = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture", 
    "cognizant", "capgemini", "wipro technologies", "tech mahindra", "hcl", "hcltech"
}

def read_docx_text_fallback(docx_path):
    """Reads text from a .docx file without using python-docx library."""
    try:
        import zipfile
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(docx_path) as docx:
            xml_content = docx.read('word/document.xml')
            root = ET.fromstring(xml_content)
            paragraphs = []
            for para in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                texts = [node.text for node in para.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text]
                if texts:
                    paragraphs.append("".join(texts))
            return "\n".join(paragraphs).strip()
    except Exception as e:
        print(f"Error reading docx via fallback: {e}")
        return ""

def get_job_description_text(script_dir):
    """Loads job description text from txt or docx file."""
    txt_path = os.path.join(script_dir, "job_description.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
            
    docx_path = os.path.join(script_dir, "job_description.docx")
    if os.path.exists(docx_path):
        return read_docx_text_fallback(docx_path)
        
    return ""

def is_honeypot(cand):
    """Identifies impossible/anomalous candidate profiles."""
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    edu = cand.get("education", [])
    
    # 1. Negative YoE
    yoe = profile.get("years_of_experience", 0)
    if yoe < 0:
        return True
        
    # 2. Expert skills with 0 months duration
    # "expert" proficiency in 5+ skills with 0 years used
    expert_zero_duration = 0
    for s in skills:
        dur = s.get("duration_months", 0)
        prof = s.get("proficiency", "").lower()
        if prof == "expert" and dur == 0:
            expert_zero_duration += 1
    if expert_zero_duration >= 5:
        return True
        
    # 3. Career duration mismatch (start/end date vs duration_months)
    for job in career:
        start = job.get("start_date")
        end = job.get("end_date")
        dur_months = job.get("duration_months", 0)
        if start:
            try:
                start_dt = datetime.strptime(start, "%Y-%m-%d")
                if end:
                    end_dt = datetime.strptime(end, "%Y-%m-%d")
                else:
                    end_dt = datetime(2026, 6, 19)
                
                calculated_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
                if abs(calculated_months - dur_months) > 3:
                    return True
            except:
                pass
                
    # 4. Job started before company was founded
    for job in career:
        desc = job.get("description", "").lower()
        start = job.get("start_date")
        if start and desc:
            try:
                start_year = int(start.split("-")[0])
                found_match = re.search(r'founded\s+(?:in\s+)?(\d{4})', desc)
                if found_match:
                    found_year = int(found_match.group(1))
                    if start_year < found_year:
                        return True
            except:
                pass

    # 5. Career started before education (allow 3 years overlap for internship)
    if edu and career:
        earliest_edu_start = min([e.get("start_year", 9999) for e in edu if e.get("start_year")])
        earliest_career_start = 9999
        for job in career:
            start = job.get("start_date")
            if start:
                try:
                    yr = int(start.split("-")[0])
                    if yr < earliest_career_start:
                        earliest_career_start = yr
                except:
                    pass
        if earliest_career_start < earliest_edu_start - 3:
            return True

    # 6. Multiple current jobs
    current_jobs = sum([1 for job in career if job.get("is_current")])
    if current_jobs > 1:
        return True

    return False

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        return None

def compute_heuristics_score(cand):
    """Calculates heuristic match score for candidate."""
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    edu = cand.get("education", [])
    signals = cand.get("redrob_signals", {})
    
    score = 0.0
    
    # 1. Experience Check (5-9 years ideal, 4-15 allowed)
    yoe = profile.get("years_of_experience", 0)
    if 5.0 <= yoe <= 9.0:
        score += 25.0
    elif 4.0 <= yoe < 5.0 or 9.0 < yoe <= 15.0:
        score += 15.0
    elif yoe > 15.0:
        score += 5.0
        
    # 2. Location & Relocation Check
    country = profile.get("country", "").strip().lower()
    location = profile.get("location", "").strip().lower()
    willing_reloc = signals.get("willing_to_relocate", False)
    
    is_in_india = (country == "india") or ("india" in location) or any(city in location for city in [
        "noida", "pune", "delhi", "gurgaon", "hyderabad", "bangalore", "mumbai", "chennai", "kolkata"
    ])
    
    is_near_office = any(city in location for city in ["pune", "noida", "delhi", "gurgaon"])
    
    if is_in_india:
        if is_near_office:
            score += 20.0
        elif willing_reloc:
            score += 15.0
        else:
            score += 5.0
    else:
        if willing_reloc:
            score += 5.0
        else:
            score -= 10.0
            
    # 3. Consulting/Services Company Check
    companies = [job.get("company", "").strip().lower() for job in career]
    all_consulting = len(companies) > 0 and all(any(c in comp for c in CONSULTING_COMPANIES) for comp in companies)
    has_consulting_now = len(companies) > 0 and any(c in companies[0] for c in CONSULTING_COMPANIES)
    
    if all_consulting:
        score -= 40.0
    elif has_consulting_now:
        score -= 10.0
        
    # 4. Pure Academic/Research check
    titles = [job.get("title", "").strip().lower() for job in career]
    all_research = len(titles) > 0 and all(
        "research" in t or "phd" in t or "student" in t or "academic" in t or "fellow" in t or "professor" in t or "intern" in t
        for t in titles
    )
    if all_research:
        score -= 40.0
        
    # 5. Core AI/ML Engineering & Retrieval System Skills
    skill_names = [s.get("name", "").strip().lower() for s in skills]
    career_descs = [job.get("description", "").strip().lower() for job in career]
    headline = profile.get("headline", "").strip().lower()
    summary = profile.get("summary", "").strip().lower()
    
    has_embeddings = any(kw in "".join(skill_names) or kw in "".join(career_descs) for kw in ["embedding", "sentence-transformers", "bge", "e5", "openai embeddings"])
    has_vector_db = any(kw in "".join(skill_names) or kw in "".join(career_descs) for kw in ["pinecone", "weaviate", "qdrant", "milvus", "faiss", "elasticsearch", "opensearch", "vector db", "vector database", "hybrid search"])
    has_python = "python" in "".join(skill_names) or "python" in "".join(career_descs) or "python" in headline or "python" in summary
    has_eval = any(kw in "".join(skill_names) or kw in "".join(career_descs) for kw in ["ndcg", "mrr", "map", "ranking evaluation", "offline benchmark", "learning to rank", "ab test"])
    
    if has_embeddings:
        score += 15.0
    if has_vector_db:
        score += 15.0
    if has_python:
        score += 5.0
    if has_eval:
        score += 15.0
        
    if not (has_embeddings or has_vector_db):
        score -= 30.0
        
    # Check if they have a non-AI current title (blocks keyword stuffers)
    current_title = profile.get("current_title", "").strip().lower()
    non_tech_titles = ["marketing", "sales", "accountant", "hr manager", "human resources", "ops manager", "operations manager", "customer support", "designer", "writer", "mechanical"]
    if any(nt in current_title for nt in non_tech_titles):
        score -= 50.0

    has_langchain = any("langchain" in s or "langchain" in desc for s in skill_names for desc in career_descs)
    if has_langchain and not (has_embeddings or has_vector_db or has_eval):
        score -= 10.0

    # 6. Behavioral signals & Platform activity
    notice = signals.get("notice_period_days", 90)
    if notice <= 30:
        score += 10.0
    elif notice <= 60:
        score += 5.0
    elif notice > 90:
        score -= 15.0
        
    last_active_str = signals.get("last_active_date")
    last_active = parse_date(last_active_str)
    if last_active:
        curr_date = datetime(2026, 6, 19)
        days_inactive = (curr_date - last_active).days
        if days_inactive > 180:
            score -= 20.0
        elif days_inactive < 30:
            score += 5.0
            
    resp_rate = signals.get("recruiter_response_rate", 0.5)
    if resp_rate < 0.15:
        score -= 20.0
    elif resp_rate >= 0.70:
        score += 10.0
        
    open_to_work = signals.get("open_to_work_flag", False)
    if open_to_work:
        score += 5.0
    else:
        score -= 5.0
        
    gh_score = signals.get("github_activity_score", -1)
    if gh_score >= 50:
        score += 5.0
    elif gh_score == -1:
        score -= 5.0

    edu_tiers = [e.get("tier", "unknown") for e in edu]
    if "tier_1" in edu_tiers:
        score += 10.0
    elif "tier_2" in edu_tiers:
        score += 5.0

    return score

def main():
    parser = argparse.ArgumentParser(description="Rank candidates for the Redrob AI Founding Team Senior AI Engineer role.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl file")
    parser.add_argument("--out", default="redrob_output.csv", help="Path to write the ranked output CSV (default: redrob_output.csv)")

    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_path = os.path.join(script_dir, "vector_embeddings", "candidate_embeddings.npz")
    jd_embedding_path = os.path.join(script_dir, "vector_embeddings", "jd_embedding.npy")
    model_weights_path = os.path.join(script_dir, "vector_embeddings", "model_weights")
    
    print("Loading candidate records...")
    candidates = []
    candidates_dict = {}
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cand = json.loads(line)
                candidates.append(cand)
                candidates_dict[cand["candidate_id"]] = cand
                
    print(f"Loaded {len(candidates)} candidates.")
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)

    # 1. Compute/Load JD embedding vector
    jd_vector = None
    if os.path.exists(jd_embedding_path):
        try:
            jd_vector = np.load(jd_embedding_path)
            print("Loaded pre-computed JD embedding vector.")
        except Exception as e:
            print(f"Warning: Failed to load pre-computed JD embedding: {e}")
            
    # 2. Load candidate embeddings
    candidate_embeddings_dict = {}
    if os.path.exists(embeddings_path):
        try:
            data = np.load(embeddings_path)
            embeddings = data["embeddings"]
            candidate_ids = data["candidate_ids"]
            for cid, emb in zip(candidate_ids, embeddings):
                candidate_embeddings_dict[cid] = emb
            print(f"Loaded {len(candidate_embeddings_dict)} candidate embeddings.")
        except Exception as e:
            print(f"Warning: Failed to load candidate embeddings: {e}")

    # 3. Verify embeddings exist (exit if missing to keep rank.py lightweight)
    if not candidate_embeddings_dict or jd_vector is None:
        print("Error: Pre-computed candidate embeddings or JD embedding vector are missing.")
        print("To generate them, run: python generate_embeddings.py")
        sys.exit(1)


            
    # 3. Perform Scoring and Filtering
    scored_candidates = []
    
    for cand in candidates:
        cid = cand["candidate_id"]
        
        # Exclude honeypots
        if is_honeypot(cand):
            continue
            
        # Calculate semantic cosine similarity score if vector is available
        sim_score = 0.0
        if jd_vector is not None and cid in candidate_embeddings_dict:
            cand_vector = candidate_embeddings_dict[cid]
            # Cosine similarity
            dot = np.dot(jd_vector, cand_vector)
            norm_jd = np.linalg.norm(jd_vector)
            norm_cand = np.linalg.norm(cand_vector)
            if norm_jd > 0 and norm_cand > 0:
                sim_score = dot / (norm_jd * norm_cand)
                
        # Calculate heuristic score
        h_score = compute_heuristics_score(cand)
        
        # Combine scores (semantic similarity scaled by 100 + heuristics)
        # We handle case where candidate failed pre-filter (sim_score is 0.0, which naturally downweights them)
        composite_score = (sim_score * 100.0) + h_score
        
        # Scale score to 0.0 - 1.0 range for readability/consistency (e.g. dividing by 150)
        scaled_score = max(0.0001, min(0.9999, composite_score / 150.0))
        # Round to 4 decimal places before sorting to guarantee the tie-breakers match CSV output
        rounded_score = round(scaled_score, 4)
        
        scored_candidates.append((rounded_score, cid, cand))
        
    # 4. Sort and select top 100
    # Sorting is descending by score, ascending by candidate_id for strict tie-breaking
    scored_candidates.sort(key=lambda x: (-x[0], x[1]))
    
    top_100 = scored_candidates[:100]
    
    # 5. Write to output CSV
    print(f"Writing top 100 ranked candidates to {args.out}...")
    
    # Ensure parent directory of output exists
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for index, (score_val, cid, cand) in enumerate(top_100):
            rank = index + 1
            scaled_score_str = f"{score_val:.4f}"
            
            # Generate fact-based reasoning description
            reasoning = generate_reasoning(cand, rank, score_val)
            
            writer.writerow([cid, rank, scaled_score_str, reasoning])
            
    print("Ranking successfully completed! Validation script can now be run.")

if __name__ == "__main__":
    main()
