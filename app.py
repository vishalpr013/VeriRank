import gradio as gr
import subprocess
import os
import pandas as pd

def run_ranking(file_obj):
    if file_obj is None:
        return "Please upload a candidates.jsonl file.", None
    
    input_path = file_obj.name
    output_path = "redrob_output.csv"
    
    # Remove existing output if any
    if os.path.exists(output_path):
        os.remove(output_path)
        
    try:
        # Run rank.py via subprocess to parse the candidates dataset
        result = subprocess.run(
            ["python", "rank.py", "--candidates", input_path, "--out", output_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Check if output file was created
        if not os.path.exists(output_path):
            return f"Error: Output file was not generated.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}", None
            
        # Load output CSV to display preview
        df = pd.read_csv(output_path)
        preview_df = df.head(10) # Show top 10 as preview
        
        status_msg = f"Success! VeriRank processed the dataset.\n\n{result.stdout}"
        return status_msg, preview_df, output_path
        
    except subprocess.CalledProcessError as e:
        return f"Pipeline failed with exit code {e.returncode}.\nError details:\n{e.stderr}", None, None
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}", None, None

# Define Gradio Interface
with gr.Blocks(title="VeriRank Candidate Discovery Engine", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔍 VeriRank: Candidate Discovery & Ranking Engine
        ### Redrob AI Data & AI Challenge
        
        Upload your candidate profile dataset (`candidates.jsonl`) to retrieve the top 100 ranked candidates with deterministic, hallucination-free reasoning.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload candidates.jsonl", file_types=[".jsonl"])
            run_btn = gr.Button("🚀 Run VeriRank Engine", variant="primary")
            
        with gr.Column(scale=2):
            status_output = gr.Textbox(label="Execution Status", interactive=False, placeholder="Run status will appear here...")
            output_file = gr.File(label="Download Ranked Shortlist (CSV)", interactive=False)
            
    gr.Markdown("### 📊 Preview: Top 10 Ranked Candidates")
    table_preview = gr.Dataframe(headers=["candidate_id", "rank", "score", "reasoning"], interactive=False)
    
    run_btn.click(
        fn=run_ranking,
        inputs=[file_input],
        outputs=[status_output, table_preview, output_file]
    )

if __name__ == "__main__":
    demo.launch()
