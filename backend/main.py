from app.core.pipeline import run_pipeline
import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

if __name__ == "__main__":
    query = input("Enter your research query: ")
    result = run_pipeline(query)

    print("\n=== FINAL OUTPUT ===\n")
    print(result)