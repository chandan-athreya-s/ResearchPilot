from app.core.pipeline import run_pipeline

if __name__ == "__main__":
    query = input("Enter your research query: ")
    result = run_pipeline(query)

    print("\n=== FINAL OUTPUT ===\n")
    print(result)