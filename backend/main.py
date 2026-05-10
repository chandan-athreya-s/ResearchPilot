from app.agents.orchestrator import orchestrate
import os
from dotenv import load_dotenv
from huggingface_hub import login
import argparse
import sys

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

def main():
    parser = argparse.ArgumentParser(description="ResearchPilot - AI Research Assistant")
    parser.add_argument("--agent-mode", action="store_true", 
                       help="Use agent orchestrator mode (default: pipeline mode)")
    parser.add_argument("--session-id", type=str, 
                       help="Session ID for persistence (auto-generated if not provided)")
    parser.add_argument("--follow-up", action="store_true", 
                       help="Treat query as follow-up in existing session")
    parser.add_argument("--list-sessions", action="store_true", 
                       help="List available sessions")
    
    args = parser.parse_args()
    
    # Handle list-sessions command
    if args.list_sessions:
        list_available_sessions()
        return
    
    # Get query from args or prompt
    if len(sys.argv) == 1 or (len([arg for arg in sys.argv if not arg.startswith('--')]) == 1):
        # No positional args, prompt for query
        query = input("Enter your research query: ")
    else:
        # Query provided as positional arg
        query = ' '.join([arg for arg in sys.argv[1:] if not arg.startswith('--')])
    
    session_id = args.session_id
    follow_up = args.follow_up
    agent_mode = args.agent_mode
    
    # Determine mode: agent-mode flag or default to pipeline
    if agent_mode:
        print("Running in AGENT MODE (orchestrator)")
        result = orchestrate(query, session_id=session_id, follow_up=follow_up, verbose=True)
    else:
        print("Running in PIPELINE MODE")
        # Import pipeline here to avoid circular imports
        from app.core.pipeline import run_pipeline
        result = run_pipeline(query, session_id=session_id, follow_up=follow_up)

    print("\n=== FINAL OUTPUT ===\n")
    print(f"Session ID: {result['session_id']}")
    print(f"Mode: {result.get('mode', 'unknown')}")
    print(f"Iterations: {result.get('iteration_count', 0)}")
    print(f"Papers Used: {result['papers_used']}")
    print(f"\nReport:\n{result['answer']}")
    
    if result['sources']:
        print(f"\nSources ({len(result['sources'])}):")
        for i, source in enumerate(result['sources'], 1):
            title = source.get('title') or source.get('display_name') or 'Unknown title'
            year  = source.get('year')  or source.get('publication_year') or 'n.d.'
            url   = source.get('url')   or source.get('id') or ''
            print(f"{i}. {title} ({year}) - {url}")

def list_available_sessions():
    """List all available sessions in the sessions directory."""
    sessions_dir = "sessions"
    if not os.path.exists(sessions_dir):
        print("No sessions directory found.")
        return
    
    sessions = []
    for item in os.listdir(sessions_dir):
        session_path = os.path.join(sessions_dir, item)
        if os.path.isdir(session_path):
            # Check if it has index directory (indicating a valid session)
            index_path = os.path.join(session_path, "index")
            if os.path.exists(index_path):
                sessions.append(item)
    
    if not sessions:
        print("No saved sessions found.")
        return
    
    print("Available sessions:")
    for session in sorted(sessions):
        print(f"  - {session}")

if __name__ == "__main__":
    main()