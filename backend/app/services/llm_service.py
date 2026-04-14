from openai import OpenAI
import os
from dotenv import load_dotenv

#Load environment variables from .env
load_dotenv()

#Fetch API key
api_key = os.getenv("OPENAI_API_KEY")

# (Optional safety check)
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

#Initialize client
client = OpenAI(api_key=api_key)


def generate_answer(query, docs):
    """Generate an answer based on retrieved documents with proper source tracking."""
    context_parts = []
    source_references = {}  # Track unique sources
    source_to_label = {}   # Map unique sources to labels
    label_counter = 1
    
    # Process documents and create context with labels
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        url = doc.metadata.get("url", "Unknown")
        content = doc.page_content
        
        # Create unique source key
        source_key = (title, url)
        
        # Assign label if not already assigned
        if source_key not in source_to_label:
            label = f"[Source {label_counter}]"
            source_to_label[source_key] = label
            source_references[label] = {"title": title, "url": url}
            label_counter += 1
        
        label = source_to_label[source_key]
        context_parts.append(f"{label}\n{content}")
    
    context = "\n\n".join(context_parts)
    
    # Build reference guide for the prompt
    references_guide = "\n".join(
        [f"{label}: {info['title']} ({info['url']})" 
         for label, info in source_references.items()]
    )

    prompt = f"""
    Answer the query using ONLY the provided context. Structure your response in the following format:

    1. Introduction: Provide a brief overview of the topic based on the context.
    2. Key Findings: Summarize the main points and insights from the sources.
    3. Conclusion: Offer a concise conclusion based on the findings.

    Keep the response concise. Cite sources in the relevant sections where applicable using their assigned labels (e.g., [Source 1], [Source 2]).

    If the context does not contain enough information to answer the query, say:
    "I don't have enough information from the retrieved papers."

    Query:
    {query}

    Source Reference Guide:
    {references_guide}

    Context:
    {context}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    
    # Append proper References section with exact sources retrieved
    references_section = "\n\n4. References:\n"
    for label, info in source_references.items():
        references_section += f"   {label} {info['title']} ({info['url']})\n"
    
    return answer + references_section

