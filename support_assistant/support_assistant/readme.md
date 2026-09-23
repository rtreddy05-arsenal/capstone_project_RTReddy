
# TASK 7 . README FILE ON RAG PIPELINE AND FLOWCHART 




# Module 3 — Support Assistant (/support_assistant)

## RAG Pipeline Architecture & Flowchart


[1. Ingestion]   8 Policy Files (doc_01 to doc_08)
                       ↓ 
[2. Embedding]   all-MiniLM-L6-v2 → ChromaDB ('zepto_support_docs')
                       ↓ 
[3. Request]     User Query → FastAPI (POST /ask)
                       ↓ 
[4. Routing]     classify_intent
                 ├── Policy Question  ➔ retrieve_and_answer (Queries ChromaDB)
                 └── General Question ➔ direct_answer (No search)
                       ↓ 
[5. Response]    Validated JSON: { answer, sources, confidence }


# The 4 Pipeline Steps

Ingestion (Reading Files):

    Handled by app.py.

    Reads 8 text files (doc_01.txt to doc_08.txt). Each file is kept as 1 chunk.

# Embedding (Storing Searchable Data):

    Handled by app.py using the all-MiniLM-L6-v2 model.

    Converts the text into numbers (vectors) and saves them locally into ChromaDB inside the collection named zepto_support_docs.

# Retrieval (Finding the Right Info):

    When a user sends a query to POST /ask in app.py, LangGraph runs:

        classify_intent node: Checks if the question is about policies or general chat.

        retrieve_and_answer node: If it's a policy question, it searches ChromaDB and grabs the top 3 matching chunks. (This search always runs for real, even in mock mode).

# Generation (Writing the Answer):

    Handled by either retrieve_and_answer (for policies) or direct_answer (for general questions).

    Formats the final answer into a strict JSON object with 3 fields: answer, sources, and confidence.

# The MOCK_LLM Switch

The system behavior changes depending on the MOCK_LLM environment setting:

    Default Mock Mode (MOCK_LLM=1 or unset — Graded Baseline):

     Classification: Uses a simple list of keywords (like "delivery", "refund", "cancel") to detect policy questions.

     Policy Answer: Copies the first ~200 characters from the top retrieved document snippet into a template.

    General Answer: Returns the fixed message: "I can only answer questions about Zepto policies right now."

# LLM Mode (MOCK_LLM=0 — Optional Extension):

    Calls an actual LLM (like Groq) to classify the intent and write natural, grounded answers using the retrieved context.


# After result application status complete we need to need to JSON reponse 

curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d "{\"query\": \"How much does priority delivery cost?\"}"


    # To above query , open new terminal in powershell 

    Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"query": "How much does priority delivery cost?"}' 
   
    # When you post this , the terminal executes JSON query and gives its answer 
   
## Example API Calls

### 1. Policy Question (Triggers Retrieval)
**Request:**
`POST /ask {"query": "How much does priority delivery cost?"}`

**Response:**

# JSON response 

{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes...",
  "sources": ["doc_01"],
  "confidence": 1.0
}


# General Question (No Retrieval)

curl -X POST "[http://127.0.0.1:8000/ask](http://127.0.0.1:8000/ask)" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of France?"}'


# JSON RESPONSE
     
     {
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}




# Local Docker Containerization

docker build -t zepto-support-assistant .

docker run -p 7860:7860 zepto-support-assistant

curl -X POST "[http://127.0.0.1:7860/ask](http://127.0.0.1:7860/ask)" \
     -H "Content-Type: application/json" \
     -d '{"query": "How much does priority delivery cost?"}'