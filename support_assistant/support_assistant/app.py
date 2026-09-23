# Module 3 — Support Assistant

# Complete GenAI service for Zepto: A document corpus you embed and index, a LangGraph-orchestrated flow that routes each query and retrieves grounded context,
#  
# A structured-output guarantee, and a FastAPI wrapper you run locally.
# Entire pipeline is graded through a deterministic, fully offline mock mode for the LLM calls — no signup, no API key, 
# and no network access to any LLM provider are required to earn full marks on this module.
# A real LLM call and a live cloud deployment are both optional, ungraded extensions layered on top of that graded baseline.

#------------------------------------------------------------------------------------------

# LLM calls — offline mock is the graded baseline (read before starting): 
# every LLM call in this module is gated behind a single environment variable, MOCK_LLM. 
# Left unset, or set to MOCK_LLM=1, the service runs the fully deterministic, 
# rule-based mock logic described in Task 3 below — no signup, no API key, and no network call to any LLM provider. 
# This is the default state and is what gets graded; your submission must be fully correct using only this path. Only when MOCK_LLM=0 is explicitly set does the service call a real LLM.


# Embeddings — no API needed: generate embeddings locally using the open-source sentence-transformers library with the all-MiniLM-L6-v2 model,
#  and store in ChromaDB — both run entirely on your machine at no cost and require no account.

#-------------------------------------------------------------------------------------------


# CREATE DOC FILES 

import os
import uvicorn
from pathlib import Path
from typing import List, TypedDict
from fastapi import FastAPI
from pydantic import BaseModel, ValidationError
import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END

# ==========================================
# TASK 1: DOCUMENT CORPUS & DATABASE SETUP
# ==========================================
DOCUMENTS = {
    "doc_01": "Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee. Priority delivery, which reserves the next available rider slot, is available at checkout for an additional INR 15. Zepto does not currently deliver to addresses outside its listed serviceable pin codes.",
    "doc_02": "Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unopened, resalable condition. Approved refunds are credited to the original payment method within 3–5 business days, or instantly to the Zepto wallet if the customer opts for wallet credit. Personal care items that have been opened are non-returnable except in the case of a manufacturing defect. Return pickup, where required, is arranged free of cost by Zepto.",
    "doc_03": "Zepto offers three account tiers: Basic (free, default tier, standard delivery fees apply), Zepto Pass (INR 49 per month, free standard delivery on all orders and 5% off select categories), and Zepto Pass+ (INR 99 per month, free priority delivery, 10% off select categories, and early access to limited-time deals 24 hours before they go live to Basic and Pass members). Membership can be cancelled at any time from account settings; cancelling stops the next billing cycle but does not refund the current membership period.",
    "doc_04": "Every Zepto order shows a live rider-tracking map from the moment it is packed until delivery, accessible from the 'Track Order' screen. Estimated delivery time updates automatically as the rider moves. If an order's status shows no movement for more than 20 minutes past its original estimated delivery time, customers should contact support directly rather than continue waiting, since this indicates a likely delivery issue.",
    "doc_05": "Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been packed, it can no longer be cancelled through the app, since the rider is dispatched immediately after packing given Zepto's quick-delivery model. If a packed order cannot be delivered due to a Zepto-side issue (for example, rider unavailability), the order is auto-cancelled and fully refunded without any cancellation fee.",
    "doc_06": "If an order arrives with damaged, spoiled, or missing items, customers must report it within 24 hours of delivery through the 'Report an Issue' button on the order page. Zepto ships a free replacement or issues a full refund for damaged, spoiled, or missing items without requiring the customer to return the original item, unless the order value exceeds INR 1000, in which case a photo of the issue must be submitted through the report form before a replacement or refund is processed.",
    "doc_07": "Zepto gift cards are available in fixed denominations of INR 100, INR 250, INR 500, and INR 1000, and are delivered by email or SMS within minutes of purchase. Gift cards are valid for 1 year from the date of issue and carry no maintenance fees. Gift card balance can be combined with one other payment method at checkout but cannot be combined with another gift card in the same transaction. Gift card balance cannot be redeemed for cash except where required by law.",
    "doc_08": "Zepto customer support is available via in-app chat 24 hours a day, 7 days a week, given the time-sensitive nature of quick commerce deliveries. Average in-app chat response time is under 2 minutes. Email support is also available for non-urgent queries and is answered within 24 hours on business days. Phone support is not offered."
}

docs_dir = Path("docs")
docs_dir.mkdir(parents=True, exist_ok=True)

ids, documents, metadatas = [], [], []
for doc_id, text in DOCUMENTS.items():
    file_path = docs_dir / f"{doc_id}.txt"
    file_path.write_text(text, encoding="utf-8")
    ids.append(doc_id)
    documents.append(text)
    metadatas.append({"source": f"{doc_id}.txt"})

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents).tolist()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="zepto_support_docs")
collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

# ==========================================
# TASK 2: STRUCTURED PROMPT TEMPLATE
# ==========================================
PROMPT_TEMPLATE = """
**ROLE**
You are a precise and highly accurate Customer Support Assistant for Zepto.

**CONTEXT**
Retrieved Documents: {context}

**TASK**
Answer the user's support query accurately by extracting details strictly from the provided context.

**NEGATIVE CONSTRAINTS**
- Do not answer using external knowledge or assumptions not present in the provided context.
- If the answer is not present in the context, return exactly: "I cannot find the answer to this in the provided support documents."

**FORMAT**
Return your response strictly as a JSON object matching this schema:
{{
  "answer": "string",
  "sources": ["list of document IDs"],
  "confidence": float 0-1
}}

**LENGTH**
Keep the answer concise, under 50 words.

**EXAMPLE**
[Input]
User Query: What is the priority fee?
Context: [doc_01]: Priority delivery is available for INR 15.
[Output]
{{
  "answer": "Priority delivery is available for an additional INR 15.",
  "sources": ["doc_01"],
  "confidence": 1.0
}}

**CURRENT INPUT**
User Query: {query}
"""

# ==========================================
# TASK 4: JSON OUTPUT SCHEMA (PYDANTIC)
# ==========================================
class QueryRequest(BaseModel):
    query: str

class FinalResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

class GraphState(TypedDict):
    query: str
    intent: str
    response: dict

# ==========================================
# TASK 3 & 5: LANGGRAPH NODES & LOGIC
# ==========================================
def is_mock_mode() -> bool:
    return os.getenv("MOCK_LLM", "1") == "1"

def call_llm_with_retry(prompt: str) -> dict:
    """Mock helper for the real LLM extension path with retry logic."""
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            raw_output = '{"answer": "Real LLM Answer", "sources": ["doc_01"], "confidence": 0.9}' 
            validated_data = FinalResponse.model_validate_json(raw_output)
            return validated_data.model_dump()
        except ValidationError as e:
            prompt += f"\n\nValidation Error: {e}. Return ONLY valid JSON."
    
    return FinalResponse(answer="Error generating response.", sources=[], confidence=0.0).model_dump()

def classify_intent(state: GraphState):
    query_lower = state["query"].lower()
    if is_mock_mode():
        keywords = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]
        intent = "policy_question" if any(kw in query_lower for kw in keywords) else "general_question"
    else:
        intent = "policy_question"
    return {"intent": intent}

def retrieve_and_answer(state: GraphState):
    query = state["query"]
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    
    docs_list = results.get("documents") or []
    retrieved_docs = docs_list[0] if docs_list else []
    ids_list = results.get("ids") or []
    retrieved_ids = ids_list[0] if ids_list else []
    
    if is_mock_mode():
        if retrieved_docs:
            top_snippet = retrieved_docs[0][:200]
            answer_text = f"Based on the retrieved context: {top_snippet}..."
            sources = retrieved_ids
        else:
            answer_text = "No relevant documents found."
            sources = []
            
        final_res = FinalResponse(answer=answer_text, sources=sources, confidence=1.0)
        response_dict = final_res.model_dump()
    else:
        prompt = PROMPT_TEMPLATE.format(context=retrieved_docs, query=query)
        response_dict = call_llm_with_retry(prompt)
        
    return {"response": response_dict}

def direct_answer(state: GraphState):
    if is_mock_mode():
        final_res = FinalResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0
        )
        response_dict = final_res.model_dump()
    else:
        prompt = f"Answer this general query: {state['query']}"
        response_dict = call_llm_with_retry(prompt)
        
    return {"response": response_dict}

# ==========================================
# BUILD THE GRAPH & FASTAPI APP
# ==========================================
workflow = StateGraph(GraphState)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.add_edge(START, "classify_intent")
workflow.add_conditional_edges("classify_intent", lambda state: state["intent"], {
    "policy_question": "retrieve_and_answer",
    "general_question": "direct_answer"
})
workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

graph = workflow.compile()

app = FastAPI(title="Zepto Support Assistant")

@app.post("/ask", response_model=FinalResponse)
def ask(payload: QueryRequest):
    result = graph.invoke({"query": payload.query, "intent": "", "response": {}})
    return result["response"]

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


#--------------------------------------------------------------------------------------------


#TASK 6:- FAST API DOCKERFILE - 


# IN DOCKERFILE 

# JSON queries are posted in README.md