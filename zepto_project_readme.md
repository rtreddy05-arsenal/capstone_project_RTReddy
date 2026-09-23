# Zepto Data Pipeline , Analytics & Support Assitance

Welcome to the Zepto Full-Stack Data & AI Project. This repository contains three comprehensive modules that demonstrate end-to-end capabilities across Data Engineering, Machine Learning, and Generative AI (RAG).

#  Project Structure

The project is divided into three core modules:

### Module 1: Data Engineering Pipeline (`final_zepto_daatapipeline.py`)

A robust web scraping and ETL pipeline that extracts product data, cleans it, and loads it into a normalized relational database.

 Scraping: Extracts book titles, prices, ratings, and stock availability across multiple categories from `books.toscrape.com` using `requests` and `BeautifulSoup`.

 Data Cleaning & Transformation: Cleans raw text, converts string ratings to integers, maps availability to booleans, handles missing values using median imputation, and applies fixed-rate currency conversion (GBP to INR) using Pandas.

 Database Management: Creates a normalized SQLite schema (`catalog.db`) with `categories` and `books` tables, ensuring referential integrity with Foreign Keys.

 SQL Queries & Validation: Executes complex SQL queries (filtering, ordering, grouping, joins) and strictly validates SQL `JOIN` outputs against Pandas `merge` equivalence.

# Module 2: Machine Learning & EDA Pipeline (Jupyter Notebook / Python)

An extensive exploratory data analysis and predictive modeling workflow built around the classic Titanic dataset.

 Exploratory Data Analysis (EDA): Missing data profiling, threshold-based imputation/dropping rules, and outlier detection using IQR. 

 Data Storytelling: Visualizations using `seaborn` and `matplotlib` (Histograms, Box plots, Scatter plots, and Correlation Heatmaps) to identify skewness and feature relationships.

Feature Engineering & Pipeline: Prevents data leakage by utilizing `make_column_transformer` and `make_pipeline` with `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`.

Classification Modeling: Predicts passenger survival using Logistic Regression, Decision Trees, and Random Forests. Includes evaluation metrics like Accuracy, Precision, Recall, F1 Score, Confusion Matrices, and ROC-AUC curves.

Handling Imbalanced Data: Compares Baseline performance against Class Weights and SMOTE (Synthetic Minority Over-sampling Technique).

Hyperparameter Tuning: Utilizes `GridSearchCV` leveraging the Out-Of-Bag (OOB) score for Random Forests.

Regression Side-Task: Predicts ticket `fare` using `LinearRegression`, evaluated with MAE, RMSE, R-Squared, Adjusted R-Squared, and Residual Plots.

Deployment Ready: Exports the finalized preprocessing and model pipeline using `joblib` for immediate deployment on raw data.

### Module 3: Support Assistant RAG API (`app.py`)

A Retrieval-Augmented Generation (RAG) backend API built with FastAPI, orchestrated by LangGraph, and utilizing ChromaDB for vector storage. 

#### RAG Pipeline Architecture & Flowchart

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



# Ingestion & Embedding : Automatically embeds Zepto policy documents using `sentence-transformers` (`all-MiniLM-L6-v2`) and persists them locally via `ChromaDB`.

# Agentic Routing: Uses LangGraph to classify user intent, routing strict policy questions to the retrieval node and general questions to a fallback node.

# Structured Output: Enforces strict JSON schemas using `Pydantic` models (`FinalResponse`), ensuring responses always contain an `answer`, `sources` array, and `confidence` score.

# Mock LLM Mode: Entirely operable offline without API keys via `MOCK_LLM=1`. Uses deterministic rule-based logic for grading and testing, easily swappable for real LLM calls.

---


### Prerequisites
Make sure you have Python 3.9+ installed. Install the required dependencies:

pip install -r requirements.txt

*(Dependencies include: `fastapi`, `uvicorn`, `pydantic`, `langgraph`, `chromadb`, `sentence-transformers`, `pandas`, `scikit-learn`, `beautifulsoup4`, `matplotlib`, `seaborn`, `imblearn`)*

### Running Module 1: Data Pipeline

python final_zepto_daatapipeline.py

# This will generate `catalog.db` and output `query_results.txt`

### Running Module 3: RAG API
Start the FastAPI server:

uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Send a test request (PowerShell):
powershell

 Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"query": "How much does priority delivery cost?"}' 

Send a test request (cURL):

curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "How much does priority delivery cost?"}'


### Docker Containerization (Module 3)

Build and run the application via Docker:

docker build -t zepto-support-assistant .

docker run -p 7860:8000 zepto-support-assistant

(Note: Exposes the app on port 7860)*