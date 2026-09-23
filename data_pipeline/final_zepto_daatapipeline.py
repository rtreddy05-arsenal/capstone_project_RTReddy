import sqlite3
import requests
import pandas as pd
from bs4 import BeautifulSoup

# TASK 1: SCRAPING (>= 60 books across >= 3 categories)
print("1. Scraping data from books.toscrape.com...")

category_urls = {
    "Non-Fiction": "https://books.toscrape.com/catalogue/category/books/non-fiction_13/index.html",
    "Science-Fiction": "https://books.toscrape.com/catalogue/category/books/science-fiction_16/index.html",
    "Young-Adult": "https://books.toscrape.com/catalogue/category/books/young-adult_21/index.html",
    "Poetry": "https://books.toscrape.com/catalogue/category/books/poetry_23/index.html",
}

category_ids = {"Non-Fiction": 1, "Science-Fiction": 2, "Young-Adult": 3, "Poetry": 4}
scraped_books = []

for category_name, url in category_urls.items():
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    books = soup.find_all("article", class_="product_pod")

    for book in books:
        title = book.find("h3").find("a")["title"]
        price_text = book.find("p", class_="price_color").text.strip()
        rating_class = book.find("p", class_="star-rating")["class"][1]
        availability_text = book.find("p", class_="instock availability").text.strip()

        scraped_books.append({
            "title": title,
            "price_raw": price_text,
            "star_rating_raw": rating_class,
            "availability_raw": availability_text,
            "category_id": category_ids[category_name]
        })

books_df = pd.DataFrame(scraped_books)
print(f"Scraped {len(books_df)} books across {len(category_urls)} categories.")

# TASK 2: CLEANING, IMPUTATION & CURRENCY CONVERSION
print("\n2. Cleaning and converting data...")

# 1. Strip currency symbol and cast to float
books_df["price_gbp"] = pd.to_numeric(
    books_df["price_raw"].str.replace(r"[^\d.]", "", regex=True),
    errors="coerce"
)

# 2. Text star rating to integer (1-5)
rating_dict = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
books_df["rating"] = books_df["star_rating_raw"].str.strip().map(rating_dict)

# 3. Availability to boolean
books_df["in_stock"] = books_df["availability_raw"].str.lower().str.contains("in stock", na=False)

# 4. Handle missing/invalid data via median imputation
books_df["price_gbp"] = books_df["price_gbp"].fillna(books_df["price_gbp"].median())
books_df["rating"] = books_df["rating"].fillna(books_df["rating"].median()).astype(int)

# 5. Fixed-rate currency conversion: 1 GBP = 105.50 INR
GBP_TO_INR = 105.50
books_df["price_inr"] = (books_df["price_gbp"] * GBP_TO_INR).round(2)

# Drop raw parsing columns
books_df = books_df.drop(columns=["price_raw", "star_rating_raw", "availability_raw"])
print("Cleaned DataFrame sample:")
print(books_df.head(3))

# TASK 3 & 4: NORMALIZED SQLITE SCHEMA & DATA INSERTION
print("\n3. Setting up SQLite database and normalized schema...")

conn = sqlite3.connect("catalog.db")
cursor = conn.cursor()
cursor.execute("PRAGMA foreign_keys = ON;")

cursor.execute("DROP TABLE IF EXISTS books;")
cursor.execute("DROP TABLE IF EXISTS categories;")

cursor.execute("""
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE
);
""")

cursor.execute("""
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
""")

# Insert categories
categories_df = pd.DataFrame(list(category_ids.items()), columns=["category_name", "category_id"])
categories_df[["category_id", "category_name"]].to_sql("categories", conn, if_exists="append", index=False)

# Insert books
books_df.to_sql("books", conn, if_exists="append", index=False)
conn.commit()

# TASK 5: 5 SQL QUERIES (COVERING ALL REQUIRED CLAUSES)
print("\n4. Executing required SQL queries...")

queries = [
    ("Query 1 (SELECT/WHERE)", "SELECT title, price_inr FROM books WHERE in_stock = 1 LIMIT 5;"),
    ("Query 2 (ORDER BY/LIMIT)", "SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 5;"),
    ("Query 3 (DISTINCT)", "SELECT DISTINCT rating FROM books ORDER BY rating ASC;"),
    ("Query 4 (BETWEEN)", "SELECT title, price_inr FROM books WHERE price_inr BETWEEN 2000.00 AND 4000.00 LIMIT 5;"),
    ("Query 5 (JOIN)", """SELECT b.title, c.category_name, b.rating 
FROM books b 
JOIN categories c ON b.category_id = c.category_id 
ORDER BY b.rating DESC, b.title ASC 
LIMIT 10;""")
]

with open("query_results.txt", "w", encoding="utf-8") as f:
    for name, q in queries:
        res = pd.read_sql_query(q, conn)
        block = f"[{name}]\n{q}\n\nOUTPUT:\n{res.to_string(index=False)}\n\n"
        print(block)
        f.write(block)

# TASK 6: SQL VS PANDAS EQUIVALENCE VALIDATION
print("5. Validating SQL JOIN vs. Pandas Merge equivalence...")

# Approach A: via SQL JOIN
df_sql_join = pd.read_sql_query(queries[4][1], conn)

# Approach B: via in-memory pd.merge
df_pandas_merge = pd.merge(books_df, categories_df, on="category_id", how="inner")
df_pandas_merge = df_pandas_merge[["title", "category_name", "rating"]].sort_values(
    by=["rating", "title"], ascending=[False, True]
).head(10).reset_index(drop=True)

conn.close()

# Verify alignment
diff = df_sql_join.compare(df_pandas_merge)
if diff.empty:
    print("SUCCESS: pd.read_sql and pd.merge produce 100% identical outputs!")
else:
    print("Differences found:\n", diff)