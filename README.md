# 📚 Book Recommender System  

A **semantic Book Recommender System** powered by **Large Language Models (LLMs)**, **LangChain**, and **Gradio**.  
It classifies books into categories, performs **sentiment analysis** on descriptions, and provides **personalized recommendations** through an interactive dashboard.  

---

## 🚀 Features  

- **Text Classification** → Zero-shot classification of book descriptions into categories (Fiction, Non-Fiction, Children’s, etc.).  
- **Fill Missing Categories** → Predicts categories for books with missing labels using LLMs.  
- **Sentiment Analysis** → Uses a fine-tuned model to classify emotions (*Joy, Fear, Sadness, Surprise, etc.*) from book descriptions.  
- **Semantic Search** → Retrieves books using embeddings & similarity search.  
- **Re-ranking with Cross-Encoder** → Improves recommendation accuracy by re-scoring top candidates.  
- **Interactive Gradio Dashboard** → User-friendly web app to:  
  - Enter a query (e.g., *“A story about forgiveness”*)  
  - Filter by category  
  - Filter by emotional tone  
  - Get recommended books with covers, authors, and truncated descriptions  

---

## 📂 Project Structure  
<img width="649" height="309" alt="image" src="https://github.com/user-attachments/assets/110c9ee1-df3b-431b-b696-8d9f094bd392" />


## ⚡ Installation

1] Clone this repo <br>
2] cd book-recommender-system  <br>
3] Create a virtual environment & install dependencies <br>
python -m venv venv <br>
source venv/bin/activate   # On Mac/Linux <br>
venv\Scripts\activate      # On Windows <br>
4] pip install -r requirements.txt <br>


▶️ Usage
Run the Gradio Dashboard <br>
python notebooks/gradio_dashboard.py <br>

Then open in browser: 
👉 http://127.0.0.1:7860

📊 Example Output
<img width="1440" height="866" alt="image" src="https://github.com/user-attachments/assets/96acd8e3-1ab2-4bd0-96d0-c59443097848" />


