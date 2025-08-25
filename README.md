# Book Recommender System
A semantic Book Recommender System powered by Large Language Models (LLMs), LangChain, and Gradio.
It classifies books into categories, performs sentiment analysis on descriptions, and provides personalized recommendations through an interactive dashboard.

🚀 Features

Text Classification → Zero-shot classification of book descriptions into categories (Fiction, Non-Fiction, Children’s, etc.).

Fill Missing Categories → Predicts categories for books with missing labels using LLMs.

Sentiment Analysis → Uses a fine-tuned model to classify emotions (Joy, Fear, Sadness, Surprise, etc.) from book descriptions.

Semantic Search → Retrieves books using embeddings & similarity search.

Re-ranking with Cross-Encoder → Improves recommendation accuracy by re-scoring top candidates.

Interactive Gradio Dashboard → User-friendly web app to:

Enter a query (e.g., “A story about forgiveness”)

Filter by category

Filter by emotional tone

Get recommended books with covers, authors, and truncated descriptions.
