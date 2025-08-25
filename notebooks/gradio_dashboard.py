import pandas as pd 
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import pandas as pd
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
import gradio as gr
import numpy as np

load_dotenv()
# dataset that already has the predicted categories (fiction/non-fiction/etc.) and the emotion scores from earlier sentiment analysis steps.
books = pd.read_csv("processed_data/books_with_emotions.csv")

# The original dataset has a thumbnail column with URLs pointing to Google Books covers.
# By appending &fife=w800, you’re asking Google Books to return larger, higher-resolution cover images.
# This creates a new column large_thumbnail with the updated URLs.
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(books["large_thumbnail"].isna(), "cover-not-found.jpg",books["large_thumbnail"],)

# Loads the book descriptions from a .txt file using LangChain’s TextLoader.
raw_documents = TextLoader("processed_data/tagged_description.txt").load()
# Since descriptions can vary in length, we split the raw documents into chunks for embedding.
text_splitter = CharacterTextSplitter(chunk_size = 10000, chunk_overlap = 0, separator="\n")
# list of description chunks, ready to embed.
documents = text_splitter.split_documents(raw_documents)

# Loads a pre-trained Hugging Face sentence embedding model.
# all-MiniLM-L6-v2 is a very efficient, general-purpose model for turning text into dense vectors.
#Each book description is converted into a numerical vector (embedding) capturing semantic meaning.
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Each description chunk is embedded into a vector using embedding_model.
# All vectors are stored inside Chroma DB, which supports fast similarity search.
# This gives you db_books, which you can now query with natural language (e.g., “a book about ghosts”) to retrieve semantically similar books.
db_books = Chroma.from_documents(documents, embedding_model)


"""query → User’s search text (e.g., "books about nature for kids").

category → Optional filter (e.g., Fiction, Non-fiction, etc.).

tone → Optional filter to prioritize emotional tone (Joy, Fear, etc.).

initial_top_k → Get top 50 matches first (bigger pool).

final_top_k → Trim results down to top 16 for display in Gradio.

Returns → A pandas.DataFrame of recommended books."""
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    # Retriever: pulls 50 semantically similar books using embeddings (Chroma).
    retriever = db_books.as_retriever(search_type="similarity",search_kwargs={"k": 50})

    # CrossEncoderReranker: reranks those 50 by directly comparing (query, doc) pairs using relevance scores.
    # ContextualCompressionRetriever (ccr): combines retriever + reranker.
    # base_compressor finally outputs top 10 highest-quality matches.
    ccr = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=CrossEncoderReranker(
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base"),
        top_n=10,  
    ),
    )

    # runs the retriever + reranker pipeline.
    results = ccr.invoke(query)[:initial_top_k] 

    # Extracts the ISBN13s (page_content) from results.
    # Filters the books DataFrame to include only these recommended books.
    # Limits to final_top_k = 16 books for dashboard display.
    books_list = []
    for i in range(0, len(results)):
        books_list += [int(results[i].page_content.strip('"').split()[0])]
    book_recs = books[books["isbn13"].isin(books_list)]

    # If the user picked a specific category (e.g., "Fiction"), only show books from that category.
    # If they picked "All", no filter is applied.
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # Uses emotion probability columns (calculated earlier with sentiment analysis).
    # Sorts recommendations by highest likelihood of that tone.
    # Example: If tone="Happy", sort by the "joy" column so most joyful books appear first.
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    # Returns → A pandas.DataFrame of recommended books."""
    return book_recs

"""query → user’s search text (e.g., “adventure books for kids”).

category → filter (Fiction, Non-fiction, etc.).

tone → emotional filter (Happy, Sad, Suspenseful, etc.)."""
def recommend_books(
        query: str,
        category: str,
        tone: str
):
    # gets a DataFrame of recommended books.
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    #Iterates over each row (book) in the DataFrame.
    # row contains all book info: title, authors, description, thumbnail, etc.
    for _, row in recommendations.iterrows():
        # Takes the book description.
        # Splits into words.
        # Joins only the first 30 words, then adds "..." (to keep text short).
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        # Splits authors by ; (since sometimes multiple authors are stored that way).
        # If 2 authors → "Author1 and Author2".
        # If >2 authors → "Author1, Author2, ..., and LastAuthor".
        # If only 1 author → just that name.
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        # Create Display Caption -> Book Title, Authors (formatted nicely), Truncated Description
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        # First element = thumbnail URL (book cover).
        # Second element = caption (title + author + short description).
        results.append((row["large_thumbnail"], caption))
    return results

# Creates a list of categories for the dropdown. Includes "All" plus all unique values from books["simple_categories"].
categories = ["All"] + sorted(books["simple_categories"].unique())
# Creates a list of emotional tones the user can filter by (also starts with "All").
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# gr.Blocks → Gradio’s container for building complex UIs.
# theme=gr.themes.Glass() → Adds a nice glass-like UI theme.
# gr.Markdown → Title text at the top of the app.
with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    # Wrapped inside a gr.Row() → arranges inputs side by side.
    # Textbox → User enters a description of the kind of book they want.
    # Dropdown (categories) → User selects Fiction/Non-fiction/etc. (default "All").
    # Dropdown (tones) → User selects tone (Happy, Sad, etc., default "All").
    # Button → When clicked, it triggers the recommendation function.
    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    # Markdown → Adds a subheader “Recommendations”.
    # gr.Gallery → Displays results (book cover + caption) in a nice gallery grid.
    # columns=8, rows=2 → Shows up to 16 recommendations (8 per row × 2 rows).
    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    #Connects the Find recommendations button → to the recommend_books function.
    #Takes inputs (query, category, tone) and sends them to the function.
    #Displays outputs in the output gallery.
    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)


if __name__ == "__main__":
    dashboard.launch()