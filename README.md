# Multimodal Retrieval-Augmented Generation (RAG) System

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline capable of processing and responding to queries based on PDF documents containing both textual and visual information. The system can handle text-based and image-based queries, utilizing a Large Language Model (LLM) integrated with a vector database for contextually relevant responses.

## 1. Task-1: Multimodal RAG System

### 1.1 Objectives

- Implement a RAG pipeline for three provided PDF documents (financial statements, reports with charts/graphs).
- Extract, preprocess, chunk, embed, store, retrieve, and utilize multimodal information (text and images).
- Respond to natural language queries (text and image-based) through a ChatGPT-like interface.
- Test and utilize advanced prompting strategies (Chain-of-Thought, Zero-shot/Few-shot, Prompt Engineering) to enhance LLM reasoning and output quality.

### 1.2 Requirements

**Data Extraction and Preprocessing:**
- Parse PDFs to extract textual content (figures, tables, paragraphs).
- Extract and analyze image content (charts, plots, graphs) using OCR and image processing.
- Store extracted components as chunks with metadata.

**Text and Image Embedding:**
- Convert textual chunks to embeddings using `clip-ViT-B-32` (SentenceTransformer).
- Convert visual chunks (images) to embeddings using `clip-ViT-B-32` (SentenceTransformer/CLIPModel).
- Store embeddings in a ChromaDB vector database with indexing and metadata.

**Semantic Search and Retrieval:**
- Implement similarity-based search using ChromaDB.
- Accept textual queries or uploaded image queries.
- Retrieve and rank relevant chunks (text/image) for a given query.
- Format results coherently with source references.

**Language Model Integration:**
- Integrate an LLM (`mistralai/Mistral-7B-Instruct-v0.1`) using 4-bit quantization (BitsAndBytesConfig).
- Use retrieved information as context for generating answers.
- Employ advanced prompting strategies (Basic, Chain-of-Thought).

**User Interface (Streamlit `app.py`):**
- Web-based interface with:
    - Text box for queries.
    - Image upload for visual queries.
    - Display of ranked relevant chunks, LLM-generated answers.
    - Option to select prompt strategy (Basic/CoT).

**Evaluation and Visualization:**
- Visualize embedding space using t-SNE.
- Measure retrieval hit rates and semantic similarity.
- Evaluate LLM responses using:
    - Relevance (manual).
    - Faithfulness (manual, based on retrieved context).
    - BLEU / ROUGE for generative quality.
    - Query response time.

**Documentation:**
- Technical report (LaTeX, Springer LNCS format - not included here).
- Prompt log (not included here).
- Reproducible pipeline (Jupyter Notebook `genai_task1.ipynb` and Streamlit `app.py`).

### 1.3 Expected Outcomes

- A functional multimodal RAG system.
- Demonstrated reasoning and coherence in LLM responses via prompting.
- Visualization of embeddings and retrieval.
- A deployable web interface.

## Project Structure

- **`genai_task1.ipynb`**: Jupyter Notebook containing the end-to-end RAG pipeline implementation, including:
    - Setup and library installation.
    - Data extraction from PDFs (`UnstructuredPDFLoader`, `PyMuPDF/fitz`, `easyocr`).
    - Text and image chunking and preprocessing.
    - Embedding generation using `openai/clip-vit-base-patch32` via `transformers.CLIPModel` and `transformers.CLIPProcessor`.
    - Vector storage in ChromaDB.
    - Semantic search and retrieval logic.
    - LLM integration (`mistralai/Mistral-7B-Instruct-v0.1` with 4-bit quantization).
    - Prompt engineering (Basic and CoT).
    - Evaluation metrics and visualization (t-SNE, Hit Rate, BLEU, ROUGE).
- **`app.py`**: Streamlit application for the user interface, leveraging the backend functions developed in the notebook.
- **`data/`**: Folder assumed to contain the input PDF documents:
    - `1. Annual Report 2023-24.pdf`
    - `2. financials.pdf`
    - `3. FYP-Handbook-2023.pdf`
- **`extractedpdfimages/`**: Directory created by the notebook to store images extracted from PDFs.
- **`chroma_multimodal_db/`**: Directory created by ChromaDB for persistent storage of the vector database.

## Technologies Used

- **Core Libraries:**
    - `langchain`, `langchain-community`
    - `unstructured[local-inference,pdf]` for PDF parsing
    - `pypdf`, `pdf2image`, `pytesseract`, `Pillow`, `poppler-utils`, `fitz` (PyMuPDF) for PDF/image processing
    - `easyocr` for Optical Character Recognition
    - `chromadb` for vector storage
    - `sentence-transformers` (`clip-ViT-B-32`) for multimodal embeddings (also directly using `transformers.CLIPModel`)
    - `transformers`, `accelerate`, `bitsandbytes`, `torch` for LLM (`mistralai/Mistral-7B-Instruct-v0.1`)
    - `huggingface-hub` for model downloading
- **UI:** `streamlit`
- **Evaluation:** `matplotlib`, `seaborn`, `scikit-learn` (for t-SNE), `rouge-score`, `nltk` (for BLEU)
- **Environment:** Google Colab (notebook designed for GPU acceleration)

## Setup and Execution

### 1. Jupyter Notebook (`genai_task1.ipynb`) - RAG Pipeline and Evaluation

**Prerequisites:**
- Google Colab environment with GPU runtime (e.g., T4).
- Upload the PDF documents to the `/content/` directory in Colab (or adjust paths in the notebook).
- Set up Hugging Face token in Colab Secrets (`HF_TOKEN`) if required by models (Mistral model used here is usually open access, but good practice).

**Steps:**
1.  **Open `genai_task1.ipynb` in Google Colab.**
2.  **Ensure GPU is enabled:** Runtime -> Change runtime type -> Hardware accelerator: GPU.
3.  **Upload Data:** Upload the three PDF files (`1. Annual Report 2023-24.pdf`, `2. financials.pdf`, `3. FYP-Handbook-2023.pdf`) to the `/content/` directory of your Colab instance.
4.  **Execute Cells Sequentially:**
    *   **Phase 1: Setup and Environment:** Installs all necessary dependencies. This might take some time.
    *   **Phase 2: Data Extraction and Preprocessing:** Extracts text, tables, and images (performing OCR on images) from the PDFs. Extracted images are saved to `extractedpdfimages/`.
    *   **Phase 3: Chunk and Embed:** Chunks the extracted text and prepares image data for embedding. Initializes the `openai/clip-vit-base-patch32` model and generates embeddings for all chunks.
    *   **Phase 4: Vector Storage:** Initializes a persistent ChromaDB client and stores the embedded chunks in a collection named `multimodal_rag_collection` at `./chroma_multimodal_db`. *You might be prompted to clear an existing collection if re-running.*
    *   **Phase 5: Semantic Search and Retrieval:** Reconnects to ChromaDB and tests the retrieval function with sample text and image queries.
    *   **Phase 6: Language Model Integration and Generation:** Loads the `mistralai/Mistral-7B-Instruct-v0.1` LLM with 4-bit quantization and tests the full RAG pipeline with basic and CoT prompts.
    *   **Evaluation:** Installs evaluation libraries, defines an evaluation dataset, and runs retrieval and generation metrics (Hit Rate, Similarity, BLEU, ROUGE, Response Time). Includes t-SNE visualization of embeddings. *Manual evaluation scores for relevance and faithfulness are prompted for but not programmatically collected in the notebook.*

### 2. Streamlit Application (`app.py`) - User Interface

**Prerequisites:**
- The `genai_task1.ipynb` notebook must have been run successfully at least once to:
    - Create and populate the ChromaDB in `./chroma_multimodal_db/`.
    - Download the necessary embedding and LLM models to the Hugging Face cache (or local Colab cache).
    - Create the `extractedpdfimages/` directory with extracted images if you plan to display them in the UI.
- Python environment with all dependencies from the notebook installed.
- Hugging Face token available as a Colab Secret named `HF_TOKEN` (the `app.py` script uses `google.colab.userdata` which is specific to Colab. For local deployment, you'll need to modify how the token is accessed, e.g., environment variables).

**Steps (primarily for Colab execution as `app.py` uses Colab-specific userdata):**
1.  **Ensure `app.py` is in your Colab `/content/` directory.**
2.  **Ensure the `chroma_multimodal_db` directory (created by the notebook) is present in `/content/`.**
3.  **Ensure the `extractedpdfimages` directory (created by the notebook) is present if image display within Streamlit is desired.**
4.  **Install `localtunnel` globally (if not already done in the notebook):**
    ```bash
    !npm install -g localtunnel
    ```
5.  **Run the Streamlit app and expose it via localtunnel (from a Colab cell):**
    ```bash
    # Make sure app.py is in the current directory (/content/) or specify the path
    !streamlit run app.py &>/dev/null&

    # Get your public IP (needed for localtunnel in some setups)
    !curl https://loca.lt/mytunnelpassword

    # Expose port 8501 using localtunnel
    !npx localtunnel --port 8501
    ```
6.  **Access the App:** Click the URL provided by `localtunnel` (e.g., `https://your-unique-subdomain.loca.lt`).

**Note for Local `app.py` Execution (outside Colab):**
- You would need to manage the `HF_TOKEN` via environment variables or another secure method.
- Adjust paths for `CHROMA_DB_PATH` and potentially image paths if they are not relative to `app.py`.
- Run Streamlit directly: `streamlit run app.py`.

## Key Components and Logic

- **Multimodal Data Handling:** The system processes both text and images. Images are OCR'd, and both raw images and their OCR'd text can be embedded and retrieved.
- **CLIP Embeddings:** The `clip-ViT-B-32` model (via SentenceTransformers and directly via `transformers.CLIPModel`) is used to generate joint text-image embeddings, allowing for cross-modal retrieval.
- **ChromaDB:** Serves as the vector store for efficient similarity search over the embeddings.
- **Mistral-7B LLM:** A powerful open-source LLM used for generating answers based on the retrieved context. 4-bit quantization is employed to manage resource usage.
- **Prompting Strategies:**
    - **Basic Prompt:** Instructs the LLM to answer based *only* on the provided context.
    - **Chain-of-Thought (CoT) Prompt:** Guides the LLM to follow a step-by-step reasoning process to derive the answer from the context.
- **Streamlit UI:** Provides an interactive way to query the RAG system with text or images and view results.

## Evaluation Insights (from Notebook)

- **t-SNE Visualization:** Helps understand the clustering of text and image embeddings in a 2D space.
- **Retrieval Metrics:**
    - **Hit Rate@K:** Percentage of queries for which at least one of the expected relevant documents is found within the top K retrieved results. The notebook achieved ~80% @K=5.
    - **Mean Average Similarity@K:** Average similarity score of the top K retrieved results, indicating how closely related the retrieved chunks are to the query.
- **Generation Metrics:**
    - **Response Time:** Average time taken to process a query and generate an LLM response.
    - **BLEU & ROUGE:** Standard metrics for evaluating the quality of generated text against reference answers (n-gram overlap, longest common subsequence).
    - **Manual Evaluation (Relevance & Faithfulness):** Placeholder for subjective human assessment of the generated answers' accuracy and grounding in the provided context.

This project demonstrates a robust approach to building a multimodal RAG system, incorporating advanced techniques for data processing, embedding, retrieval, and LLM-based generation.
