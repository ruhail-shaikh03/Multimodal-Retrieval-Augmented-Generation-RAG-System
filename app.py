# app.py
import streamlit as st
from PIL import Image
import os
import torch
import numpy as np
import time
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from google.colab import userdata # For HF Token

# --- Configuration ---
CHROMA_DB_PATH = "./chroma_multimodal_db" # Make sure this path is correct
COLLECTION_NAME = "multimodal_rag_collection"
EMBEDDING_MODEL_NAME = 'clip-ViT-B-32'
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
DEFAULT_PROMPT_TYPE = 'basic'
N_RESULTS_RETRIEVAL = 5

# --- Helper: Get HF Token ---
# Wrapped in a function to easily handle potential errors
@st.cache_data # Cache the token value itself
def get_hf_token():
    try:
        token = userdata.get('HF_TOKEN')
        if not token:
            st.warning("Hugging Face token not found in Colab Secrets. LLM loading might fail if model requires authentication.", icon="⚠️")
        return token
    except Exception as e:
        st.error(f"Error retrieving Hugging Face token: {e}")
        return None

# --- Cached Resource Loading ---

@st.cache_resource # Caches the actual model object
def load_embedding_model():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    if torch.cuda.is_available():
        model = model.to('cuda')
        print("Embedding model moved to GPU.")
    else:
        print("GPU not available for embedding model, using CPU.")
    return model

@st.cache_resource # Caches the LLM model and tokenizer objects
def load_llm_and_tokenizer():
    print(f"Loading LLM: {LLM_MODEL_NAME}...")
    hf_token = get_hf_token()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    try:
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            token=hf_token
        )
        tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME,
            token=hf_token
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            llm_model.config.pad_token_id = tokenizer.eos_token_id
        print("LLM and Tokenizer loaded.")
        return llm_model, tokenizer
    except Exception as e:
        st.error(f"Error loading LLM/Tokenizer: {e}. Please ensure model access and token are correct.")
        return None, None

@st.cache_resource # Caches the ChromaDB client and collection objects
def load_chroma_collection():
    print("Connecting to ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        print(f"Connected to ChromaDB collection '{COLLECTION_NAME}' with {count} items.")
        if count == 0:
            st.warning("ChromaDB collection is empty. Retrieval will not work. Please run the data loading/embedding phases first.", icon="⚠️")
        return collection
    except Exception as e:
        st.error(f"Error connecting to ChromaDB: {e}. Please ensure the path '{CHROMA_DB_PATH}' exists and contains the DB.")
        return None

# --- Load Resources ---
embedding_model = load_embedding_model()
llm_model, tokenizer = load_llm_and_tokenizer()
collection = load_chroma_collection()

# --- RAG Backend Functions (Copied from previous steps) ---

def retrieve_relevant_chunks(query, n_results=N_RESULTS_RETRIEVAL):
    if collection is None or embedding_model is None:
        st.error("Database or embedding model not loaded correctly.")
        return None
    query_vector = None
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if isinstance(query, str):
            query_vector = embedding_model.encode(query, convert_to_tensor=True, device=device)
        elif isinstance(query, Image.Image):
            query_vector = embedding_model.encode(query, convert_to_tensor=True, device=device)
        else:
            st.error("Invalid query type for retrieval.")
            return None

        query_vector_list = query_vector.cpu().numpy().tolist()

        results = collection.query(
            query_embeddings=[query_vector_list],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )

        formatted_results = []
        if results and results.get('ids') and results['ids'][0]:
            ids, distances, metadatas, documents = results['ids'][0], results['distances'][0], results['metadatas'][0], results['documents'][0]
            for i in range(len(ids)):
                similarity = max(0.0, 1.0 - distances[i])
                formatted_results.append({
                    'id': ids[i], 'distance': distances[i], 'similarity': similarity,
                    'metadata': metadatas[i], 'content': documents[i]
                })
            return formatted_results
        else:
            return []
    except Exception as e:
        st.error(f"Error during retrieval: {e}")
        return None

def format_context_for_llm(results):
    context_str = "--- Retrieved Context ---\n"
    if not results:
        return context_str + "No relevant context found.\n--- End Context ---"
    for i, res in enumerate(results):
        metadata = res.get('metadata', {})
        content = res.get('content', '')
        source_file = os.path.basename(metadata.get('source', 'N/A'))
        page = metadata.get('page', 'N/A')
        chunk_type = metadata.get('type', 'N/A')
        similarity = res.get('similarity', 'N/A')
        similarity_str = f"{similarity:.4f}" if isinstance(similarity, (int, float)) else str(similarity)
        context_str += f"\nChunk {i+1} (Source: {source_file}, Page: {page}, Type: {chunk_type}, Similarity: {similarity_str}):\n{content}\n---\n"
    context_str += "--- End Context ---"
    return context_str

def create_mistral_basic_prompt(query, context):
    if not isinstance(query, str):
        query = "Describe the visual content based on the provided text context extracted from the image or retrieved relevant text."
    prompt = f"""<s>[INST] You are a helpful AI assistant. Answer the following query based *only* on the provided context. If the context does not contain the answer, state that the information is not available in the provided documents. Do not add information that is not present in the context.

    Query: {query}

    Context:
    {context}

    Answer: [/INST]"""
    return prompt

def create_mistral_cot_prompt(query, context):
    if not isinstance(query, str):
        query = "Analyze the visual content based on the provided text context extracted from the image or relevant text, thinking step-by-step."
    prompt = f"""<s>[INST] You are a helpful AI assistant. Answer the following query based *only* on the provided context. Follow these steps:
    1. Carefully read the query and the provided context.
    2. Identify the key pieces of information within the context that are relevant to the query.
    3. Synthesize these pieces of information to formulate a step-by-step reasoning process.
    4. Based on the reasoning, provide a final answer to the query.
    5. If the context does not contain the information needed to answer the query, explicitly state that.
    Do not add external information.

    Query: {query}

    Context:
    {context}

    Reasoning Steps and Final Answer: [/INST]
    Reasoning:
    1. """
    return prompt

def generate_local_llm_answer(prompt, model, tokenizer, max_new_tokens=300):
    if not model or not tokenizer:
        return "[LLM not loaded correctly]"
    start_gen_time = time.time()
    print("Generating LLM response...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=0.7,
                do_sample=True, top_p=0.9, pad_token_id=tokenizer.pad_token_id
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_part = generated_text.split("[/INST]")[-1].strip()
        end_gen_time = time.time()
        print(f"LLM generation took {end_gen_time - start_gen_time:.2f} seconds.")
        return response_part
    except Exception as e:
        st.error(f"Error during LLM generation: {e}")
        return "[LLM generation failed]"

def get_rag_response_streamlit(query, prompt_type='basic', n_results=N_RESULTS_RETRIEVAL):
    start_pipeline_time = time.time()
    # 1. Retrieve
    results = retrieve_relevant_chunks(query, n_results=n_results)
    if results is None: return {"error": "Retrieval failed."}

    # 2. Format context
    formatted_context = format_context_for_llm(results)

    # 3. Create prompt
    if prompt_type == 'basic':
        final_prompt = create_mistral_basic_prompt(query, formatted_context)
    elif prompt_type == 'cot':
        final_prompt = create_mistral_cot_prompt(query, formatted_context)
    else:
        final_prompt = create_mistral_basic_prompt(query, formatted_context)

    # 4. Generate answer
    llm_answer = generate_local_llm_answer(final_prompt, llm_model, tokenizer)
    end_pipeline_time = time.time()

    return {
        "query_display": query if isinstance(query, str) else f"[Image Query: {getattr(query, 'filename', 'Uploaded Image')}]",
        "retrieved_context_formatted": formatted_context,
        "prompt_used": final_prompt,
        "llm_answer": llm_answer,
        "retrieved_chunks_details": results,
        "total_time": end_pipeline_time - start_pipeline_time
    }


# --- Streamlit UI ---

st.set_page_config(layout="wide") # Use wide layout
st.title("📄 Multimodal RAG Document Q&A 🖼️")
st.markdown("Ask questions about the uploaded PDF documents (Financial Statements, University Report, FYP Handbook) or upload an image for visual queries.")

# Check if resources loaded
if not all([embedding_model, llm_model, tokenizer, collection]):
    st.error("Core resources (models or database) failed to load. Please check the Colab logs. The app cannot function.")
    st.stop() # Stop execution if resources aren't available

# --- User Inputs ---
col1, col2 = st.columns([2, 1]) # Create columns for inputs

with col1:
    text_query = st.text_area("Enter your text query:", height=100, key="text_query")

with col2:
    uploaded_image = st.file_uploader("Or upload an image query:", type=["png", "jpg", "jpeg"], key="image_query")
    # Display uploaded image preview
    if uploaded_image is not None:
        st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)


prompt_strategy = st.radio(
    "Select Prompt Strategy:",
    ('Basic Prompt', 'Chain-of-Thought (CoT) Prompt'),
    key='prompt_strategy', horizontal=True)

submit_button = st.button("Get Answer", key="submit")

# --- Process Query and Display Results ---
if submit_button:
    query_input = None
    query_type = None

    # Determine query type and content
    if uploaded_image is not None:
        try:
            query_input = Image.open(uploaded_image)
            # Store filename if available, otherwise indicate it's from upload
            query_input.filename = getattr(uploaded_image, 'name', 'Uploaded Image')
            query_type = "image"
            st.info("Processing image query...")
        except Exception as e:
            st.error(f"Error opening uploaded image: {e}")
            query_input = None
    elif text_query.strip():
        query_input = text_query.strip()
        query_type = "text"
        st.info("Processing text query...")
    else:
        st.warning("Please enter a text query or upload an image.")

    # If we have valid input, run the RAG pipeline
    if query_input:
        selected_prompt = 'cot' if prompt_strategy == 'Chain-of-Thought (CoT) Prompt' else 'basic'

        with st.spinner("Retrieving context and generating answer... please wait."):
            response_data = get_rag_response_streamlit(query_input, prompt_type=selected_prompt)

        st.divider() # Visual separator

        if response_data and "error" not in response_data:
            # Display Answer
            st.subheader("💬 Answer")
            st.markdown(response_data['llm_answer']) # Use markdown for better formatting
            st.caption(f"Total processing time: {response_data['total_time']:.2f} seconds")

            st.divider()

            # Display Retrieved Context
            st.subheader("📚 Retrieved Context Chunks")
            retrieved_chunks = response_data.get('retrieved_chunks_details', [])

            if retrieved_chunks:
                for i, chunk in enumerate(retrieved_chunks):
                    metadata = chunk.get('metadata', {})
                    content = chunk.get('content', '[No Content]')
                    source = os.path.basename(metadata.get('source', 'N/A'))
                    page = metadata.get('page', 'N/A')
                    chunk_type = metadata.get('type', 'N/A')
                    similarity = chunk.get('similarity', 'N/A')
                    similarity_str = f"{similarity:.4f}" if isinstance(similarity, (int, float)) else str(similarity)
                    image_path = metadata.get('image_path', None)

                    expander_title = f"Chunk {i+1}: Type: {chunk_type}, Source: {source}, Page: {page}, Similarity: {similarity_str}"
                    with st.expander(expander_title):
                        # Display image if it's an image chunk and path exists
                        if chunk_type == 'image' and image_path and os.path.exists(image_path):
                             try:
                                 st.image(image_path, caption=f"Source Image (Page {page})")
                             except Exception as img_e:
                                 st.warning(f"Could not display image {image_path}: {img_e}")
                        elif chunk_type == 'image':
                             st.warning(f"Image chunk found but path '{image_path}' is missing or invalid.")

                        # Display text content (original text, table text, or OCR text)
                        st.write("Content:")
                        st.text(content) # Use st.text to preserve formatting/whitespace better

                        # Display full metadata for debugging/reference
                        with st.container(border=True):
                             st.caption("Full Metadata:")
                             st.json(metadata, expanded=False) # Show metadata as collapsible JSON

            else:
                st.info("No context chunks were retrieved for this query.")

            # Optional: Display the prompt used (for debugging)
            # with st.expander("Show Prompt Used"):
            #    st.text(response_data['prompt_used'])

        else:
            st.error(f"Failed to get response. Error: {response_data.get('error', 'Unknown error')}")