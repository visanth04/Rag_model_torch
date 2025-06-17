import os
import fitz
import spacy
import torch
import textwrap
import requests
import gradio as gr
import pandas as pd  # NEW: For CSV handling
import ast  # NEW: For parsing stringified lists
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Tuple

# Text formatting
def text_formatter(text: str) -> str:
    return text.replace("\n", " ").strip()

# Read PDF
def open_and_read_pdf(pdf_path: str) -> List[Dict]:
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        doc = fitz.open(pdf_path)
        pages_and_text = []
        for page_num, page in tqdm(enumerate(doc), total=len(doc), desc="Reading PDF"):
            text = page.get_text()
            text = text_formatter(text)
            pages_and_text.append({
                "page_number": page_num - 10,
                "page_char_count": len(text),
                "page_word_count": len(text.split()),
                "page_sentence_count": len(text.split(". ")),
                "page_token_count": len(text) / 4,
                "text": text
            })
        doc.close()
        return pages_and_text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []

# Sentence processing
def load_spacy_model(model_name: str = "en_core_web_sm") -> spacy.language.Language:
    try:
        nlp = spacy.load(model_name, disable=["ner", "lemmatizer"])
        nlp.add_pipe("sentencizer")
        return nlp
    except OSError as e:
        print(f"Error: SpaCy model '{model_name}' not found. Please install it using: python -m spacy download {model_name}")
        raise

nlp = load_spacy_model("en_core_web_sm")

def process_sentences(items: List[Dict]) -> None:
    texts = [item["text"] for item in items]
    docs = list(nlp.pipe(texts, batch_size=50))
    for item, doc in zip(items, docs):
        item["sentences"] = [str(sent).strip() for sent in doc.sents]
        item["page_sentence_count_spacy"] = len(item["sentences"])

# Sentence chunking
def split_list(input_list: List[str], slice_size: int = 10) -> List[List[str]]:
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

def chunk_sentences(items: List[Dict], chunk_size: int = 10) -> None:
    for item in tqdm(items, desc="Chunking sentences"):
        item["sentence_chunks"] = split_list(item["sentences"], chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])

# Embedding generation (fallback if CSV not available)
def generate_embeddings(chunks: List[Dict], model: SentenceTransformer, batch_size: int = 16) -> torch.Tensor:
    texts = [chunk["sentence_chunk"] for chunk in chunks]
    return model.encode(texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)

# Retrieval
def retrieve_relevant_resources(
    query: str,
    embeddings: torch.Tensor,
    model: SentenceTransformer,
    n_resources_to_return: int = 5,
    use_cosine: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=False)
    scores = util.cos_sim(query_embedding, embeddings)[0] if use_cosine else util.dot_score(query_embedding, embeddings)[0]
    scores, indices = torch.topk(scores, k=n_resources_to_return)
    return scores, indices

# LLM generation
def generate_response(
    query: str,
    context: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    dialogue_template = [
        {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
    ]
    prompt = tokenizer.apply_chat_template(
        conversation=dialogue_template,
        tokenize=False,
        add_generation_prompt=True
    )
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(
        **input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("<start_of_turn>model")[1].strip()

# Download files from Google Drive
def download_file(url: str, dest: str):
    if not os.path.exists(dest):
        try:
            file_id = url.split("id=")[1] if "id=" in url else url.split("/d/")[1].split("/")[0]
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            response = requests.get(download_url)
            if response.status_code == 200:
                with open(dest, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {dest} from Google Drive")
            else:
                raise Exception(f"Failed to download {dest}. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error downloading {dest}: {e}")
            raise

# Initialize models and data
pdf_path = "book.pdf"
embedding_file = "embeddings.csv"  # CHANGED: Use CSV instead of .pt
pdf_url = os.getenv("PDF_URL", "")
embedding_url = os.getenv("EMBEDDING_URL", "")
if pdf_url:
    download_file(pdf_url, pdf_path)
if embedding_url:
    download_file(embedding_url, embedding_file)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
pages_and_text = open_and_read_pdf(pdf_path)
if not pages_and_text:
    raise RuntimeError("Failed to load PDF content")

process_sentences(pages_and_text)
chunk_sentences(pages_and_text)

# Create pages_and_chunks from PDF processing
pages_and_chunks = [
    {"page_number": item["page_number"], "sentence_chunk": " ".join(chunk)}
    for item in pages_and_text
    for chunk in item["sentence_chunks"]
]

# Load embeddings from CSV
if os.path.exists(embedding_file):
    # NEW: Read CSV and convert embeddings to tensor
    df = pd.read_csv(embedding_file)
    # Assume 'embedding' column contains stringified lists, e.g., "[0.1, 0.2, ...]"
    embeddings = torch.tensor([ast.literal_eval(emb) for emb in df["embedding"]], dtype=torch.float32)
    # Update pages_and_chunks with CSV data (ensuring alignment)
    pages_and_chunks = [
        {"page_number": row["page_number"], "sentence_chunk": row["sentence_chunk"]}
        for _, row in df.iterrows()
    ]
else:
    # Fallback: Generate embeddings if CSV is missing
    embeddings = generate_embeddings(pages_and_chunks, embedding_model)
    # Save to CSV for future use
    embedding_list = embeddings.cpu().numpy().tolist()
    df = pd.DataFrame([
        {"page_number": chunk["page_number"], "sentence_chunk": chunk["sentence_chunk"], "embedding": emb}
        for chunk, emb in zip(pages_and_chunks, embedding_list)
    ])
    df.to_csv(embedding_file, index=False)

# Load LLM
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    low_cpu_mem_usage=False,
    attn_implementation="sdpa"
)
if torch.cuda.is_available():
    llm_model.to("cuda")
else:
    llm_model.to("cpu")

# Gradio interface
def process_query(query, n_resources=5, use_cosine=False):
    try:
        scores, indices = retrieve_relevant_resources(
            query, embeddings, embedding_model, n_resources, use_cosine
        )
        context = pages_and_chunks[indices[0]]["sentence_chunk"]
        response = generate_response(query, context, tokenizer, llm_model)
        return (
            response,
            context,
            f"Score: {scores[0].item():.4f}",
            f"Page Number: {pages_and_chunks[indices[0]]['page_number']}"
        )
    except Exception as e:
        return f"Error: {str(e)}", "", "", ""

iface = gr.Interface(
    fn=process_query,
    inputs=[
        gr.Textbox(label="Query", value="psychotic behavior in the postnatal period"),
        gr.Slider(1, 10, value=5, step=1, label="Number of Resources"),
        gr.Checkbox(label="Use Cosine Similarity", value=False)
    ],
    outputs=[
        gr.Textbox(label="Response"),
        gr.Textbox(label="Top Retrieved Chunk"),
        gr.Textbox(label="Score"),
        gr.Textbox(label="Page Number")
    ],
    title="RAG Model Demo for Obstetrics",
    description="Enter a query to retrieve and generate answers from 'Essentials of Obstetrics'."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)