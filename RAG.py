import streamlit as st
from sentence_transformers import SentenceTransformer, util
import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai

# Configuration de la page Streamlit
st.set_page_config(
    page_title="RAG Recherche d'information",
    page_icon="FR",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.header('Bienvenue sur RAG !!')

# Fonction pour extraire le texte et les tables des fichiers PDF
def extract_text_and_tables(
    directory_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    results = []

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    raw_text = page.extract_text() or ""
                    raw_text = raw_text.strip()

                    if raw_text:
                        chunks = text_splitter.split_text(raw_text)
                        results.extend(chunks)

                    page_tables = page.extract_tables() or []
                    for table in page_tables:
                        if not table:
                            continue

                        lines = []
                        for row in table:
                            cleaned_row = [cell if cell else "" for cell in row]
                            line = " | ".join(cleaned_row)
                            lines.append(line)

                        table_text = "\n".join(lines).strip()
                        if table_text:
                            table_chunks = text_splitter.split_text(table_text)
                            results.extend(table_chunks)

    return results

# Fonction pour trouver les segments les plus similaires
def find_similar_segments(doc_list, query, top_n=10):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(doc_list)
    query_embedding = model.encode(query)
    similarities = util.cos_sim(query_embedding, embeddings)
    top_results = similarities.topk(top_n)[1]
    top_n_paragraphs = [doc_list[i] for i in top_results[0]]
    return top_n_paragraphs

# Fonction pour interagir avec l'API OpenAI
def get_openai_response(api_key, top_n_paragraphs, query):
    openai.api_key = api_key
    context = "\n".join(top_n_paragraphs)
    prompt = f"Contexte :\n{context}\n\nQuestion : {query}\nRéponse :"
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Modèle GPT-4, assurez-vous de l'ID correct
        messages=[
            {"role": "system", "content": "Tu es un assistant spécialisé dans la recherche d'information à partir de documents fournis. Tes réponses doivent absolument provenir du contexte fourni."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    generated_response = response['choices'][0]['message']['content'].strip()
    return generated_response



# Interface utilisateur Streamlit
st.sidebar.title("Charger des fichiers PDF")
directory_path = st.sidebar.text_input("Entrez le chemin du répertoire contenant les PDFs")

if directory_path:
    # Extraire les textes et les tables
    doc_list = extract_text_and_tables(directory_path)
    if doc_list:
        st.write(f"Nombre de segments extraits : {len(doc_list)}")
    else:
        st.write("Aucun texte ou table extrait.")

# Zone pour saisir une requête
query = st.text_input("Entrez votre question :", "")

if query:
    if doc_list:
        # Trouver les segments similaires
        top_n_paragraphs = find_similar_segments(doc_list, query)

        # Demander la réponse à OpenAI
        api_key = ""
        response = get_openai_response(api_key, top_n_paragraphs, query)

        # Afficher la réponse générée
        st.write("Réponse générée :")
        st.write(response)
    else:
        st.write("Veuillez d'abord charger des fichiers PDF pour effectuer la recherche.")
