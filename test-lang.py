import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import transformers
import torch
from transformers import BitsAndBytesConfig
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
import numpy as np
from scipy.spatial.distance import cosine


# Charger les embeddings stockés dans data.json
with open("data.json", "r", encoding="utf-8") as f:
    embedding_data = json.load(f)

# Extraire les textes et leurs embeddings
texts = [entry["text"] for entry in embedding_data.values()]
embeddings = [entry["embedding"] for entry in embedding_data.values()]
embeddings = np.array(embeddings)  # Conversion pour calculs rapides


embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

prompt_template = PromptTemplate.from_template("""Tu es un assistant pour des tâches de questions-réponses. Utilise les morceaux de contexte suivants pour répondre à la question. 
Si tu ne connais pas la réponse, dis simplement que tu ne sais pas. Utilise trois phrases maximum et garde la réponse concise.
Question: {question} 
Contexte: {context} 
Réponse: """)

def rag_pipeline(query):
    # Étape 1 : Encoder la requête
    query_embedding = embedding.embed_query(query)

    # Étape 2 : Calculer les similarités cosinus
    similarities = [1 - cosine(query_embedding, emb) for emb in embeddings]

    # Étape 3 : Sélectionner les 3 documents les plus proches
    top_indices = np.argsort(similarities)[-3:][::-1]
    top_texts = [texts[i] for i in top_indices]

    # Étape 4 : Construire le prompt avec les documents les plus pertinents
    context = "\n\n".join(top_texts)
    prompt = prompt_template.invoke({
        "question": query,
        "context": context
    })

    # Étape 5 : Appeler le LLM avec le prompt
    return llm.invoke(prompt).strip()

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# Chargement de la configuration du modèle
model_config = transformers.AutoConfig.from_pretrained(model_id)

# Initialiser le tokeniseur
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    device_map='auto'
)
llm=HuggingFacePipeline(
    pipeline=pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=4096,
        do_sample=False,
        return_full_text=False  # Très important ! On ne veut pas le prompt initial
    )
)
# Étape 1 : Charger le texte brut depuis text.txt
with open("text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Étape 2 : Créer un objet Document
documents = [Document(page_content=text)]

# Étape 3 : Diviser le texte en chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=60,
    length_function=len,
    separators=["\n\n", "\n"]
)
chunks = text_splitter.split_documents(documents=documents)




# Étape 5 : Calculer les embeddings
texts = [chunk.page_content for chunk in chunks]
embeddings = embedding.embed_documents(texts)

# Étape 6 : Sauvegarder les embeddings dans un fichier JSON
embeddings_data = {
    f"chunk_{i}": {
        "text": texts[i],
        "embedding": embeddings[i]
    }
    for i in range(len(texts))
}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(embeddings_data, f, ensure_ascii=False, indent=2)

print(f"{len(embeddings)} embeddings ont été générés et sauvegardés dans 'data.json'.")


query = """
Qui est le responsable des informaticien au sein de voxens
"""

# Effectuer une requête
response = rag_pipeline(query)
print(response)