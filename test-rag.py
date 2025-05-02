import transformers
import os
import glob
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.schema import Document
from scipy.spatial.distance import cosine

# Texte de l'histoire de Voxens
voxens_histoire = """
Voxens : Une histoire de croissance et de convictions

En 2009, avec seulement 37 positions, Voxens est créé à Rouen. À l'époque, alors que tous les acteurs du marché des centres d'appel font le pari de l'offshore et de la délocalisation, Voxens décide de prendre un risque audacieux : celui de s'implanter en France, une démarche radicalement opposée à la tendance générale du marché.

En 2012, l'entreprise fait son entrée dans le Top 50 des outsourceurs, selon l'étude EY-SP2C. En seulement trois ans, Voxens a démontré que son choix stratégique de baser ses sites exclusivement en France était non seulement un moteur de croissance pour l'entreprise, mais aussi un véritable avantage pour les clients qu'elle sert.

Cinq ans plus tard, en 2015, Voxens crée son logo emblématique "Normand'Shore", célébrant fièrement ses racines normandes et son ancrage dans le territoire français. Cette même année, l'entreprise continue d'évoluer et en 2017, elle atteint les 350 emplois. Chaque emploi est essentiel pour Voxens, qui continue d'investir dans la formation et la création d'opportunités professionnelles, que ce soit en apprentissage ou en poste. Grâce à cette synergie, l'équipe de Voxens devient plus déterminée que jamais à dépasser les attentes de ses clients.

En 2018, Voxens figure pour la quatrième fois parmi les "Champions de la Croissance" selon le classement des Echos. Cette distinction vient souligner une fois de plus le dynamisme et l'ambition de l'entreprise. Mais cette année marque aussi un tournant dans l'histoire de Voxens : l'entreprise décide de rejoindre le groupe Stelliant, une rencontre stratégique qui permet à Voxens de continuer à se développer aux côtés d'un acteur partageant ses valeurs. Cette collaboration permet à Voxens de proposer des prestations toujours plus performantes et optimisées.

En 2021, Voxens renouvelle cette distinction en figurant à nouveau dans le palmarès des "Champions de la Croissance" des Echos, confirmant sa position de leader dans son secteur. Cette année-là, l'entreprise décide également de prendre une nouvelle direction en créant un pôle offshore à Madagascar. Cette initiative, loin de renier les convictions françaises de l'entreprise, est née du souhait de son fondateur, Chakil Mahter, de créer des opportunités de carrière dans son pays natal, tout en continuant à satisfaire les besoins de ses clients. En 2022, Voxens compte 800 positions mobilisables, dont 250 à Madagascar.

Enfin, avec des talents comme Mehdi, chef des développeurs au sein de Voxens, l'entreprise continue de renforcer son équipe et son expertise technique. Voxens poursuit ainsi sa mission : allier croissance, innovation et valeurs humaines.
"""

# Initialiser le séparateur de texte pour diviser les documents en morceaux
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,  # Taille maximale des morceaux de texte
    chunk_overlap=100,  # Augmenter le chevauchement pour un meilleur contexte
    length_function=len,
    separators=["\n\n", "\n"]
)

# Charger le modèle d'encodage de texte - utiliser un modèle existant et fiable
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                  encode_kwargs={"normalize_embeddings": True})

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
try:
    tokenizer_sum = AutoTokenizer.from_pretrained(model_name)
    model_sum = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model_sum, tokenizer=tokenizer_sum)
except Exception as e:
    print(f"Erreur lors du chargement du modèle de génération: {e}")
    # Fallback sur un modèle plus simple
    qa_model_id = "distilbert-base-cased-distilled-squad"
    qa_pipeline = pipeline("question-answering", model=qa_model_id)
    summarizer = None

# Liste pour stocker les documents chargés
documents = []

# Charger les fichiers PDF dans le dossier "pdf"
if os.path.exists("pdf"):
    for file in glob.glob("pdf/*.pdf"):
        try:
            file_path = os.path.normpath(file)
            loader = PyPDFLoader(file_path)
            documents += loader.load()
        except Exception as e:
            print(f"Erreur survenue pour le fichier '{file}': {e}")
else:
    print("Le dossier 'pdf' n'existe pas. Utilisation uniquement du texte sur Voxens.")

# Diviser les documents en morceaux (chunks)
chunks = []
if documents:
    chunks = text_splitter.split_documents(documents=documents)

# Ajouter 'voxens_histoire' dans les documents à traiter
voxens_chunks = text_splitter.split_documents([Document(page_content=voxens_histoire)])
chunks.extend(voxens_chunks)

# Encoder les morceaux de texte
encoded_embeddings = embedding.embed_documents([chunk.page_content for chunk in chunks])
print(f"Nombre de chunks encodés: {len(encoded_embeddings)}")

# Sauvegarder les embeddings dans un fichier JSON
embeddings_data = {f"doc_{i + 1}": embedding for i, embedding in enumerate(encoded_embeddings)}
with open("embeddings.json", "w") as f:
    json.dump(embeddings_data, f)

print(f"{len(encoded_embeddings)} embeddings ont été ajoutés et sauvegardés dans 'embeddings.json'.")

def rag_pipeline(query):
    # Encoder la question
    query_embedding = embedding.embed_query(query)

    # Calculer la similarité entre l'embedding de la question et ceux des documents
    similarities = []
    for doc, doc_embedding in zip(chunks, encoded_embeddings):
        similarity = 1 - cosine(query_embedding, doc_embedding)
        similarities.append((doc, similarity))

    # Trier les documents par ordre décroissant de similarité
    retrieved_docs = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]  # Top 3 documents

    # Vérification avec le meilleur document après tri
    max_similarity = retrieved_docs[0][1]

    # Nouveau seuil plus strict
    if max_similarity < 0.7:
        return "Désolé, je n'ai pas d'information pertinente pour cette question."

    # Générer le contexte complet pour le prompt
    context = "\n\n".join([doc[0].page_content for doc in retrieved_docs])

    if summarizer:
        try:
            input_text = f"Question: {query}\n\nContexte: {context}\n\nRésumez les informations pertinentes du contexte pour répondre à la question."

            # Limiter à 512 tokens
            input_ids = tokenizer_sum(input_text, truncation=True, max_length=512, return_tensors="pt").input_ids

            # Générer un résumé
            summary = summarizer(input_text, max_length=200, min_length=50, do_sample=False)[0]['summary_text']

            # Protection supplémentaire sur le résumé
            if len(summary.strip()) < 30 or "je ne peux pas" in summary.lower() or "je n'ai pas" in summary.lower():
                return "Je ne peux pas répondre à cette question avec les informations disponibles."

            return summary

        except Exception as e:
            print(f"Erreur lors de la génération du résumé: {e}")
            return "Désolé, une erreur est survenue en générant la réponse."

    # Fallback: sans summarizer
    response = "Basé sur les informations disponibles, voici ce que je peux vous dire:\n\n"
    response += retrieved_docs[0][0].page_content + "\n\n"

    return response



'''
# Test avec une question sur Voxens
query = "Ou est ce qu'on a créer voxens en pmremier"
response = rag_pipeline(query)
print("\nRéponse à la question:", response)
'''


# Autre exemple de question
query = "Qui est le supérieur  des dev au sein de voxens"
response = rag_pipeline(query)
print("\nRéponse à la question:", response)
