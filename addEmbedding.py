import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Initialisation du séparateur et du modèle d'embedding
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n"]
)

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                  encode_kwargs={"normalize_embeddings": True})

# Ton nouveau texte
voxens_histoire = """
Voxens : Une histoire de croissance et de convictions

En 2009, avec seulement 37 positions, Voxens est créé à Rouen. À l'époque, alors que tous les acteurs du marché des centres d'appel font le pari de l'offshore et de la délocalisation, Voxens décide de prendre un risque audacieux : celui de s'implanter en France, une démarche radicalement opposée à la tendance générale du marché.

En 2012, l'entreprise fait son entrée dans le Top 50 des outsourceurs, selon l'étude EY-SP2C. En seulement trois ans, Voxens a démontré que son choix stratégique de baser ses sites exclusivement en France était non seulement un moteur de croissance pour l'entreprise, mais aussi un véritable avantage pour les clients qu'elle sert.

Cinq ans plus tard, en 2015, Voxens crée son logo emblématique "Normand'Shore", célébrant fièrement ses racines normandes et son ancrage dans le territoire français. Cette même année, l'entreprise continue d'évoluer et en 2017, elle atteint les 350 emplois. Chaque emploi est essentiel pour Voxens, qui continue d'investir dans la formation et la création d'opportunités professionnelles, que ce soit en apprentissage ou en poste. Grâce à cette synergie, l'équipe de Voxens devient plus déterminée que jamais à dépasser les attentes de ses clients.

En 2018, Voxens figure pour la quatrième fois parmi les "Champions de la Croissance" selon le classement des Echos. Cette distinction vient souligner une fois de plus le dynamisme et l'ambition de l'entreprise. Mais cette année marque aussi un tournant dans l'histoire de Voxens : l'entreprise décide de rejoindre le groupe Stelliant, une rencontre stratégique qui permet à Voxens de continuer à se développer aux côtés d'un acteur partageant ses valeurs. Cette collaboration permet à Voxens de proposer des prestations toujours plus performantes et optimisées.

En 2021, Voxens renouvelle cette distinction en figurant à nouveau dans le palmarès des "Champions de la Croissance" des Echos, confirmant sa position de leader dans son secteur. Cette année-là, l'entreprise décide également de prendre une nouvelle direction en créant un pôle offshore à Madagascar. Cette initiative, loin de renier les convictions françaises de l'entreprise, est née du souhait de son fondateur, Chakil Mahter, de créer des opportunités de carrière dans son pays natal, tout en continuant à satisfaire les besoins de ses clients. En 2022, Voxens compte 800 positions mobilisables, dont 250 à Madagascar.

Enfin, avec des talents comme Mehdi, chef des développeurs au sein de Voxens, l'entreprise continue de renforcer son équipe et son expertise technique. Voxens poursuit ainsi sa mission : allier croissance, innovation et valeurs humaines.
"""

# 1. Charger l'ancien embeddings.json
if os.path.exists("embeddings.json"):
    with open("embeddings.json", "r") as f:
        old_embeddings = json.load(f)
else:
    old_embeddings = {}

# 2. Créer des chunks à partir du nouveau texte
new_documents = text_splitter.split_documents([Document(page_content=voxens_histoire)])

# 3. Encoder les nouveaux chunks
new_embeddings = embedding.embed_documents([doc.page_content for doc in new_documents])

# 4. Préparer les nouveaux embeddings dans le même format
starting_index = len(old_embeddings) + 1  # Pour éviter d'écraser les anciens
new_embeddings_data = {
    f"doc_{starting_index + i}": emb for i, emb in enumerate(new_embeddings)
}

# 5. Fusionner ancien + nouveau
all_embeddings = {**old_embeddings, **new_embeddings_data}

# 6. Sauvegarder dans embeddings.json
with open("embeddings.json", "w") as f:
    json.dump(all_embeddings, f)

print(f"Ajouté {len(new_embeddings)} nouveaux embeddings dans 'embeddings.json' !")
