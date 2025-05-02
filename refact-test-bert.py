import torch
import nltk
from transformers import BertModel, BertTokenizer, BertForQuestionAnswering
from transformers import pipeline, logging
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import multiprocessing

logging.set_verbosity_error()

# Télécharger les tokenizers NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Charger les modèles
# 1. Modèle BERT de base pour l'encodage
bert_encoder = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 2. Modèle BERT pour répondre aux questions
bert_qa = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# Essayer de charger vos poids personnalisés s'ils existent
try:
    state_dict = torch.load("./model/pytorch_model.bin", map_location="cpu")
    bert_encoder.load_state_dict(state_dict)
    print("Modèle personnalisé chargé avec succès!")
except:
    print("Utilisation du modèle BERT pré-entraîné")

# Mettre les modèles en mode évaluation
bert_encoder.eval()
bert_qa.eval()

# Créer un pipeline pour les questions-réponses sans multiprocessing
qa_pipeline = pipeline('question-answering', model=bert_qa, tokenizer=tokenizer, device=-1)

# Dataset voxens base
knowledge_voxens = [
    "En 2009 avec quelques 37 positions, Voxens est créé à Rouen. Alors que tous les acteurs du marché des centres d'appel font le pari de l'offshore et de la délocalisation, nous faisons un pari risqué, celui de nous baser en France, à l'inverse totale du mouvement.",
    "En 2012 nous entrons dans le Top 50 des outsourceurs selon l'étude EY-SP2C. En seulement 3 ans, Voxens a su démontrer que sa conviction première, à savoir baser ses sites en France exclusivement, est un véritable moteur de croissance pour l'entreprise mais aussi et avant tout pour le client que nous servons.",
    "En 2015 création de notre logo Normand'Shore. Nous célébrons fièrement la région d'origine de Voxens et notre appartenance au territoire français.",
    "En 2017 Voxens compte désormais 350 emplois. Parce que chaque emploi compte, Voxens attache une grande importance à la création d'opportunités pour tous et forme chaque année encore davantage de personnes, qu'elles soient en apprentissage ou en poste. Une synergie se met en place et de cette dernière en résulte une équipe déterminée à atteindre les objectifs de nos clients et même à les dépasser.",
    "En 2018 pour la quatrième fois dans son histoire, Voxens fait partie du palmarès des Champions de la Croissance par Les Echos, prouvant encore une fois le dynamisme et l'ambition qui nous animent. Cette année marque un véritable tournant dans notre histoire puisque c'est aussi l'année où nous décidons de rejoindre le groupe Stelliant. Cette rencontre nous permet alors de continuer à nous développer avec un acteur partageant nos valeurs, pour vous proposer des prestations toujours plus optimisées et performantes.",
    "En 2021 pour la quatrième fois dans son histoire, Voxens fait partie du palmarès des Champions de la Croissance par Les Echos, prouvant encore une fois le dynamisme et l'ambition qui nous animent. Cette année marque un véritable tournant dans notre histoire puisque c'est aussi l'année où nous décidons de rejoindre le groupe Stelliant. Cette rencontre nous permet alors de continuer à nous développer avec un acteur partageant nos valeurs, pour vous proposer des prestations toujours plus optimisées et performantes.",
    "En 2022 nous ouvrons notre unique pôle en offshore, à Madagascar. Loin de renoncer à notre ambition française, la création de ce pôle née de la volonté du fondateur de Voxens, Chakil Mahter, de créer des opportunités de carrières dans son pays natal, qui bénéficieraient également à nos clients. Voxens compte alors 800 positions mobilisables dont 250 à Madagascar.",
    "Mehdi est chef des développeurs au sein de voxens."
]

# Corpus de connaissances (simuler des datasets prédéfinis)
knowledge_base = [
    "BERT est un modèle de langage développé par Google en 2018. Il signifie Bidirectional Encoder Representations from Transformers et a révolutionné le NLP. BERT est pré-entraîné sur des tâches comme la modélisation de langage masqué (MLM) et la prédiction de la phrase suivante (NSP).",
    "Le NLP (Natural Language Processing) est un domaine de l'intelligence artificielle qui se concentre sur l'interaction entre les ordinateurs et le langage humain. Il inclut des tâches comme l'analyse syntaxique, la traduction automatique, la reconnaissance d'entités nommées et la génération de texte.",
    "Les Transformers sont une architecture de réseaux de neurones introduite en 2017 qui utilise des mécanismes d'attention pour comprendre les relations entre les mots dans un texte. Leur architecture permet de traiter efficacement de grandes séquences de texte en parallèle, contrairement aux RNN.",
    "PyTorch est une bibliothèque open source de machine learning développée par Facebook. Elle est particulièrement populaire pour le deep learning et le NLP. PyTorch offre une grande flexibilité grâce à son mode dynamique (eager execution), ce qui facilite le débogage et la recherche.",
    "Un chatbot est un programme informatique qui simule une conversation humaine à travers du texte ou de la voix. Les chatbots modernes utilisent souvent des techniques de NLP et de deep learning. Ils peuvent être basés sur des règles, sur des modèles statistiques ou sur des réseaux de neurones comme BERT ou GPT.",
    "Le fine-tuning est une technique qui consiste à prendre un modèle pré-entraîné et à l'adapter à une tâche spécifique en utilisant un ensemble de données plus restreint. Cette approche permet d'obtenir de bonnes performances même avec peu de données annotées, car le modèle a déjà appris des représentations générales.",
    "GPT (Generative Pre-trained Transformer) est une famille de modèles de langage créée par OpenAI qui excelle dans la génération de texte. Contrairement à BERT, GPT est un modèle unidirectionnel (généralement de gauche à droite) et est pré-entraîné avec une tâche d'auto-régression.",
    "La similarité cosinus est une mesure utilisée pour déterminer à quel point deux vecteurs sont similaires. Elle est souvent utilisée pour comparer des représentations vectorielles de mots, de phrases ou de documents dans des systèmes de recherche ou de recommandation.",
    "Le dataset utilisé pour l'entraînement ou le fine-tuning dans le NLP peut contenir des intentions (intents), des exemples de phrases, ou des paires question-réponse. Une bonne qualité de dataset est cruciale pour les performances d'un chatbot ou d'un modèle de classification.",
    "Les embeddings de mots, comme ceux produits par Word2Vec, GloVe ou les couches initiales de BERT, transforment les mots en vecteurs numériques qui capturent leur signification sémantique. Ces vecteurs permettent aux modèles d'apprendre les relations entre les mots.",
    "L'attention multi-tête (multi-head attention) est un mécanisme clé dans les Transformers qui permet au modèle de se concentrer simultanément sur différentes positions du texte pour capter diverses relations contextuelles."
]

# Fusionner les bases de connaissances
#knowledge_base = knowledge_base + knowledge_voxens

# Salutations et réponses générales
greetings = {
    "bonjour": ["Bonjour! Comment puis-je vous aider aujourd'hui?", "Salut! Que puis-je faire pour vous?"],
    "salut": ["Salut! Comment puis-je vous aider?", "Bonjour! Quelle est votre question?"],
    "bonsoir": ["Bonsoir! En quoi puis-je vous être utile?", "Bonsoir! Comment puis-je vous aider?"],
    "au revoir": ["Au revoir! À bientôt!", "À la prochaine! N'hésitez pas à revenir si vous avez d'autres questions."],
    "merci": ["De rien!", "Je vous en prie!", "C'est un plaisir de vous aider."]
}

# Réponses pour les questions hors domaine
out_of_domain_responses = [
    "Je n'ai pas d'information sur ce sujet dans ma base de connaissances.",
    "Cette question semble en dehors de mon domaine d'expertise. Je peux vous aider sur des sujets concernant Voxens, le NLP, BERT ou les technologies associées.",
    "Je ne dispose pas des informations nécessaires pour répondre à cette question.",
    "Je suis désolé, mais je n'ai pas de données sur ce sujet dans ma base de connaissances actuelle."
]

# Messages en cas de faible confiance
low_confidence_responses = [
    "Je ne suis pas sûr de comprendre votre question. Pourriez-vous la reformuler?",
    "Je ne suis pas certain de ma réponse. Pourriez-vous être plus spécifique?",
    "Ma confiance est faible pour cette réponse. Essayons une autre approche ou question.",
    "Je n'ai pas assez d'informations pour répondre avec confiance à cette question."
]


def get_full_sentence(context, start, end):
    try:
        sentences = sent_tokenize(context)
        for sent in sentences:
            if context[start:end] in sent:
                return sent
    except Exception as e:
        print(f"Erreur lors de l'extraction de phrase: {e}")
    return context[start:end]  # fallback


# Fonction pour générer des embeddings de texte
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_encoder(**inputs)
    # Utiliser la moyenne de la dernière couche cachée comme embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()


# Fonction pour déterminer la pertinence d'une question par rapport à notre base de connaissances
def is_question_relevant(question):
    # Générer l'embedding de la question
    question_embedding = get_embedding(question)

    # Générer les embeddings pour chaque élément de la base de connaissances
    kb_embeddings = []
    for item in knowledge_base:
        kb_embeddings.append(get_embedding(item))

    # Calculer la similarité cosinus maximale
    max_similarity = 0
    for kb_embedding in kb_embeddings:
        similarity = cosine_similarity([question_embedding], [kb_embedding])[0][0]
        max_similarity = max(max_similarity, similarity)

    # Définir un seuil de pertinence (à ajuster selon les tests)
    relevance_threshold = 0.4
    return max_similarity >= relevance_threshold, max_similarity


# Fonction pour déterminer si une entrée est une salutation
def is_greeting(text):
    text_lower = text.lower()
    for greeting in greetings:
        if greeting in text_lower:
            return greeting
    return None


# Fonction pour vérifier si la réponse contient des informations pertinentes
def validate_answer(question, answer, score):
    # Seuil de confiance plus élevé
    CONFIDENCE_THRESHOLD = 0.2

    # Si le score est trop faible, on rejette la réponse
    if score < CONFIDENCE_THRESHOLD:
        return False, random.choice(low_confidence_responses)

    # Vérifier si la réponse est vide ou trop courte
    if not answer or len(answer.strip().split()) < 3:
        return False, random.choice(out_of_domain_responses)

    # Vérifier la pertinence de la question par rapport à notre base de connaissances
    is_relevant, similarity = is_question_relevant(question)
    if not is_relevant:
        return False, f"Cette question semble en dehors de mon domaine d'expertise (similarité: {similarity:.2f}). Je peux vous aider sur des sujets concernant Voxens, le NLP, BERT ou les technologies associées."

    return True, answer


# Fonction pour répondre aux questions en utilisant le pipeline QA
def answer_from_knowledge_base(question):
    # Concaténer toute la base de connaissances en un seul contexte
    context = " ".join(knowledge_base)

    try:
        # Utiliser le pipeline pour obtenir une réponse
        result = qa_pipeline(question=question, context=context)
        full_answer = get_full_sentence(context, result['start'], result['end'])

        # Valider la réponse
        is_valid, response = validate_answer(question, full_answer, result['score'])

        if is_valid:
            return response, result['score']
        else:
            return response, 0.0

    except Exception as e:
        print(f"Erreur lors de la recherche de réponse: {str(e)}")
        return random.choice(out_of_domain_responses), 0.0


# Fonction principale du chatbot
def chat():
    print(
        "BertBot: Bonjour! Je suis un chatbot basé sur BERT. Je peux répondre à vos questions sur Voxens, le NLP, BERT et d'autres sujets liés. (tapez 'quit' pour quitter)")

    while True:
        user_input = input("Vous: ")
        if user_input.lower() in ["quit", "exit", "bye", "quitter"]:
            print("BertBot: Au revoir!")
            break

        # Vérifier si c'est une salutation
        greeting_type = is_greeting(user_input)
        if greeting_type:
            response = random.choice(greetings[greeting_type])
            print(f"BertBot: {response}")
            continue

        # Utiliser le modèle QA pour répondre
        try:
            response, confidence = answer_from_knowledge_base(user_input)
            print(f"BertBot: {response}")
            # Pour le débogage, on peut afficher le score de confiance
            if confidence > 0:
                print(f"[Confiance: {confidence:.2f}]")
        except Exception as e:
            print(f"BertBot: Désolé, j'ai rencontré une erreur: {str(e)}")
            print("BertBot: Essayons avec une autre question!")


# Lancer le chatbot AVEC protection pour multiprocessing
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Ajoute le support pour les environnements Windows
    chat()