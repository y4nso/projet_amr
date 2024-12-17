from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import string
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import torch

app = Flask(__name__)
CORS(app)

knowledge_base = {
    "énergie": """Pour réduire votre consommation d’énergie, vous pouvez :
    - Diminuer la luminosité de votre écran.
    - Activer la mise en veille automatique.
    - Éteindre vos appareils lorsque non utilisés.
    - Privilégier des appareils labellisés "Energy Star".
    - Utiliser un mode sombre ou des thèmes à faible éclairage.
    """,

    "sécurité": """Pour protéger vos données personnelles :
    - Utilisez des mots de passe forts et uniques.
    - Activez l’authentification à deux facteurs.
    - Mettez à jour régulièrement vos logiciels.
    - Méfiez-vous des liens suspects et du phishing.
    - Utilisez un antivirus et un pare-feu.
    """,

    "environnement": """L’impact environnemental du numérique comprend :
    - Extraction de métaux rares pour les composants.
    - Forte consommation d’énergie des data centers.
    - Émissions de gaz à effet de serre liées au cycle de vie des appareils.
    - Consommation d’eau pour la fabrication des composants.
    """,

    "déchets": """Réduire les déchets électroniques :
    - Réparez ou mettez à niveau vos appareils plutôt que de les jeter.
    - Donnez ou vendez les appareils que vous n’utilisez plus.
    - Déposez les appareils en fin de vie dans un centre de recyclage.
    - Privilégiez des appareils modulaires et durables.
    """,

    "équilibre": """Pour une utilisation saine et équilibrée des technologies :
    - Définissez des plages horaires sans écran.
    - Faites des pauses régulières, toutes les 30-45 minutes.
    - Utilisez des applications de gestion du temps d’écran.
    - Pratiquez des activités hors-ligne (sport, lecture, jardinage).
    """,

    "éthique": """Aspects éthiques du numérique :
    - Respecter la vie privée et les données des utilisateurs.
    - Favoriser des chaînes d’approvisionnement responsables et équitables.
    - Lutter contre l’obsolescence programmée.
    - Encourager le partage équitable des bénéfices des technologies.
    """,

    "logiciels_libres": """Avantages des logiciels libres et open source :
    - Transparence et auditabilité du code.
    - Communauté active et réactive.
    - Respect de la vie privée et des standards ouverts.
    - Possibilité de modifier et d’adapter le logiciel à ses besoins.
    """,

    "accessibilité": """Pour une informatique plus inclusive et accessible :
    - Choisir des sites et des logiciels compatibles avec les lecteurs d’écran.
    - Fournir des sous-titres, transcriptions et descriptions visuelles.
    - Adapter les contrastes et la taille du texte.
    - Tester sur différents terminaux et équipements spécialisés.
    """,

    "sobriété": """Pour une sobriété numérique au quotidien :
    - Limiter le streaming vidéo en ultra-HD si non nécessaire.
    - Éviter l’envoi massif de mails non essentiels.
    - Préférer des sites et services légers (moins gourmands en bande passante).
    - Acheter des appareils reconditionnés ou de seconde main.
    """,

    "responsabilité": """Responsabilité sociale des acteurs du numérique :
    - Assurer des conditions de travail dignes dans la production de matériel.
    - Soutenir des initiatives de formation et d’éducation numérique.
    - Participer à des projets de coopération numérique internationale.
    - Contribuer à réduire la fracture numérique.
    """
}

categories = list(knowledge_base.keys())
lemmatizer = WordNetLemmatizer()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
cat_embeddings = model.encode(categories, convert_to_tensor=True)

def preprocess_user_input(user_input):
    user_input = user_input.lower()
    user_input = user_input.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(user_input)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)

def generate_response(user_input):
    text = preprocess_user_input(user_input)
    query_embedding = model.encode(text, convert_to_tensor=True)
    similarities = torch.nn.functional.cosine_similarity(query_embedding, cat_embeddings)
    best_idx = torch.argmax(similarities).item()
    best_category = categories[best_idx]
    best_score = similarities[best_idx].item()

    if best_score < 0.2:
        return "Je suis désolé, je ne suis pas sûr de pouvoir répondre à cette question. Pouvez-vous préciser votre demande ?"
    else:
        return knowledge_base[best_category]

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_question = data.get("question", "")
    answer = generate_response(user_question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
