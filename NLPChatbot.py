from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

class NLPChatbot:

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = SVC(kernel='linear', probability=True)

    def train(self, texts: list[str], labels: list[int]):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def predict(self, user_input: str) -> str:
        X = self.vectorizer.transform([user_input])
        pred = self.model.predict(X)[0]
        prob = self.model.predict_proba(X).max()
        return f"Classe: {pred}, Confiance: {prob:.2f}"

    def save(self, path: str):
        joblib.dump({'vectorizer': self.vectorizer, 'model': self.model}, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.vectorizer = data['vectorizer']
        self.model = data['model']


def test_chatbot3():
    texts = [
        "Je veux parier sur PSG vs OM",
        "Quels conseils pour un pari sur Liverpool?",
        "Faut-il éviter ce pari?",
        "Le Real a-t-il une chance?"
    ]
    # Labels : 0 (éviter pari), 1 (pari modéré), 2 (pari recommandé)
    labels = [2, 2, 0, 1]

    bot3 = NLPChatbot()
    bot3.train(texts, labels)

    # Test prédictions
    print(bot3.predict("Je souhaite un conseil pour pari Real vs Barça"))
    print(bot3.predict("Est-ce un bon pari?"))

    # Optionnel : sauvegarde du modèle
    # bot3.save('chatbot3_model.joblib')

if __name__ == "__main__":
    test_chatbot3()
