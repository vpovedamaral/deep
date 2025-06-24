import os
import openai

class LLMChatbotGPT:

    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        # Récupération de la clé API depuis la variable d'environnement si non fournie
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Clé API OpenAI non trouvée. Définissez OPENAI_API_KEY.")
        openai.api_key = self.api_key
        self.model = model

    def get_response(self, user_input: str) -> str:

        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Vous êtes un assistant spécialiste des paris sportifs."},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            max_tokens=150,
        )
        return completion.choices[0].message.content.strip()


def main():
    try:
        bot = LLMChatbotGPT()
    except ValueError as e:
        print(e)
        return

    print("=== Chatbot Paris Sportifs (GPT) ===")
    print("Tapez 'quit' ou 'exit' pour arrêter.")
    while True:
        user_input = input("Vous: ")
        if user_input.lower() in {"quit", "exit"}:
            print("Au revoir !")
            break
        try:
            response = bot.get_response(user_input)
            print(f"Bot: {response}\n")
        except Exception as err:
            print(f"Erreur lors de la génération de réponse: {err}")
            break

if __name__ == "__main__":
    main()
