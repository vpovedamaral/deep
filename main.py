from LLMChatbotGPT import LLMChatbotGPT
from DeepLearningChatbot import test_chatbot2
from NLPChatbot import test_chatbot3


def run_llm_chatbot():
    """
    Lance une session de chat interactive avec le chatbot GPT (paris sportifs).
    """
    bot = LLMChatbotGPT()
    print("=== LLM Chatbot (Paris Sportifs) ===")
    print("Tapez 'quit' ou 'exit' pour arrêter.")
    while True:
        user_input = input("Vous: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Au revoir !")
            break
        try:
            response = bot.get_response(user_input)
            print(f"Bot: {response}\n")
        except Exception as e:
            print(f"Erreur: {e}")
            break


def main():
    """
    Menu de sélection pour tester chaque chatbot.
    """
    print("Sélectionnez le chatbot à tester:")
    print("1 - LLMChatbotGPT (Deep Learning + NLP via GPT)")
    print("2 - DeepLearningChatbot (Pur Deep Learning)")
    print("3 - NLPChatbot (Pur NLP)")
    choice = input("Votre choix (1/2/3): ")

    if choice == "1":
        run_llm_chatbot()
    elif choice == "2":
        print("=== Test DeepLearningChatbot (Pur Deep Learning) ===")
        test_chatbot2()
    elif choice == "3":
        print("=== Test NLPChatbot (Pur NLP) ===")
        test_chatbot3()
    else:
        print("Choix invalide. Veuillez relancer le script et choisir 1, 2 ou 3.")


if __name__ == "__main__":
    main()
