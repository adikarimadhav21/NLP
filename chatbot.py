"""

Course Name: CMPS-5443-201 AdvTopCS:NaturalLangProc
Programmers:
Name: Madhav Adhikari
Name: Neeraj Chandragiri
Name: Nirupavardhan Lingareddygari

Description :
This  program uses the ChatterBot library to create a chatbot with various logic adapters.
It includes the BestMatch adapter for finding the most appropriate response, the TimeLogicAdapter for
handling time-related queries, and the MathematicalEvaluation adapter for evaluating mathematical expressions.

The chatbot is trained on custom responses and dialogues from a CSV file.
Custom responses cover greetings, inquiries about well-being, location, and farewells.
The CSV file contains dialogues for further training the chatbot on various conversation topics.

The program provides a command-line interface for users to interact with the chatbot.
Users can input statements, and the chatbot responds accordingly based on its training and logic adapters.
The chatbot continues to respond until the user enters an exit command(":q", "quit", "exit").
"""


from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.comparisons import LevenshteinDistance
from chatterbot.response_selection import get_first_response
from chatterbot.logic import TimeLogicAdapter, MathematicalEvaluation
import pandas as pd
import re

# Create a new instance of a ChatBot with all configurations
chatbot = ChatBot(
    "Chatbot_program",
    logic_adapters=[
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'I am sorry, but I do not understand.',
            'maximum_similarity_threshold': 0.90,
            "statement_comparison_function": LevenshteinDistance,
            "response_selection_method": get_first_response
        },
        {
            'import_path': 'chatterbot.logic.TimeLogicAdapter'
        },
        {
            'import_path': 'chatterbot.logic.MathematicalEvaluation'
        }
    ])

exit_conditions = (":q", "quit", "exit")

# Define custom responses
custom_responses = [
    "Hi",
    "Hello",
    "How are you ?",
    "Fine ! what about you ?",
    "Where are you now ?",
    "At NLP class ",
    "Why was the math book sad?",
    "Because it had too many problems.",
    "Bye",
    "Have a good one !bye "
]

# Train the chatbot on custom responses
list_trainer = ListTrainer(chatbot)
list_trainer.train(custom_responses)

# Train the chatbot on dialogues from the input file
df = pd.read_csv("daily_dialog_data.csv", header=None)
dialogs = df[0].tolist()
list_trainer.train(dialogs)

while True:
    query = input("You::")
    if query in exit_conditions:
        break
    else:
         # Preprocess the input statement to remove non-numeric characters and words
       # query = re.sub(r'[^\d\s]', '', query)
        print(f"🤖 {chatbot.get_response(query)}")
