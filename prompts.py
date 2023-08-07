from langchain.prompts import PromptTemplate

prompt_template_questions = """
Vous êtes un expert dans la création de questions d'entraînement basées un document.
Votre objectif est de préparer un étudiant à son examen. Pour ce faire, vous posez des questions sur le texte ci-dessous :

------------
{text}
------------

Créez des questions qui prépareront l'étudiant à son examen. Veillez à ne pas perdre d'informations importantes.

QUESTIONS:
"""
PROMPT_QUESTIONS = PromptTemplate(template=prompt_template_questions, input_variables=["text"])

refine_template_questions = ("""
Vous êtes un expert dans la création de questions d'entraînement basées sur un document.
Votre objectif est d'aider un étudiant à se préparer à un examen.
Nous avons reçu les questions d'entraînement suivantes :  {existing_answer}.
Nous avons la possibilité d'affiner les questions existantes ou d'en ajouter de nouvelles.
(seulement si nécessaire) avec un peu plus de contexte ci-dessous.
------------
{text}
------------

Compte tenu du nouveau contexte, affinez les questions originales.
Si le contexte n'est pas utile, veuillez fournir les questions originales.
QUESTIONS:
"""
)
REFINE_PROMPT_QUESTIONS = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template_questions,
)