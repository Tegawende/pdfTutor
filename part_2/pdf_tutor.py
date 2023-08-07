import streamlit as st


from llm_functions import load_data, split_text, initialize_llm, generate_questions, create_retrieval_qa_chain

# Initialization of session states
# Since Streamlit always reruns the script when a widget changes, we need to initialize the session states
if 'questions' not in st.session_state:
    st.session_state['questions'] = 'empty'
    st.session_state['seperated_question_list'] = 'empty'
    st.session_state['questions_to_answers'] = 'empty'
    st.session_state['submitted'] = 'empty'

with st.container():
    st.markdown("""# PDF Tutor""")

# Get user's OpenAI API Key
openai_api_key = st.text_input(label="Clé API OpenAI ",  value="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", key="openai_api_key_input")


temp = st.slider('Température (Créativé)', 0.0, 2.0, step=0.1,)


# Let user upload a file
uploaded_file = st.file_uploader("Chargez votre document pdf", type=['pdf'])

if uploaded_file is not None:
    # Check whether user entered an API key
    if not openai_api_key:
        st.error("Veuillez saisir votre clé API OpenAI")
    else:
        # Load data from PDF
        text_from_pdf = load_data(uploaded_file)

        # Split text for question generation
        documents_for_question_gen = split_text(text_from_pdf, chunk_size=10000, chunk_overlap=200)

        # Split text for question answering
        documents_for_question_answering = split_text(text_from_pdf, chunk_size=500, chunk_overlap=200)

        # Initialize large language model for question generation
        llm_question_gen = initialize_llm(openai_api_key=openai_api_key, model="gpt-3.5-turbo-16k", temperature=0.4)

        # Initialize large language model for question answering
        llm_question_answering = initialize_llm(openai_api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0.1)

        # Create questions if they have not yet been generated
        if st.session_state['questions'] == 'empty':
            with st.spinner("Génération des questions..."):
                # Assign the generated questions to the session state. This way, the questions are only generated once.
                st.session_state['questions'] = generate_questions(llm=llm_question_gen, chain_type="refine", documents=documents_for_question_gen)

        if st.session_state['questions'] != 'empty':
            # Show questions on screen. You could use st.code for easy copy-pasting.
            st.info(st.session_state['questions'])

            # Split questions into a list
            st.session_state['questions_list'] = st.session_state['questions'].split('\n')

            with st.form(key='my_form'):
                # Create a list of questions that have to be answered
                st.session_state['questions_to_answers'] = st.multiselect(label="Sélectionner les questions auxquelles répondre", placeholder="Choisir", options=st.session_state['questions_list'])
                submitted = st.form_submit_button('Générer les réponses')
                if submitted:
                    st.session_state['submitted'] = True

            if st.session_state['submitted']:
            # Initialize the Retrieval QA Chain
                with st.spinner("Génération des réponses..."):
                    generate_answer_chain = create_retrieval_qa_chain(openai_api_key=openai_api_key, documents=documents_for_question_answering, llm=llm_question_answering)
                    # For each question, generate an answer
                    for question in st.session_state['questions_to_answers']:
                        # Generate answer
                        answer = generate_answer_chain.run(question)
                        # Show answer on screen
                        st.write(f"Question: {question}")
                        st.info(f"Réponse: {answer}")