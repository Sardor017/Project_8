import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from io import BytesIO
from reportlab.pdfgen import canvas
# from streamlit_extras.colored_header import colored_header
# from streamlit_extras.card import card


# Define constants and configuration
pdf_path = r"interpretation-of-full-blood-count-parameters-in-health-and-disease.pdf"
# Function to process the PDF and create a retriever

def create_pdf(content):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(50, 800, "Chatbot Response")
    text = pdf.beginText(50, 780)
    text.setFont("Helvetica", 12)
    text.setLeading(14)
    for line in content.split('\n'):
        text.textLine(line)
    pdf.drawText(text)
    pdf.save()
    buffer.seek(0)
    return buffer

@st.cache_resource
def initialize_retriever(api_key):
    # Step 1: Extract text and tables from the PDF
    documents = []  # List to store the extracted content as text
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            # Extract text
            text = page.extract_text()
            if text:
                documents.append(f"Page {page_number} Text:\n{text}")

            # Extract tables
            tables = page.extract_tables()
            for table_index, table in enumerate(tables, start=1):
                cleaned_table = [[cell if cell is not None else "" for cell in row] for row in table]
                table_content = "\n".join([" | ".join(row) for row in cleaned_table])
                documents.append(f"Page {page_number} Table {table_index}:\n{table_content}")

    combined_content = "\n\n".join(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2100, chunk_overlap=50)
    chunks = text_splitter.split_text(combined_content)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("faiss_index")
    # st.write("Retriever successfully created and saved locally.")

    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store.as_retriever()

@st.cache_resource
def initialize_chatbot(api_key):
    system_prompt = ("""
    You are a medical assistant chatbot. Your task is to:
    1. Analyze the blood test results provided by the user.
    2. Provide expected diagnoses and recommendations for consultation with a doctor.
    3. Answer additional questions based on the user's input.
    
    Important:
    - Be cautious and provide clear disclaimers that you are not a substitute for a doctor.
    - Avoid making definitive diagnoses; only suggest possibilities based on test results.
    - Use both table data and contextual sentences from the provided document.
    - Be empathetic and professional.
    - Analyze and answer only according to the document provided.
    - Answer questions using both table data and contextual sentences from the document.
    
    Context: {context}
    """)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=api_key,
        temperature=0,
        max_tokens=1000
    )

    retriever = initialize_retriever(api_key)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(
        retriever,
        question_answer_chain)

    return qa_chain


page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://assets.medpagetoday.net/media/images/106xxx/106690_m.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stSidebar"] {
    background-color: rgba(240, 240, 240, 0.8);
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Streamlit UI
st.title("Medical Assistant Chatbot")

# Step 1: User enters OpenAI API key
# Sidebar for OpenAI API key input
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
if not api_key:
    st.sidebar.warning("Please enter your OpenAI API key to continue.")
else:
    # Initialize chatbot with the provided API key
    qa_chain = initialize_chatbot(api_key)

    st.write("Hello! I'm here to assist you with analyzing blood test results. Please provide the necessary information below.")

    gender = st.selectbox("What is your gender?", ["Male", "Female", "Other"])
    age = st.number_input("What is your age?", min_value=0, max_value=120, step=1)

    # Add inputs for blood test parameters
    st.subheader("Enter Your Blood Test Results (Optional):")
    wbc = st.text_input("White Blood Cells (WBCs) (x10^9/L):")
    rbc = st.text_input("Red Blood Cells (RBC) (x10^12/L):")
    hct = st.text_input("Haematocrit (Hct) (%):")
    hgb = st.text_input("Haemoglobin (Hgb) (g/dL):")
    mcv = st.text_input("Mean Corpuscular Volume (MCV) (fL):")
    mch = st.text_input("Mean Corpuscular Haemoglobin (MCH) (pg):")
    mchc = st.text_input("Mean Corpuscular Hemoglobin Concentration (MCHC) (g/dL):")
    rdw = st.text_input("Red Cell Distribution Width (RDW) (%):")
    plt = st.text_input("Platelets (PLT) (x10^9/L):")
    pdw = st.text_input("Platelet Distribution Width (PDW):")
    mpv = st.text_input("Mean Platelet Volume (MPV) (fL):")

    if st.button("Analyze Blood Test Results"):
        # Dynamically construct input for RAG, skipping empty fields
        blood_test_results = []
        if wbc: blood_test_results.append(f"WBC: {wbc}")
        if rbc: blood_test_results.append(f"RBC: {rbc}")
        if hct: blood_test_results.append(f"Hct: {hct}")
        if hgb: blood_test_results.append(f"Hgb: {hgb}")
        if mcv: blood_test_results.append(f"MCV: {mcv}")
        if mch: blood_test_results.append(f"MCH: {mch}")
        if mchc: blood_test_results.append(f"MCHC: {mchc}")
        if rdw: blood_test_results.append(f"RDW: {rdw}")
        if plt: blood_test_results.append(f"PLT: {plt}")
        if pdw: blood_test_results.append(f"PDW: {pdw}")
        if mpv: blood_test_results.append(f"MPV: {mpv}")

        blood_test_results_str = ", ".join(blood_test_results)

        input_data = f"Patient details: Gender: {gender}, Age: {age}. Blood test results: {blood_test_results_str}. Please analyze and provide suggestions."
        with st.spinner("Processing your blood test results..."):
            response = qa_chain.invoke({"input": input_data})
        # Save analysis results in session state
        st.session_state['analysis_results'] = response['answer']

    # Display Analysis Results if available
    if 'analysis_results' in st.session_state:
        st.subheader("Analysis Results:")
        st.write(st.session_state['analysis_results'])
        st.subheader("Sources of Information:")
        st.write('Haematology International Journal')
        st.info("Based on the analysis, I recommend consulting a doctor for a thorough review.")
        if st.button("Download Response as PDF"):
            pdf_file = create_pdf(st.session_state['analysis_results'])
            st.download_button(
                label="Click to Download PDF",
                data=pdf_file,
                file_name="response.pdf",
                mime="application/pdf"
            )

    # Initialize conversation history if not already present
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []

    # Display conversation history and dynamic question fields
    st.subheader("Ask Any Follow-up Questions:")

    for idx, qa in enumerate(st.session_state['conversation_history']):
        st.markdown(f"**Question {idx + 1}:** {qa['question']}")
        st.markdown(f"**Answer {idx + 1}:** {qa['answer']}")
        st.divider()

    # Add a new question field
    st.markdown("### New Question")
    user_query = st.text_area(f"Enter Question {len(st.session_state['conversation_history']) + 1} here:")

    # Submit button to add the new question and get the response
    if st.button("Submit New Question"):
        if user_query.strip() == "":
            st.warning("Please enter a question before submitting.")
        else:
            with st.spinner("Processing your question..."):
                # Invoke the chatbot to get the answer
                followup_response = qa_chain.invoke({"input": user_query})

            # Append the new question and answer to the conversation history
            st.session_state['conversation_history'].append({
                "question": user_query,
                "answer": followup_response['answer']
            })

            # Clear the input field for the next question
            st.experimental_rerun()  # Refresh the app to display the new question field dynamically




















    # After analysis, show a question field
    # st.subheader("Ask Any Follow-up Questions:")
    # user_query = st.text_area("Enter your question here:")
    # if st.button("Submit Follow-up Question"):
    #     if user_query.strip() == "":
    #         st.warning("Please enter a question before submitting.")
    #     else:
    #         with st.spinner("Processing your question..."):
    #             followup_response = qa_chain.invoke({"input": user_query})
    #         st.subheader("Assistant's Answer:")
    #         st.write(followup_response['answer'])

            

# # Title and Introduction
# st.markdown(
#     "<h1 style='text-align: center; color: #4CAF50;'>Medical Assistant Chatbot</h1>",
#     unsafe_allow_html=True,
# )
# st.markdown(
#     "<p style='text-align: center; color: #666;'>Your friendly assistant for blood test analysis and medical queries.</p>",
#     unsafe_allow_html=True,
# )
# st.divider()

# # Sidebar for API Key
# with st.sidebar:
#     st.markdown(
#         "<h3 style='color: #4CAF50;'>Settings</h3>",
#         unsafe_allow_html=True,
#     )
#     api_key = st.text_input("Enter your OpenAI API Key:", type="password")
#     st.info(
#         "⚠️ Your API key is required to use the chatbot. It will not be stored anywhere."
#     )

# # Main Section
# if not api_key:
#     st.warning("Please enter your OpenAI API key in the sidebar to continue.")
# else:
#     # Welcome Message
#     st.markdown("<h3 style='color: #4CAF50;'>👋 Hello!</h3>", unsafe_allow_html=True)
#     st.markdown(
#         "<p style='color: #666;'>I'm here to assist you with blood test analysis and answer your medical questions.</p>",
#         unsafe_allow_html=True,
#     )
#     st.divider()

#     # Gender and Age Inputs
#     gender, age = st.columns(2)
#     with gender:
#         gender_value = st.selectbox("Gender", ["Male", "Female", "Other"])
#     with age:
#         age_value = st.number_input("Age", min_value=0, max_value=120, step=1)

#     st.markdown("<h4 style='color: #4CAF50;'>Enter Blood Test Results (Optional)</h4>", unsafe_allow_html=True)
#     st.info("💡 Tip: Leave fields empty if you don't have all the data.")
    
#     # Blood Test Inputs
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         wbc = st.text_input("White Blood Cells (WBC):")
#         plt = st.text_input("Platelets (PLT):")
#         mch = st.text_input("Mean Corpuscular Haemoglobin (MCH):")
#         rdw = st.text_input("Red Cell Distribution Width (RDW):")
#     with col2:
#         rbc = st.text_input("Red Blood Cells (RBC):")
#         mpv = st.text_input("Mean Platelet Volume (MPV):")
#         mcv = st.text_input("Mean Corpuscular Volume (MCV):")
#         mchc = st.text_input("Mean Corpuscular Haemoglobin Concentration (MCHC):")
        
#     with col3:
#         hgb = st.text_input("Haemoglobin (Hgb):")
#         hct = st.text_input("Haematocrit (Hct):")
#         pdw = st.text_input("Platelet Distribution Width:")
        

#     st.divider()

#     # Analysis Button
#     if st.button("🔍 Analyze Blood Test Results"):
#         st.markdown(
#             "<h4 style='color: #4CAF50;'>Analysis Results</h4>",
#             unsafe_allow_html=True,
#         )
#         with st.spinner("Analyzing blood test results..."):
#             # Example Response (replace with real analysis code)
#             st.success("Your blood test results have been analyzed successfully!")
#             st.info(
#                 "Based on the results, consult a doctor for further evaluation. I'm not a substitute for professional advice."
#             )

#         st.divider()
#         if st.button("Download Analysis as PDF"):
#             st.download_button("📥 Download PDF", data="PDF_CONTENT_HERE", file_name="analysis.pdf")

#     st.divider()

#     # Ask Follow-up Questions
#     st.markdown("<h4 style='color: #4CAF50;'>Ask a Follow-up Question</h4>", unsafe_allow_html=True)
#     question = st.text_area("Type your question here")
#     if st.button("Submit Question"):
#         with st.spinner("Thinking..."):
#             # Example Response (replace with real chatbot code)
#             st.success("Here's the answer to your question.")
#             st.info(
#                 "💡 Remember, this is based on the document and not a substitute for professional advice."
#             )

#     st.divider()

#     # Conversation History
#     st.markdown("<h4 style='color: #4CAF50;'>Conversation History</h4>", unsafe_allow_html=True)
#     st.write("1. User: What does my RBC mean?\n   Chatbot: RBC refers to...")
#     st.write("2. User: Should I be concerned about HGB levels?\n   Chatbot: HGB indicates...")

