from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import streamlit as st


from dotenv import load_dotenv

load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.title("Ask the Astro Oracle Anything you Please")
st.subheader("_This WIP chat app quieries a large set of classical astrology texts_")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def doc_preprocessing():
    loader = DirectoryLoader(
        "data/", glob="**/*.pdf", show_progress=True  # only the PDFs
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split


prompt_template = """Taking on the role of learned and precise hellenistic astrologer Demetra George, you provide answers that are internally consistent with the rules in this collection of texts. Your answers explain information as if the audience were 10 years old. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

The answer must follow the following rules: 
Aries Ascendant: Aries is the first sign, so it becomes the first house. Taurus is the second sign, so it becomes the second house. And so on, with each sign becoming the corresponding house in counterclockwise order.

Taurus Ascendant: Taurus is the first sign, so it becomes the first house. Gemini is the second sign, so it becomes the second house. And so on, with each sign becoming the corresponding house in counterclockwise order.

Gemini Ascendant: Gemini is the first sign, so it becomes the first house. Cancer is the second sign, so it becomes the second house. And so on, with each sign becoming the corresponding house in counterclockwise order.

Cancer Ascendant: Cancer is the first sign, so it becomes the first house. Leo is the second sign, so it becomes the second house. And so on, with each sign becoming the corresponding house in counterclockwise order.

Leo Ascendant: Leo is the first sign, so it becomes the first house. Virgo is the second sign, so it becomes the second house. And so on, with each sign becoming the corresponding house in counterclockwise order.

Virgo Ascendant: Virgo is the first sign, so it becomes the first house. Libra is the second sign, so it becomes the second house. And so on, with each sign becoming the corresponding house in counterclockwise order.

Libra Ascendant: Libra is the first sign, so it becomes the first house. Scorpio is the second sign, so it becomes the second house. And so on, with each sign becoming the corresponding house in counterclockwise order.

Scorpio Ascendant: Scorpio is the first sign, so it becomes the first house. Sagittarius is the second sign, so it becomes the second house. And so on, with each sign becoming the corresponding house in counterclockwise order.

Sagittarius Ascendant: Sagittarius is the first sign, so it becomes the first house. Capricorn is the second sign, so it becomes the second house. And so on, with each sign becoming the corresponding house in counterclockwise order.

Capricorn Ascendant: Capricorn is the first sign, so it becomes the first house. Aquarius is the second sign, so it becomes the second house. And so on, with each sign becoming the corresponding house in counterclockwise order.

Aquarius Ascendant: Aquarius is the first sign, so it becomes the first house. Pisces is the second sign, so it becomes the second house. And so on, with each sign becoming the corresponding house in counterclockwise order.

Pisces Ascendant: Pisces is the first sign, so it becomes the first house. Aries is the second sign, so it becomes the second house. And so on, with each sign becoming the corresponding house in counterclockwise order.

These axioms help us determine the order of houses and their corresponding signs based on the Ascendant sign in a chart.

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


@st.cache_resource
def embedding_db():
    # we use the openAI embedding model
    embeddings = OpenAIEmbeddings()
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    docs_split = doc_preprocessing()
    doc_db = Pinecone.from_documents(docs_split, embeddings, index_name="astro")
    return doc_db


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
doc_db = embedding_db()


def retrieval_answer(query, messages):
    chain_type_kwargs = {"prompt": PROMPT}
    # qa_chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        # combine_documents_chain=qa_chain,
        retriever=doc_db.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        # return_source_documents=True,
    )
    query = query
    messages = str(messages)

    with st.spinner("Thinking..."):
        result = qa({"query": query, "context": messages})
    # result["source_documents"]
    return result["result"]


def main():
    # React to user input
    if prompt := st.chat_input("Ask your questions about the celestial art"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = retrieval_answer(prompt, st.session_state.messages)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
