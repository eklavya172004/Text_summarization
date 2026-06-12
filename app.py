import streamlit as st
import validators

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    YoutubeLoader,
    UnstructuredURLLoader,
)

# ----------------------------------
# Streamlit Config
# ----------------------------------

st.set_page_config(
    page_title="AI Content Summarizer",
    page_icon="🦜",
    layout="wide"
)

st.title("🦜 AI Content Summarizer")
st.write("Summarize YouTube videos and websites using Groq + LangChain")

# ----------------------------------
# Sidebar
# ----------------------------------

with st.sidebar:
    st.header("Settings")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password"
    )

    model_name = st.selectbox(
        "Model",
        [
            "gemma2-9b-it",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant"
        ]
    )

# ----------------------------------
# Main Input
# ----------------------------------

url = st.text_input(
    "Enter Website or YouTube URL"
)

# ----------------------------------
# Prompt
# ----------------------------------

prompt = PromptTemplate.from_template(
    """
You are an expert content summarizer.

Summarize the following content in approximately 300 words.

Focus on:
- Main ideas
- Important insights
- Key conclusions

Content:
{text}
"""
)

# ----------------------------------
# Summarize Button
# ----------------------------------

if st.button("Generate Summary"):

    if not groq_api_key:
        st.error("Please enter your Groq API Key")
        st.stop()

    if not url:
        st.error("Please enter a URL")
        st.stop()

    if not validators.url(url):
        st.error("Invalid URL")
        st.stop()

    try:

        with st.spinner("Loading content..."):

            # -------------------------
            # Load Data
            # -------------------------

            # NEW (working)
            if "youtube.com" in url or "youtu.be" in url:
                loader = YoutubeLoader.from_youtube_url(
                    url,
                    add_video_info=False   # <-- disables pytube metadata fetch (the broken part)
                )

            else:

                loader = UnstructuredURLLoader(
                    urls=[url],
                    ssl_verify=False,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 "
                            "(Windows NT 10.0; Win64; x64)"
                        )
                    }
                )

            docs = loader.load()

            if not docs:
                st.error("No content found.")
                st.stop()

            # -------------------------
            # Chunk Large Documents
            # -------------------------

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=4000,
                chunk_overlap=300
            )

            split_docs = splitter.split_documents(docs)

            content = "\n\n".join(
                doc.page_content
                for doc in split_docs
            )

        # -------------------------
        # LLM
        # -------------------------

        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model=model_name,
            temperature=0.2
        )

        chain = (
            prompt
            | llm
            | StrOutputParser()
        )

        with st.spinner("Generating summary..."):

            summary = chain.invoke({
                "text": content[:25000]
            })

        st.success("Summary Generated!")

        st.subheader("Summary")
        st.write(summary)

        st.divider()

        st.subheader("Document Stats")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Documents Loaded",
                len(docs)
            )

        with col2:
            st.metric(
                "Characters",
                len(content)
            )

    except Exception as e:
        st.exception(e)