import validators, streamlit as st, re
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi

def fetch_youtube_transcript(video_url):
    match = re.search(r"v=([a-zA-Z0-9_-]+)", video_url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    video_id = match.group(1)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([item["text"] for item in transcript])
    return [Document(page_content=text)]

st.set_page_config(page_title="LangChain Summarizer", page_icon="🦜")
st.title("🦜 Groq + LangChain Summarizer")

with st.sidebar:
    groq_api_key = st.text_input("🚀 Groq API Key", type="password")

generic_url = st.text_input("URL (YouTube or Website)")

# ✅ Updated model used here
llm = ChatGroq(model_name="gemma2-9b-it", groq_api_key=groq_api_key)

prompt = PromptTemplate(
    template="Provide a summary of the following content in ~300 words:\n{text}",
    input_variables=["text"]
)

if st.button("Summarize"):
    if not groq_api_key or not generic_url:
        st.error("Please enter both your API key and a valid URL.")
    elif not validators.url(generic_url):
        st.error("Invalid URL entered.")
    else:
        try:
            with st.spinner("Processing…"):
                if "youtube.com" in generic_url:
                    try:
                        docs = fetch_youtube_transcript(generic_url)
                    except Exception as e:
                        st.error("⚠️ Could not fetch YouTube transcript.")
                        st.exception(e)
                        st.stop()
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    docs = loader.load()

                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                summary = chain.run(docs)
                st.success("✅ Summary:")
                st.write(summary)

        except Exception as e:
            st.error("❌ Error during summarization.")
            st.exception(e)
