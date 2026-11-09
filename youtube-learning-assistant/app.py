
import os
import re
import time
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# =====================================================
# 1. CONFIGURATION
# =====================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env", override=True)

API_KEY = os.getenv("OPENROUTER_API_KEY")

st.set_page_config(
    page_title="YouTube Learning Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 YouTube Learning Assistant")
st.caption(
    "Learn from YouTube videos using AI-powered "
    "transcripts, interview questions and RAG."
)

if not API_KEY:
    st.error("OPENROUTER_API_KEY missing. Check your .env file.")
    st.stop()


# =====================================================
# 2. INITIALIZE MODELS
# =====================================================

@st.cache_resource
def get_models():

    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_tokens=2000
    )

    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    return llm, embeddings


llm, embeddings = get_models()


# =====================================================
# 3. EXTRACT YOUTUBE VIDEO ID
# =====================================================

def extract_video_id(video_url):

    parsed = urlparse(video_url.strip())
    host = (parsed.hostname or "").lower()

    video_id = None

    if host in ["youtu.be", "www.youtu.be"]:

        video_id = parsed.path.lstrip("/").split("/")[0]

    elif host in [
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com"
    ]:

        if parsed.path == "/watch":

            video_id = parse_qs(
                parsed.query
            ).get("v", [None])[0]

        elif parsed.path.startswith(
            ("/shorts/", "/embed/", "/live/")
        ):

            video_id = parsed.path.split("/")[2]

    if not video_id or not re.fullmatch(
        r"[A-Za-z0-9_-]{11}", video_id
    ):
        raise ValueError("Invalid YouTube video URL.")

    return video_id


# =====================================================
# 4. FETCH TRANSCRIPT
# =====================================================

def fetch_transcript(video_id):

    api = YouTubeTranscriptApi()

    transcript_list = api.list(video_id)

    try:

        selected = transcript_list.find_transcript(
            ["hi", "en"]
        )

    except NoTranscriptFound:

        selected = next(iter(transcript_list), None)

        if selected is None:
            raise NoTranscriptFound(
                video_id, ["hi", "en"]
            )

    fetched = selected.fetch()

    transcript = " ".join(
        snippet.text for snippet in fetched
    )

    return transcript, selected.language_code


# =====================================================
# 5. SPLIT TRANSCRIPT
# =====================================================

def split_transcript(transcript):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.create_documents([transcript])


# =====================================================
# 6. PARALLEL TRANSLATION
# =====================================================

def translate_one(chunk, translation_llm):

    prompt = f"""
    Translate the following transcript into English.

    Instructions:
    1. Translate the complete content.
    2. Do not summarize or omit anything.
    3. Preserve technical terminology.
    4. Maintain the original meaning and sequence.
    5. Return only translated text.

    Transcript:
    {chunk.page_content}
    """

    for attempt in range(3):

        try:

            response = translation_llm.invoke(prompt)

            return response.content

        except Exception:

            if attempt == 2:
                raise

            time.sleep(2 ** attempt)


def translate_transcript(chunks, workers=8):

    translation_llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=2000
    )

    translated = [None] * len(chunks)

    progress = st.progress(0)
    status = st.empty()

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = {
            executor.submit(
                translate_one,
                chunk,
                translation_llm
            ): i
            for i, chunk in enumerate(chunks)
        }

        completed = 0

        for future in as_completed(futures):

            index = futures[future]

            translated[index] = future.result()

            completed += 1

            progress.progress(
                completed / len(chunks)
            )

            status.write(
                f"Translated {completed}/{len(chunks)} chunks"
            )

    progress.empty()
    status.empty()

    return [
        Document(page_content=text)
        for text in translated
    ]


# =====================================================
# 7. GENERATE INTERVIEW QUESTIONS
# =====================================================

def generate_questions(transcript, language):

    prompt = f"""
    You are an experienced technical interviewer.

    Analyze the following YouTube video transcript.

    Generate exactly 10 potential interview questions.

    Instructions:

    1. Use only concepts covered in the transcript.
    2. Include conceptual and technical questions.
    3. Include definitions, working principles and
       applications where relevant.
    4. Avoid duplicate questions.
    5. Arrange questions from basic to advanced.
    6. Do not invent topics.
    7. Write questions in {language}.

    Transcript:

    {transcript}

    Return exactly 10 numbered questions.
    """

    response = llm.invoke(prompt)

    return response.content


# =====================================================
# 8. BUILD VECTOR DATABASE
# =====================================================

def build_vector_store(documents):

    return FAISS.from_documents(
        documents,
        embeddings
    )


# =====================================================
# 9. RAG QUESTION ANSWERING
# =====================================================

def answer_question(vector_store, question, language):

    retrieved_docs = vector_store.similarity_search(
        question,
        k=4
    )

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    prompt = f"""
    You are a helpful YouTube learning assistant.

    Answer the question using only the provided
    video transcript context.

    Instructions:

    1. Respond in {language}.
    2. Explain concepts clearly.
    3. Use technical terminology correctly.
    4. Give examples when supported by the context.
    5. Do not invent information.
    6. If the video context does not contain
       the answer, clearly say that the topic
       is not sufficiently covered in the video.

    Video Context:

    {context}

    User Question:

    {question}

    Answer:
    """

    response = llm.invoke(prompt)

    return response.content


# =====================================================
# 10. SESSION STATE
# =====================================================

if "video_data" not in st.session_state:
    st.session_state.video_data = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# =====================================================
# 11. SIDEBAR SETTINGS
# =====================================================

with st.sidebar:

    st.header("⚙️ Settings")

    response_language = st.selectbox(
        "Response Language",
        ["English", "Hindi"]
    )

    workers = st.slider(
        "Translation Workers",
        min_value=1,
        max_value=16,
        value=8
    )

    st.caption(
        "Higher concurrency may trigger API rate limits."
    )


# =====================================================
# 12. VIDEO INPUT AND PROCESSING
# =====================================================

st.subheader("📺 Load YouTube Video")

with st.form("video_form"):

    video_url = st.text_input(
        "Paste YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    submitted = st.form_submit_button(
        "🚀 Process Video",
        use_container_width=True
    )


if submitted:

    if not video_url.strip():

        st.warning("Please enter a YouTube URL.")

    else:

        try:

            with st.status(
                "Processing your video...",
                expanded=True
            ) as process_status:

                # Extract ID
                video_id = extract_video_id(video_url)

                st.write("Video ID extracted successfully.")

                # Fetch transcript
                transcript, source_language = (
                    fetch_transcript(video_id)
                )

                st.write(
                    f"Transcript language: {source_language}"
                )

                st.write("Transcript fetched successfully.")

                # Chunking
                chunks = split_transcript(transcript)

                st.write(
                    f"Created {len(chunks)} transcript chunks."
                )

                # Translate only if necessary
                if source_language.lower().startswith("en"):

                    english_docs = chunks

                    st.write(
                        "English transcript detected. "
                        "Translation skipped."
                    )

                else:

                    st.write(
                        f"Translating using {workers} workers..."
                    )

                    english_docs = translate_transcript(
                        chunks,
                        workers
                    )

                english_transcript = "\n\n".join(
                    doc.page_content
                    for doc in english_docs
                )

                # Build vector store
                st.write("Generating embeddings and building FAISS index...")

                vector_store = build_vector_store(
                    english_docs
                )

                # Generate interview questions
                st.write("Generating 10 interview questions...")

                questions = generate_questions(
                    english_transcript,
                    response_language
                )

                process_status.update(
                    label="Video processing completed!",
                    state="complete",
                    expanded=False
                )

            # Save processed data
            st.session_state.video_data = {
                "video_id": video_id,
                "source_language": source_language,
                "transcript": english_transcript,
                "vector_store": vector_store,
                "questions": questions
            }

            st.session_state.messages = []
            st.session_state.pending_question = None

            st.success("Your video is ready for learning!")

        except TranscriptsDisabled:

            st.error("Captions are disabled for this video.")

        except NoTranscriptFound:

            st.error("No suitable transcript was found.")

        except VideoUnavailable:

            st.error("This video is unavailable.")

        except Exception as e:

            st.error(f"Processing failed: {str(e)}")


# =====================================================
# 13. DISPLAY PROCESSED VIDEO
# =====================================================

data = st.session_state.video_data

if data:

    st.divider()

    st.subheader("📚 Video Information")

    st.write("**Video ID:**", data["video_id"])

    st.write(
        "**Transcript language:**",
        data["source_language"]
    )

    # Interview questions
    st.subheader("🎯 Generated Interview Questions")

    questions = data["questions"]

    st.markdown(questions)

    # Extract numbered questions for Ask buttons
    question_list = re.findall(
        r"^\s*\d+[\.\)]\s*(.+?)\s*$",
        questions,
        flags=re.MULTILINE
    )

    if question_list:

        with st.expander(
            "Ask any generated interview question",
            expanded=True
        ):

            for i, question in enumerate(question_list):

                col1, col2 = st.columns([5, 1])

                with col1:
                    st.write(f"**{i+1}.** {question}")

                with col2:

                    if st.button(
                        "Ask",
                        key=f"ask_{i}"
                    ):

                        st.session_state.pending_question = (
                            question
                        )

    # =================================================
    # 14. CHAT INTERFACE
    # =================================================

    st.divider()

    st.subheader("💬 Ask Anything About This Video")

    st.caption(
        "Ask your own questions or select one "
        "of the generated interview questions."
    )

    # Display chat history
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # User input
    user_question = st.chat_input(
        "Ask a question about the video..."
    )

    question_to_answer = (
        user_question
        or st.session_state.pending_question
    )

    if question_to_answer:

        st.session_state.pending_question = None

        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "content": question_to_answer
        })

        # Generate answer
        with st.spinner("Thinking..."):

            try:

                answer = answer_question(
                    data["vector_store"],
                    question_to_answer,
                    response_language
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error generating answer: {e}"
                })

        st.rerun()

else:

    st.info(
        "Paste a YouTube URL and click Process Video "
        "to start learning."
    )
