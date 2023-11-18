import streamlit as st
import streamlit_scrollable_textbox as sst

from utils import get_topic_data, get_yt_transcript, request_summary_from_gpt3, request_qa_from_gpt3

if 'my_api' not in st.session_state:
    st.session_state.my_api = None
if 'my_prompt' not in st.session_state:
    st.session_state.my_prompt = None
if 'my_list' not in st.session_state:
    st.session_state.my_list = []
if 'my_topic' not in st.session_state:
    st.session_state.my_topic = None
if 'my_summary' not in st.session_state:
    st.session_state.my_summary = None
if 'my_thumbnails' not in st.session_state:
    st.session_state.my_thumbnails = []
if 'my_content_ids' not in st.session_state:
    st.session_state.my_content_ids = []
if 'my_raw_summary' not in st.session_state:
    st.session_state.my_raw_summary = []

st.markdown("<h1 style='text-align: center; color: red;'>Synopysor-Pro</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: black;'>Content summarization tool</h4>", unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.markdown("This is a tool to help you summarise the content of a video with ease")
    st.markdown("1. Please enter the topic you want to search for and click on the button to obtain the contents")
    st.markdown("2. Once you have obtained the contents, click on the button to start the extraction process")
    st.markdown("3. Once the extraction process is completed, the bot will be ready for QnA")
    st.markdown("4. Please enter your question and click on the send button to get your answer")
    st.markdown(
        "5. Please note that the bot will only be able to answer questions related to the topic you have searched")

st.subheader("⚙️ Setup")
api_key = st.text_input("Enter the OpenAI API key: ")
st.session_state.my_api_key = api_key

if st.session_state.my_api_key is not None and st.session_state.my_api_key != '':
    topic_input = st.text_input("Enter the topic you want to search the content for: ")
    st.session_state.my_topic = topic_input
    if st.session_state.my_topic is not None and st.session_state.my_topic != '':
        wcol1, wcol2 = st.columns(2)
        wcol2 = st.slider("Number of contents to be obtained", min_value=2, max_value=50, value=4, step=1)
        wcol1 = st.button("Obtain contents", type="primary")
        if wcol1:
            with st.spinner("Searching for the contents..."):
                results = get_topic_data(st.session_state.my_topic, 4 if wcol2 is None else wcol2)
            if results.__len__() == 0:
                st.error("No results found")
                st.stop()
            st.session_state.my_content_ids = [result['id'] for result in results]
            st.session_state.my_thumbnails = [result['thumbnails'] for result in results]
            colm1, colm2 = st.columns(2)
            colm3, colm4 = st.columns(2)
            img_grid = [colm1, colm2, colm3, colm4]
            if st.session_state.my_thumbnails.__len__() <= 2:
                img_grid = [colm1, colm2]
            for col in img_grid:
                col.image(st.session_state.my_thumbnails[img_grid.index(col)])

if (st.session_state.my_content_ids is not None
        and st.session_state.my_content_ids != []
        and st.session_state.my_summary is None):
    st.divider()
    st.subheader("📋 Summarize")

    with st.spinner("Obtaining transcriptions..."):
        transcript_datas = get_yt_transcript(st.session_state.my_content_ids)
    with st.spinner("Converting transcripts to summaries..."):
        for transcript_data in transcript_datas:
            st.session_state.my_raw_summary.append(
                request_summary_from_gpt3(st.session_state.my_api_key, st.session_state.my_topic, transcript_data,
                                          1))
    sst.scrollableTextbox(''.join(st.session_state.my_raw_summary), height=400)

    with st.spinner("Finalising summarised data..."):
        final_summary = request_summary_from_gpt3(st.session_state.my_api_key, st.session_state.my_topic,
                                                  ''.join(st.session_state.my_raw_summary), 2)
    st.session_state.my_summary = final_summary

if st.session_state.my_summary is not None and st.session_state.my_summary != '':
    st.divider()
    st.title("💬 Synopysor-Bot")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{ "role": "assistant", "content": "How can I help you?" }]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Create an input field at the bottom
    if prompt := st.chat_input():
        st.session_state.my_prompt = prompt
        st.session_state.messages.append({ "role": "user", "content": st.session_state.my_prompt })
        st.chat_message("user").write(st.session_state.my_prompt)

        # Add assistant response to message
        with st.spinner("Thinking..."):
            response = request_qa_from_gpt3(
                st.session_state.my_api_key, st.session_state.my_topic,
                st.session_state.my_summary, st.session_state.my_prompt)
            st.session_state["messages"].append({ "role": "assistant", "content": response })
            st.chat_message("assistant").write(response)