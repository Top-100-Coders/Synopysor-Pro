import streamlit as st
import streamlit_scrollable_textbox as sst
from streamlit_extras import add_vertical_space as avs

from utils import get_topic_data, get_yt_transcript, chunk_summary, request_summary_from_gpt3, request_qa_from_gpt3

# results = None
# final_summary = None
summarised_data = []
content_ids = []

# st.title("Synopysor-Pro")
st.markdown("<h1 style='text-align: center; color: red;'>Synopysor-Pro</h1>", unsafe_allow_html=True)
st.divider()
avs.add_vertical_space(2)
if 'my_api_key' not in st.session_state:
    st.session_state.my_api_key = None

with st.sidebar:
    st.title("Synopysor-Pro")
    st.header("Content summarization tool")

    st.markdown("This is a tool to help you summarise the content of a video with ease")
    # st.markdown("The tool is divided into 3 sections: Setup, Summarize and QnA")
    # st.markdown("Before using the tool, enter ur API key")
    st.markdown("1. Please enter the topic you want to search for and click on the button to obtain the contents")
    st.markdown("2. Once you have obtained the contents, click on the button to start the extraction process")
    st.markdown("3. Once the extraction process is completed, the bot will be ready for QnA")
    st.markdown("4. Please enter your question and click on the send button to get your answer")
    st.markdown(
        "5. Please note that the bot will only be able to answer questions related to the topic you have searched")
# tab1, tab2, tab3 = st.tabs(["⚙️ Setup", "📋 Summarize", "🙋 QnA"])

if 'my_list' not in st.session_state:
    st.session_state.my_list = []
if 'my_topic' not in st.session_state:
    st.session_state.my_topic = None
if 'my_summary' not in st.session_state:
    st.session_state.my_summary = None
if "messages" not in st.session_state:
    st.session_state["messages"] = [{ "role": "assistant", "content": "How can I help you?" }]

# with tab1:
st.subheader("⚙️ Setup")
api_key = st.text_input("Enter the OpenAI API key: ")
if api_key is not None and api_key != '':
    topic_input = st.text_input("Enter the topic you want to search the content for: ")

    if topic_input is not None and topic_input != '':
        if st.button("Obtain contents", type="primary"):
            with st.spinner("Searching for the contents..."):
                results = get_topic_data(topic_input)
            content_ids = [result['id'] for result in results]
            thumbnails = [result['thumbnails'] for result in results]
            colm1, colm2 = st.columns(2)
            img_grid = [colm1, colm2]
            for col in img_grid:
                col.image(thumbnails[img_grid.index(col)])

        if content_ids is not None and content_ids != []:
            st.divider()
            st.subheader("📋 Summarize")
            # if st.button("Start extraction", type="primary"):

            with st.spinner("Obtaining transcriptions..."):
                transcript_datas = get_yt_transcript(content_ids)
            with st.spinner("Converting transcripts to summaries..."):
                for transcript_data in transcript_datas:
                    summarised_data.append(
                        request_summary_from_gpt3(api_key, topic_input, transcript_data,
                                                  1))
                    print("API call")
            sst.scrollableTextbox(''.join(summarised_data), height=400)

            with st.spinner("Finalising summarised data..."):
                final_summary = request_summary_from_gpt3(api_key, topic_input,
                                                          ''.join(summarised_data), 2)
            st.session_state.my_summary = final_summary
            if st.session_state.my_summary is not None and final_summary != '':
                st.divider()
                st.title("💬 Chatbot")
                if "messages" not in st.session_state:
                    st.session_state["messages"] = [{ "role": "assistant", "content": "How can I help you?" }]
                for msg in st.session_state.messages:
                    st.chat_message(msg["role"]).write(msg["content"])

                # Create an input field at the bottom
                if prompt := st.chat_input():
                    st.session_state.messages.append({ "role": "user", "content": prompt })

                    # Add assistant response to message
                    with st.spinner("Thinking..."):
                        response = request_qa_from_gpt3(api_key, topic_input, st.session_state.my_summary, prompt)
                        st.session_state["messages"].append({ "role": "assistant", "content": response })
                        st.chat_message("assistant").write(response)