# Synopysor-Pro
Synopysor-Pro is a Python information retrieval and summarization application designed to streamline the process of 
extracting key insights from YouTube video content. Leveraging OpenAI's powerful GPT-3 Turbo API and the user-friendly 
Streamlit framework, this application is an ideal solution for individuals seeking quick and efficient information retrieval. The application is an effective solution when you need to summarize a lot of youtube videos for quick data / info retireval.

Try it out here: https://synopysor-pro.streamlit.app/

![image](https://github.com/suryan-s/Synopysor-Pro/assets/76394506/e572ca45-c754-4ccb-a5d5-516bfaaacae5)

![image](https://github.com/suryan-s/Synopysor-Pro/assets/76394506/b360d732-a401-4770-acbc-146519565727)


## Working
1. **YouTube Video Summarization**: Synopysor-Pro focuses on summarizing YouTube videos, providing users with concise and informative summaries.
2. **Search and Scrap**: Users input their search item, and Synopysor-Pro scraps the top 50 results from YouTube. The application allows users to select the desired number of results (default is 2) for summarization.
3. **Combining Multiple Summaries**: The summaries from the selected results are combined and processed through a ChatBot for an interactive question-and-answer session.


## Installation
1. Clone the repository
2. Set up the environment by running `python -m venv env`
3. Install the requirements by running `pip install -r requirements.txt`
4. Run the streamlit application by running `streamlit run main.py`

## Usage
1. **API Key Setup**: Add your OpenAI API key to enable integration with the powerful GPT-3 Turbo API.
2. **Search and Summarize**: Input your search tag, and Synopysor-Pro will retrieve and summarize the top results from YouTube.
3. **ChatBot Q/A**: Once the summarized output is ready, engage with the built-in chatbot for interactive question-and-answer sessions.


## Contributors
- [@suryan-s](https://github.com/suryan-s)
- [@Abhishek-S-Lal](https://github.com/Abhishek-S-Lal)


Feel free to explore Synopysor-Pro for a seamless and efficient experience in extracting valuable information from YouTube videos.
