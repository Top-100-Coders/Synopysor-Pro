# Synopysor-Pro
A Python information retrival and summarization application for video contents
using OpenAI's GPT 3-Turbo API and Streamlit.
This would be an ideal solution for those who are looking for quick information retrieval.

## Working
1. Synopysor-Pro works currently with YouTube video for summarization.
2. The application takes in your search item and scraps the results from YouTube. The Top 50 results are obtained, and 
from them, the result count mentioned in the slider (by default 2) would be 
taken into account for summarization.
3. Multiple summaries data is combined and provided to the ChatBot for the Question-Answer.


## Installation
1. Clone the repository
2. Set up the environment by running `python -m venv env`
3. Install the requirements by running `pip install -r requirements.txt`
4. Run the streamlit application by running `streamlit run main.py`

## Usage
1. Add your OpenAI API key and input your search tag
2. Once the summarized output is ready, the chatbot would be activated for the Q/A.