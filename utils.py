import operator
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from openai import OpenAI
from youtube_search import YoutubeSearch
from youtube_transcript_api import YouTubeTranscriptApi


# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# tokenizer = AutoTokenizer.from_pretrained("t5-base")

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# model = AutoModelForSeq2SeqLM.from_pretrained(
#     "t5-base",
#     return_dict=True)
# model.to(device)

# API_KEY = config('API_KEY', cast=str)


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(time.time() - start)
        return result

    return wrapper


# @timeit
# @lru_cache(maxsize=None)
def get_transcript_for_video(video_id):
    """
    Function to get transcript from YouTube
    :param video_id:
    :return:
    """
    content = ''
    try:
        srt = YouTubeTranscriptApi.get_transcript(video_id)
        for i in range(len(srt)):
            content += srt[i]['text']
            content = content.replace('\n', ' ').replace('[Music]', ' ')
    except Exception:
        pass
    return content


# @lru_cache(maxsize=None)
def get_yt_transcript(video_ids):
    content = []
    video_ids = tuple(video_ids)
    with ProcessPoolExecutor() as executor:
        transcripts = list(executor.map(get_transcript_for_video, video_ids))

    for transcript in transcripts:
        content.append(f"{transcript}\n")
    return content


def get_topic_data(topic: str, quantity: int = 2):
    """
    This function is to get the topic-based details using the YouTube search function
    :topic: str
    :return:
    """
    results = YoutubeSearch(topic, max_results=50).to_dict()
    for result in results:
        result['views'] = int(result['views'].replace(' views', '').replace(',', ''))
        result['thumbnails'] = result['thumbnails'][1] if len(result['thumbnails']) > 1 else result['thumbnails'][0]
        try:
            result['duration'] = int(result['duration'].split(':')[0])
        except ValueError:
            result['duration'] = 0
        del result['long_desc']
        del result['channel']
        del result['publish_time']
        del result['url_suffix']
    results = list(filter(lambda x: x['duration'] > 1, results))
    results.sort(key=operator.itemgetter('views'), reverse=True)
    return results[:quantity]


# def get_summary(content):
#     inputs = tokenizer.encode(
#         "summarize: " + content,
#         return_tensors="pt",
#         max_length=2048,
#         truncation=True).to(device)
#
#     summary_ids = model.generate(inputs, max_length=2048, min_length=2000, length_penalty=5.0, num_beams=5,
#                                  early_stopping=True)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return summary
#
#
# def process_summary(summaries):
#     content = ''
#     summaries = tuple(summaries)
#
#     for summary in summaries:
#         content += f" {get_summary(summary)}\n"
#     return content


def chunk_summary(content):
    content = content.split(' ')
    chunks = np.array_split(content, 4)
    sentences = ' '.join(list(chunks[0]))
    prompt = f"{sentences}\n\ntl;dr:"
    return prompt


def multiproc_summarizer(data):
    content = []
    data = tuple(data)
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(request_summary_from_gpt3, data))

    for result in results:
        content.append(f"{result}\n")
    return content


def request_summary_from_gpt3(key_, topic, content, stage):
    """
    Function to request summary from GPT-3
    :param key_: API key of OpenAI GPT-3
    :param topic: Topic of the content
    :param content: Content to be summarized
    :param stage: Stage of the summarization, 1 for initial and 2 for final
    :return:
    """
    if len(content) >= 16385:
        content = content[:15500]
    messages_initial = [
        { "role": "system",
          "content": f"The AI system is designed to intelligently summarize information on a given topic, specializing "
                     f"in contextual understanding and descriptive formatting. For the specific task at hand, it is "
                     f"focused on generating a concise summary of the content from a YouTube video about {topic}. The AI "
                     f"model excels at crafting summaries that are both descriptive and easily understandable." },
        { "role": "user",
          "content": f"You are a user seeking a detailed and well-structured summary of content related to {topic} from "
                     f"a YouTube video. The AI language model is proficient in understanding the context and summarizing "
                     f"information in a descriptive yet clear manner. Provide the content of the video, and the model "
                     f"will generate a summary based on the given information." },
        { "role": "assistant", "content": f"**Content:** {content}" },
        ]

    messages_final = [
        { "role": "system",
          "content": f"The AI system is designed to intelligently summarize information on a given topic, specializing in"
                     f" contextual understanding and descriptive formatting. For the specific topic of {topic}, "
                     f"it is tasked with summarizing content from the top 10 YouTube videos. The system receives a "
                     f"combined content of all videos and a pre-existing summary, aiming to generate a concise and "
                     f"descriptive summary in response." },
        { "role": "user",
          "content": f"You are a user seeking a detailed and well-structured summary of content related to {topic}"
                     f". The AI language model is proficient in understanding the context and "
                     f"summarizing information in a descriptive yet clear manner. Provide the combined content of the "
                     f"videos, and the model will generate a summary based on the given information." },
        { "role": "assistant",
          "content": f"You are a very intelligent AI language model with expertise in summarizing information and "
                     f"formatting content contextually. Specifically tasked with summarizing the textual transcriptions"
                     f"from YouTube videos on {topic}, you are provided with combined pre-existing summaries. Your goal "
                     f"is to generate a concise and descriptive summary that captures the key points from the provided "
                     f"information." },
        { "role": "user", "content": f"Here is the content for summarization:\n{content}" },
        ]

    # print(len(base_prompt_initial))
    # print(len(base_prompt_final))

    client = OpenAI(
        api_key=key_,
        )
    try:
        chat_completion = client.chat.completions.create(
            messages=messages_final if stage == 2 else messages_initial,
            model="gpt-3.5-turbo",
            )
        return str(chat_completion.choices[0].message.content)
    except Exception:
        return ''


def request_qa_from_gpt3(key_, topic, summary, prompt):
    """
    Function to request summary from GPT-3
    :param key_:
    :param topic:
    :param summary:
    :param prompt:
    :return:
    """

    client = OpenAI(
        api_key=key_,
        max_retries=3
        )
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                { "role": "system", "content": "You are a helpful and intelligent assistant who is designed to be a "
                                               "knowledgeable resource for users. You are able to answer questions "
                                               "about a given topic, and you are also able to provide a summary of "
                                               "information on a given topic.You are provided with a summary of the "
                                               "topic and a prompt with questions to answer. Your goal is to answer the "
                                               "questions to the best of your ability, using the information provided in "
                                               "the summary and only to answer from the summary. If the question doesn't "
                                               "have any related terms or answers in the summary, then just reply you"
                                               "don't know the answer. Make sure you never ask questions and only answer"
                                               " questions" },
                { "role": "user", "content": f"What is the {topic} about?" },
                { "role": "assistant", "content": f"The content is about {summary}." },
                { "role": "user", "content": f"Can you answer the question: {prompt}" }
                ],
            model="gpt-3.5-turbo"
            )
        return str(chat_completion.choices[0].message.content)
    except Exception:
        return ''