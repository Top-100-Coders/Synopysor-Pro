import operator
import time
from decouple import config
from concurrent.futures import ProcessPoolExecutor

import numpy as np
# import torch
from openai import OpenAI
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("t5-base")

from youtube_search import YoutubeSearch
from youtube_transcript_api import YouTubeTranscriptApi

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
    if len(content) >= 4000:
        content = content[:3900]
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


def get_topic_data(topic: str):
    """
    This function is to get the topic-based details using the Youtube search function
    :topic: str
    :return:
    """
    results = YoutubeSearch(topic, max_results=50).to_dict()
    for result in results:
        result['views'] = int(result['views'].replace(' views', '').replace(',', ''))
        result['thumbnails'] = result['thumbnails'][1] if len(result['thumbnails']) > 1 else result['thumbnails'][0]
        del result['long_desc']
        del result['channel']
        del result['duration']
        del result['publish_time']
        del result['url_suffix']
    results.sort(key=operator.itemgetter('views'), reverse=True)
    return results[:2]


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


def request_summary_from_gpt3(key_, topic, content, stage):
    """
    Function to request summary from GPT-3
    :param key_: API  key
    :param topic:
    :param content:
    :param stage:
    :return:
    """
    content = content[:3400]
    base_prompt_initial = f"""
    **Prompt for Summarization**
    **Description:**
    You are a highly capable AI language model with expertise in content summarization and contextual understanding. 
    Your proficiency extends to crafting descriptive yet comprehensible summaries. In this scenario, your task is to 
    generate a concise summary of the content from one of the top 10 YouTube videos about {topic} to be provided to you.
    Only the content of the video is to be summarized. The summary should be descriptive enough to be understood by
    anyone. The content of the video is provided below.
        ---     
        **Content:** {content}
        ---
    """
    base_prompt_final = f"""
        You are a very intelligent AI language model tasked with summarizing information about anything. Your are also an 
        expert in formatting specialising in understanding the context of the content and summarizing it in a concise manner.
        You are also very good at writing summaries in a very descriptive but enough to be understood by anyone. For the
        topic of {topic}, you are tasked with summarizing the top 10 videos from YouTube. You would be provided with a 
        combined content of all the videos. You are also provided with a summary of the content. Generate a summary of the
        content provided in the prompt ahead.
        ---     
        **Content:** {content}
        ---
    """

    # print(len(base_prompt_initial))
    # print(len(base_prompt_final))

    client = OpenAI(
        api_key=key_,
        )
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": base_prompt_final if stage == 2 else base_prompt_initial,
                    }
                ],
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
    base_prompt_final = f"""
        You are a very intelligent AI language model tasked with question and answering about anything provided as 
        summary and prompt. Your are also an expert in formatting specialising in understanding the context of the
        content and answering the questions in a concise manner. Make sure to answer the questions in a very descriptive
        way and if the answer to the question could not be determined with the information provided, answer it with a
        "No answer available" statement. You are tasked with answering the questions provided for the topic of {topic} 
        whose content is given below. They must be answered only if the content is available in the provided content.
        ---
        **Content:** {summary}
        ---
        **Prompt:** {prompt}
        ---
    """

    # print(len(base_prompt_initial))
    # print(len(base_prompt_final))

    client = OpenAI(
        api_key=key_,
        )
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": base_prompt_final,
                    }
                ],
            model="gpt-3.5-turbo",
            )
        return str(chat_completion.choices[0].message.content)
    except Exception:
        return ''