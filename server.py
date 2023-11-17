from fastapi import FastAPI
from requests import Request

import utils

app = FastAPI()


@app.get("/")
async def root():
    return { "message": "Hello World" }


@app.get("/search/{topic}")
async def search(topic: str):
    results = utils.get_topic_data(topic)
    content_ids = [result['id'] for result in results]
    thumbnails = [result['thumbnails'] for result in results]
    return { "content_ids": content_ids, "thumbnails": thumbnails }, 200


@app.get("/transcript/{video_id}")
async def transcript(video_id: str):
    srt = utils.get_transcript_for_video(video_id)
    return { "transcript": srt }, 200


@app.get("/summary")
async def video_summary(request: Request):
    data = await request.json()
    topic = data['topic']
    srt = data['transcript']
    num = int(data['num'])
    summary_ = utils.request_summary_from_gpt3(topic, srt, num)
    return { "summary": summary_ }, 200


@app.get("/QA")
async def qa(request: Request):
    data = await request.json()
    prompt = data['prompt']
    context = data['context']
    topic = data['topic']
    answer_ = utils.request_qa_from_gpt3(topic, context, prompt)
    return { "answer": answer_ }, 200


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app)