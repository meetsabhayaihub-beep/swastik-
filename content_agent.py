import os
import time
import requests
from dotenv import load_dotenv

from agents import Agent, Runner, function_tool
from pydantic import BaseModel

load_dotenv()


class ContentPack(BaseModel):
    linkedin_post: str
    reel_hook: str
    reel_script: str
    reel_caption: str
    hashtags: list[str]
    cta: str


# ---------- Agent tools ----------

@function_tool
def publish_linkedin_post(text: str) -> str:
    """Publish a text-only post to LinkedIn. Requires LinkedIn API access."""
    url = "https://api.linkedin.com/rest/posts"
    headers = {
        "Authorization": f"Bearer {os.environ['LINKEDIN_ACCESS_TOKEN']}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
        "LinkedIn-Version": "202610",
    }
    payload = {
        "author": os.environ["LINKEDIN_AUTHOR_URN"],
        "commentary": text,
        "visibility": "PUBLIC",
        "distribution": {
            "feedDistribution": "MAIN_FEED",
            "targetEntities": [],
            "thirdPartyDistributionChannels": [],
        },
        "lifecycleState": "PUBLISHED",
        "isReshareDisabledByAuthor": False,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.headers.get("x-restli-id", "LinkedIn post created")


@function_tool
def create_instagram_reel(video_url: str, caption: str) -> str:
    """Create and publish an Instagram Reel from a public MP4 URL."""
    ig_user_id = os.environ["INSTAGRAM_USER_ID"]
    token = os.environ["INSTAGRAM_ACCESS_TOKEN"]
    base = f"https://graph.facebook.com/v23.0/{ig_user_id}"

    container = requests.post(
        f"{base}/media",
        data={
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "access_token": token,
        },
        timeout=30,
    )
    container.raise_for_status()
    container_id = container.json()["id"]

    # Wait briefly for Meta to finish processing the uploaded video.
    for _ in range(20):
        status = requests.get(
            f"{base}/{container_id}",
            params={"fields": "status_code", "access_token": token},
            timeout=30,
        ).json()
        if status.get("status_code") == "FINISHED":
            break
        time.sleep(3)
    else:
        raise RuntimeError("Instagram Reel processing did not finish in time.")

    publish = requests.post(
        f"{base}/media_publish",
        data={
            "creation_id": container_id,
            "access_token": token,
        },
        timeout=30,
    )
    publish.raise_for_status()
    return publish.json()["id"]


# ---------- Content agent ----------

content_agent = Agent(
    name="Social Content Agent",
    model="gpt-5-nano",
    output_type=ContentPack,
    instructions="""
You are an expert social-media content strategist.

For every user topic:
1. Write a LinkedIn post: strong first line, practical insight, short paragraphs,
   no more than 1,300 characters, professional but human tone.
2. Write an Instagram Reel: a 3-second hook, 4-6 short scene directions,
   on-screen text, voiceover, and a clear CTA. Keep it under 45 seconds.
3. Write a concise Instagram caption.
4. Return 8-12 relevant hashtags.
5. Return one specific call to action.

Never invent statistics, customer results, product claims, or credentials.
If publishing tools are available, ask for explicit confirmation before publishing.
""",
    tools=[publish_linkedin_post, create_instagram_reel],
)


# ---------- Runner ----------

async def generate_content(topic: str):
    result = await Runner.run(
        content_agent,
        f"Create LinkedIn and Instagram Reel content for this topic:\n\n{topic}",
    )
    return result.final_output


if __name__ == "__main__":
    import asyncio

    topic = input("Enter your topic: ")
    pack = asyncio.run(generate_content(topic))

    print("\n--- LINKEDIN POST ---")
    print(pack.linkedin_post)

    print("\n--- INSTAGRAM REEL ---")
    print(f"Hook: {pack.reel_hook}\n")
    print(pack.reel_script)
    print(f"\nCaption: {pack.reel_caption}")
    print(f"Hashtags: {' '.join(pack.hashtags)}")
    print(f"CTA: {pack.cta}")
