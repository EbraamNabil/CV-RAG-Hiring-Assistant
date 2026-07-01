import json
import os

from groq import Groq
from dotenv import load_dotenv

from cache.manager_cache import (
    text_hash,
    cache_exists,
    load_cache,
    save_cache
)

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL_NAME = "llama-3.3-70b-versatile"


def evaluate_candidate(profile, job):
    
    
    cache_key = text_hash(

        json.dumps(
            profile,
            sort_keys=True
        )

        +

        json.dumps(
            job,
            sort_keys=True
        )

    )
    
    if cache_exists(cache_key):

        return load_cache(cache_key)
    

    prompt = f"""
You are a Senior Technical Recruiter.

Evaluate this candidate against the job.

Return ONLY valid JSON.

Schema:

{{
    "overall_assessment":"",
    "technical_fit":"",
    "strengths":[],
    "weaknesses":[],
    "interview_questions":[],
    "recommendation":""
}}

Job:

{json.dumps(job, indent=2)}

Candidate:

{json.dumps(profile, indent=2)}
"""

    response = client.chat.completions.create(

        model=MODEL_NAME,

        temperature=0,

        response_format={"type":"json_object"},

        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    
    
    report = json.loads(
        response.choices[0].message.content
    )

    save_cache(

        cache_key,

        report

    )

    return report