import json
import os

from dotenv import load_dotenv
from groq import Groq

from cache.manager_cache import (
    file_hash,
    cache_exists,
    load_cache,
    save_cache
)
from cache.manager_cache import file_hash

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL_NAME = "llama-3.3-70b-versatile"

DEBUG = False


def extract_candidate_profile(cv_text , file_path):

    # ===============================
    # Check Cache
    # ===============================

    cache_key = file_hash(file_path)

    if cache_exists(cache_key):

        if DEBUG:
            print("✅ Loaded from Cache")

        return load_cache(cache_key)

    # ===============================
    # Prompt
    # ===============================

    prompt = f"""
You are an expert AI Recruitment Assistant.

Analyze the following CV carefully.

Return ONLY valid JSON.

Do not write markdown.
Do not explain anything.
Do not use ```json.

Return exactly this schema:

{{
    "name":"",
    "email":"",
    "phone":"",
    "linkedin":"",
    "github":"",
    "location":"",
    "years_of_experience":0,

    "education":{{
        "degree":"",
        "university":"",
        "major":""
    }},

    "skills":[],

    "projects":[
        {{
            "name":"",
            "description":"",
            "technologies":[],
            "domain":""
        }}
    ],

    "certifications":[],

    "languages":[],

    "summary":"",

    "strengths":[],

    "weaknesses":[],

    "recommended_roles":[]

}}

CV:

{cv_text}
"""

    # ===============================
    # Call Groq
    # ===============================

    response = client.chat.completions.create(

        model=MODEL_NAME,

        temperature=0,

        response_format={
            "type": "json_object"
        },

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # ===============================
    # Parse Response
    # ===============================

    content = response.choices[0].message.content

    if DEBUG:

        print("=" * 60)
        print(content)
        print("=" * 60)

    profile = json.loads(content)

    # ===============================
    # Save Cache
    # ===============================

    save_cache(
        cache_key,
        profile
    )

    return profile