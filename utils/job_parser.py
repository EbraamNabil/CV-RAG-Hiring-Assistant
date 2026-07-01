import json
import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL_NAME = "llama-3.3-70b-versatile"


def extract_job_requirements(job_description):

    prompt = f"""
You are an expert Technical Recruiter.

Analyze the following Job Description.

Return ONLY valid JSON.

Schema:

{{
    "job_title":"",
    "required_skills":[],
    "preferred_skills":[],
    "minimum_experience":0,
    "education":"",
    "certifications":[],
    "responsibilities":[]
}}

Job Description:

{job_description}
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

    return json.loads(
        response.choices[0].message.content
    )