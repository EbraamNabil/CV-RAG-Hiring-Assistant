import json
import re

with open("data/skills.json","r",encoding="utf-8") as f:

    content = f.read()

print(content)

SKILL_DB = json.loads(content)






def extract_skills(text):

    text = text.lower()

    skills = []

    for skill in SKILL_DB:

        if skill.lower() in text:
            skills.append(skill)

    return sorted(list(set(skills)))


def extract_candidate_name(text):

    lines = text.split("\n")[:15]

    blacklist = [
        "curriculum",
        "resume",
        "email",
        "phone",
        "skills",
        "experience",
        "education",
        "profile"
    ]
    
    

    for line in lines:

        line = line.strip()

        if len(line.split()) > 4:
            continue

        if not re.match(r"^[A-Za-z\s\.]+$", line):
            continue

        if any(word in line.lower() for word in blacklist):
            continue

        return line

    return "UNKNOWN"


def extract_email(text):

    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"

    match = re.search(pattern, text)

    if match:
        return match.group()

    return ""

def extract_phone(text):

    pattern = r"(\+?\d[\d\-\s]{8,}\d)"

    match = re.search(pattern, text)

    if match:
        return match.group()

    return ""

def extract_experience(text):

    text = text.lower()

    patterns = [

        r"(\d+)\+?\s*years",

        r"(\d+)\+?\s*year",

        r"experience\s*:?[\s]*(\d+)",

        r"(\d+)\+?\s*yrs"

    ]

    for pattern in patterns:

        match = re.search(pattern, text)

        if match:

            return int(match.group(1))

    return 0
