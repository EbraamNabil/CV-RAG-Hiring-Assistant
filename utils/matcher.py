import json

with open("data/skills.json", "r", encoding="utf-8") as f:
    SKILL_DB = json.load(f)


def normalize_skill(skill):

    skill = skill.lower().strip()

    for canonical, aliases in SKILL_DB.items():

        if skill == canonical.lower():
            return canonical

        for alias in aliases:

            if skill == alias.lower():
                return canonical

    return skill


def normalize_skill_list(skills):

    normalized = []

    for skill in skills:

        normalized.append(
            normalize_skill(skill)
        )

    return list(set(normalized))


def semantic_match(candidate_skills, required_skills):

    candidate = normalize_skill_list(candidate_skills)

    required = normalize_skill_list(required_skills)

    matched = []

    missing = []

    for skill in required:

        if skill in candidate:

            matched.append(skill)

        else:

            missing.append(skill)

    return matched, missing