from utils.matcher import semantic_match


# ===============================
# Skill Weights
# ===============================

SKILL_WEIGHTS = {

    "Python": 10,
    "SQL": 8,

    "Machine Learning": 15,
    "Deep Learning": 15,
    "NLP": 15,
    "Computer Vision": 15,

    "LangChain": 12,
    "LangGraph": 12,
    "RAG": 15,

    "FastAPI": 8,

    "Docker": 8,

    "Git": 5,

    "AWS": 10,
    "Azure": 10,

    "OpenAI": 10,
    "Ollama": 8
}


# ===============================
# Skills
# ===============================

def calculate_skill_score(candidate_skills, required_skills):

    total = 0
    matched = 0

    matched_skills = []
    missing_skills = []

    for skill in required_skills:

        weight = SKILL_WEIGHTS.get(skill, 5)

        total += weight

        if skill in candidate_skills:

            matched += weight
            matched_skills.append(skill)

        else:

            missing_skills.append(skill)

    if total == 0:

        return 0, matched_skills, missing_skills

    score = round((matched / total) * 50, 2)

    return score, matched_skills, missing_skills


# ===============================
# Experience
# ===============================

def calculate_experience_score(candidate_years, required_years):

    if candidate_years >= required_years + 2:

        return 20

    elif candidate_years >= required_years:

        return 15

    elif candidate_years >= max(required_years - 1, 0):

        return 10

    return 0


# ===============================
# Education
# ===============================

def calculate_education_score(candidate_degree, required_degree):

    if not required_degree:

        return 10

    if required_degree.lower() in candidate_degree.lower():

        return 10

    return 0


# ===============================
# Projects
# ===============================

def calculate_project_score(projects):

    if len(projects) >= 5:

        return 15

    elif len(projects) >= 3:

        return 10

    elif len(projects) >= 1:

        return 5

    return 0


# ===============================
# Certifications
# ===============================

def calculate_certificate_score(candidate_certs, required_certs):

    candidate = {

        c.lower()

        for c in candidate_certs

    }

    required = {

        c.lower()

        for c in required_certs

    }

    return min(

        len(candidate & required) * 5,

        5

    )


# ===============================
# Final Score
# ===============================

def calculate_final_score(profile, job):

    matched, missing = semantic_match(

        profile["skills"],

        job["required_skills"]

    )

    skills_score, _, _ = calculate_skill_score(

        matched,

        job["required_skills"]

    )

    experience_score = calculate_experience_score(

        profile["years_of_experience"],

        job["minimum_experience"]

    )

    education_score = calculate_education_score(

        profile["education"]["degree"],

        job["education"]

    )

    project_score = calculate_project_score(

        profile["projects"]

    )

    certificate_score = calculate_certificate_score(

        profile["certifications"],

        job["certifications"]

    )

    final_score = (

        skills_score

        + experience_score

        + education_score

        + project_score

        + certificate_score

    )

    final_score = min(final_score, 100)

    breakdown = {

        "Skills": round(skills_score, 1),

        "Experience": experience_score,

        "Education": education_score,

        "Projects": project_score,

        "Certifications": certificate_score

    }

    return (

        final_score,

        breakdown,

        matched,

        missing

    )