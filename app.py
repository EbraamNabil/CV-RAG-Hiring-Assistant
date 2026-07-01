import os
import json

import pandas as pd
import plotly.express as px
import streamlit as st
import hashlib
from rag.pipeline import build_rag


from dotenv import load_dotenv
from groq import Groq
from streamlit_option_menu import option_menu

from langchain_community.document_loaders import PyPDFLoader

from utils.extractor import (
    extract_candidate_name,
    extract_skills,
)

from utils.llm_extractor import extract_candidate_profile
from utils.job_parser import extract_job_requirements
from utils.scorer import calculate_final_score
from utils.evaluator import evaluate_candidate


# ===============================
# Configuration
# ===============================

load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile"

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# ===============================
# Streamlit Config
# ===============================

st.set_page_config(
    page_title="Enterprise AI HR Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Enterprise AI HR Assistant")

selected = option_menu(
    menu_title=None,

    options=[
        "HR Recruitment",
        "AI HR Chat"
    ],

    icons=[
        "people-fill",
        "robot"
    ],

    orientation="horizontal",

    default_index=0,

    styles={
        "container": {
            "padding": "8px",
            "background-color": "#111827",
            "border-radius": "12px"
        },

        "icon": {
            "color": "#60a5fa",
            "font-size": "20px"
        },

        "nav-link": {
            "font-size": "18px",
            "font-weight": "600",
            "padding": "14px",
            "border-radius": "10px"
        },

        "nav-link-selected": {
            "background-color": "#2563eb",
            "color": "white"
        }
    }
)

# ===============================
# HR Recruitment Mode
# ===============================

if selected == "HR Recruitment":

    st.header("👨‍💼 HR Recruitment")

    uploaded_files = st.file_uploader(
        "📂 Upload CVs",
        type=["pdf"],
        accept_multiple_files=True
    )

    job_description = st.text_area(
        "📝 Job Description",
        height=220,
        placeholder="""
        Example:

        Python
        LangChain
        LangGraph
        FastAPI
        Docker
        Git
        SQL
        Machine Learning
        AWS
        """
    )

    analyze = st.button(
        "🚀 Analyze Candidates",
        use_container_width=True
    )

    if not analyze:
        st.stop()

    if not uploaded_files:
        st.warning("Please upload at least one CV.")
        st.stop()

    if not job_description.strip():
        st.warning("Please enter the Job Description.")
        st.stop()

    os.makedirs("temp", exist_ok=True)

    candidates = {}
    pdf_paths = []

    with st.spinner("📄 Reading CVs..."):
        
        for file in uploaded_files:

            path = os.path.join(
                "temp",
                file.name
            )

            with open(path, "wb") as f:
                f.write(file.getbuffer())
                
            pdf_paths.append(path)

            loader = PyPDFLoader(path)

            pages = loader.load()

            full_text = "\n".join(
                page.page_content
                for page in pages
            )

            fallback_name = extract_candidate_name(
                pages[0].page_content
            )

            if fallback_name == "UNKNOWN":
                fallback_name = os.path.splitext(file.name)[0]

            profile = extract_candidate_profile(
                full_text,
                path
            )

            if not profile.get("name"):
                profile["name"] = fallback_name

            profile["raw_text"] = full_text
            profile["pages"] = pages

            profile["score"] = 0
            profile["matched_skills"] = []
            profile["missing_skills"] = []
            profile["recommendation"] = ""

            candidates[file.name] = profile

    retriever = build_rag(pdf_paths)

    st.session_state.retriever = retriever
    
    st.session_state.candidates = candidates

    st.success(
            f"✅ {len(candidates)} candidate(s) loaded successfully."
        )  
    
    # ===============================
    # Parse Job Description
    # ===============================

    with st.spinner("🧠 Understanding Job Description..."):

        job = extract_job_requirements(
            job_description
        )

    required_skills = job["required_skills"]


    # ===============================
    # Candidate Ranking
    # ===============================

    ranking = []

    for profile in candidates.values():

        score,breakdown, matched, missing = calculate_final_score(
            profile,
            job
        )

        profile["score"] = score
        profile["matched_skills"] = matched
        profile["missing_skills"] = missing
        profile["breakdown"] = breakdown

        if score >= 80:

            profile["recommendation"] = "Interview"

        elif score >= 60:

            profile["recommendation"] = "Hold"

        else:

            profile["recommendation"] = "Reject"

        ranking.append(
            {
                "Candidate": profile["name"],
                "Score": round(score, 1),
                "Matched Skills": ", ".join(matched),
                "Missing Skills": ", ".join(missing),
                "Recommendation": profile["recommendation"]
            }
        )

    ranking = sorted(
        ranking,
        key=lambda x: x["Score"],
        reverse=True
    )

    st.session_state.required_skills = required_skills


    # ===============================
    # Dashboard Statistics
    # ===============================

    interview_count = sum(
        1 for p in candidates.values()
        if p["recommendation"] == "Interview"
    )

    hold_count = sum(
        1 for p in candidates.values()
        if p["recommendation"] == "Hold"
    )

    reject_count = sum(
        1 for p in candidates.values()
        if p["recommendation"] == "Reject"
    )

    avg_score = round(

        sum(
            p["score"]
            for p in candidates.values()
        ) / len(candidates),

        1
    )


    # ===============================
    # Dashboard
    # ===============================

    st.divider()

    st.header("📊 HR Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "👥 Candidates",
            len(candidates)
        )

    with col2:

        st.metric(
            "🟢 Interview",
            interview_count
        )

    with col3:

        st.metric(
            "🟡 Hold",
            hold_count
        )

    with col4:

        st.metric(
            "🔴 Reject",
            reject_count
        )

    st.metric(
        "⭐ Average Score",
        f"{avg_score}%"
    )


    # ===============================
    # Charts
    # ===============================

    left, right = st.columns(2)

    with left:

        scores_df = pd.DataFrame(ranking)

        fig = px.bar(

            scores_df,

            x="Candidate",

            y="Score",

            color="Score",

            text="Score",

            title="Candidate Scores"

        )

        fig.update_layout(
            height=420
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        pie_df = pd.DataFrame({

            "Status":[
                "Interview",
                "Hold",
                "Reject"
            ],

            "Count":[
                interview_count,
                hold_count,
                reject_count
            ]
        })

        fig = px.pie(

            pie_df,

            names="Status",

            values="Count",

            hole=0.55,

            title="Hiring Recommendation"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ===============================
    # Ranking Table
    # ===============================

    st.divider()

    st.header("🏆 Candidate Ranking")

    st.dataframe(
        ranking,
        use_container_width=True
    )

    df = pd.DataFrame(ranking)

    st.download_button(

        label="⬇ Download Ranking",

        data=df.to_csv(index=False),

        file_name="candidate_ranking.csv",

        mime="text/csv",

        key="download_ranking"
    )


    # ===============================
    # Top Candidates
    # ===============================

    st.divider()

    st.header("🥇 Top 5 Candidates")

    top5 = sorted(

        candidates.values(),

        key=lambda x: x["score"],

        reverse=True

    )[:5]

    cols = st.columns(len(top5))

    for col, profile in zip(cols, top5):

        with col:

            st.info(
                f"""
    👤 **{profile["name"]}**

    ⭐ **{profile["score"]:.1f}%**

    💼 {profile["years_of_experience"]} Years

    🎯 {profile["recommendation"]}
    """
            )
            
    # ===============================
    # Candidate Details
    # ===============================

    st.divider()

    st.header("👤 Candidate Details")

    sorted_candidates = sorted(
        candidates.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    for profile in sorted_candidates:

        with st.expander(
            f"{profile['name']} • {profile['score']:.1f}%"
        ):

            left, right = st.columns([1, 2])

            with left:

                st.metric(
                    "Final Score",
                    f"{profile['score']:.1f}%"
                )

                st.progress(
                    profile["score"] / 100
                )
                
                st.subheader("📊 Score Breakdown")
                
                MAX_SCORES = {

                    "Skills": 50,

                    "Experience": 20,

                    "Education": 10,

                    "Projects": 15,

                    "Certifications": 5
                }
                
                for category, value in profile["breakdown"].items():

                        max_value = MAX_SCORES[category]

                        st.write(
                            f"**{category}**   ({value}/{max_value})"
                        )

                        st.progress(
                            value / max_value
                        )
                                    
                
                

                if profile["recommendation"] == "Interview":

                    st.success("🟢 Interview")

                elif profile["recommendation"] == "Hold":

                    st.warning("🟡 Hold")

                else:

                    st.error("🔴 Reject")

            with right:

                st.write("📧", profile["email"])

                st.write("📱", profile["phone"])

                st.write("📍", profile["location"])

                st.write(
                    "💼 Experience:",
                    profile["years_of_experience"],
                    "Years"
                )

                st.write(
                    "🎓 Degree:",
                    profile["education"]["degree"]
                )

                st.write(
                    "🏫 University:",
                    profile["education"]["university"]
                )

                st.write(
                    "💻 GitHub:",
                    profile["github"]
                )

                st.write(
                    "🔗 LinkedIn:",
                    profile["linkedin"]
                )

            st.subheader("🛠 Skills")

            cols = st.columns(4)

            for i, skill in enumerate(profile["skills"]):

                cols[i % 4].success(skill)

            st.subheader("✅ Matched Skills")

            cols = st.columns(4)

            for i, skill in enumerate(profile["matched_skills"]):

                cols[i % 4].success(skill)

            st.subheader("❌ Missing Skills")

            cols = st.columns(4)

            for i, skill in enumerate(profile["missing_skills"]):

                cols[i % 4].error(skill)

            if profile["certifications"]:

                st.subheader("📜 Certifications")

                for cert in profile["certifications"]:

                    st.success(cert)

            if profile["languages"]:

                st.subheader("🌍 Languages")

                for lang in profile["languages"]:

                    st.info(lang)

            if profile["projects"]:

                st.subheader("📂 Projects")

                for project in profile["projects"]:

                    st.markdown(
                        f"### {project['name']}"
                    )

                    st.write(
                        project["description"]
                    )

                    tech_cols = st.columns(4)

                    for i, tech in enumerate(
                        project["technologies"]
                    ):

                        tech_cols[i % 4].success(tech)

            st.divider()

            if st.button(
                "🤖 AI Evaluation",
                key=f"eval_{profile['name']}"
            ):

                with st.spinner(
                    "Evaluating Candidate..."
                ):

                    report = evaluate_candidate(
                        profile,
                        job
                    )
                    
                st.subheader("📊 Why This Score?")

                st.info(
                    report["score_explanation"]
                )    

                st.subheader(
                    "Overall Assessment"
                )

                st.write(
                    report["overall_assessment"]
                )

                st.subheader(
                    "Technical Fit"
                )

                st.write(
                    report["technical_fit"]
                )

                st.subheader("Strengths")

                for item in report["strengths"]:

                    st.success(item)

                st.subheader("Weaknesses")

                for item in report["weaknesses"]:

                    st.error(item)

                st.subheader(
                    "Interview Questions"
                )

                for item in report["interview_questions"]:

                    st.info(item)

                st.subheader(
                    "Recommendation"
                )

                st.success(
                    report["recommendation"]
                )            
 
# ===============================
# AI HR Chat
# ===============================

elif selected == "AI HR Chat":

    st.header("🤖 AI HR Assistant")

    st.write(
        "Ask anything about your candidates."
    )
    
    st.subheader("💡 Suggested Questions")

    col1, col2 = st.columns(2)

    with col1:

            if st.button(
                "🏆 Who is the best candidate?"
            ):

                st.session_state["preset_question"] = (
                    "Who is the best candidate?"
                )

            if st.button(
                "🐍 Which candidates know Python?"
            ):

                st.session_state["preset_question"] = (
                    "Which candidates know Python?"
                )

            if st.button(
                "🎯 Why was the top candidate recommended?"
            ):

                st.session_state["preset_question"] = (
                    "Why was the top candidate recommended?"
                )

    with col2:

            if st.button(
                "⚖️ Compare the top two candidates"
            ):

                st.session_state["preset_question"] = (
                    "Compare the top two candidates."
                )

            if st.button(
                "📜 Show candidates with AWS certification"
            ):

                st.session_state["preset_question"] = (
                    "Show candidates with AWS certification."
                )

            if st.button(
                "💼 Which candidate has the most experience?"
            ):

                st.session_state["preset_question"] = (
                    "Which candidate has the most experience?"
                )
    
    
    
    
    
    

    if "candidates" not in st.session_state:

        st.warning(
            "Please analyze candidates first."
        )

        st.stop() 
        
    if "messages" not in st.session_state:

          st.session_state.messages = []
            
    for message in st.session_state.messages:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )         
                
    question = st.chat_input(
    "Ask anything about the candidates..."
     )

    if not question and "preset_question" in st.session_state:

            question = st.session_state.pop(
                "preset_question"
            )  
                      
    
    if not question:
        st.stop()
        
    st.session_state.messages.append(

                {
                    "role": "user",
                    "content": question
                }
    )

    with st.chat_message("user"):

        st.write(question)    
        
    ####  Build Context
    
    if "retriever" not in st.session_state:

        st.error(
            "Please analyze the CVs first."
        )

        st.stop()

    retriever = st.session_state.retriever
    docs = retriever.invoke(question)

    with st.expander("Retrieved Context"):

        for doc in docs:

            st.markdown("---")

            st.write(doc.page_content)

            st.caption(doc.metadata)

    context = ""

    sources = []

    for doc in docs:

        context += doc.page_content + "\n\n"

        source = (

            f"{doc.metadata.get('source')} "

            f"(Page {doc.metadata.get('page',0)+1})"

        )

        if source not in sources:

            sources.append(source)
    
    
    prompt = f"""
        You are an Enterprise AI HR Assistant.

        You MUST answer ONLY using the retrieved context below.

        If the answer does not exist in the data, reply:

        "I couldn't find that information in the uploaded candidates."

        You are allowed to:

        - Compare candidates
        - Recommend candidates
        - Explain scores
        - Explain strengths
        - Explain weaknesses
        - Search by skills
        - Search by certifications
        - Search by experience
        - Compare projects
        - Explain why a candidate ranked higher

        Context

        {context}

        Answer ONLY from this context.

        If the answer is not found, reply:

        "I couldn't find this information in the uploaded CVs."

        User Question:

        {question}
        """
 
    
    with st.spinner("Thinking..."):

        response = client.chat.completions.create(

            model=MODEL_NAME,

            temperature=0,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

    answer = response.choices[0].message.content
     
     
    st.session_state.messages.append(

        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):

        st.markdown(answer)
        
    
    st.divider()

    st.subheader("📚 Sources")

    for source in sources:

        for i, source in enumerate(sources, start=1):

            st.caption(f"{i}. {source}")
        