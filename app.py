import streamlit as st

import pandas as pd

import json

from ortools.sat.python import cp_model

import google.generativeai as genai



# Setup page layout

st.set_page_config(page_title="LLM AI Project Matcher", layout="wide")



# -------------------------------------------------------------

# 1. API KEY CONFIGURATION IN SIDEBAR

# -------------------------------------------------------------

st.sidebar.title("Settings & Navigation")

api_key = st.sidebar.text_input("Enter Gemini API Key (Optional)", type="password")



if api_key:

    genai.configure(api_key=api_key)

    st.sidebar.success("Gemini API Linked!")

else:

    st.sidebar.warning("Running in Demo Mode. Add API key for Live LLM parsing.")



role = st.sidebar.radio("Go to:", ["Student Registration", "Faculty Dashboard", "Matching Engine"])



# -------------------------------------------------------------

# 2. IN-MEMORY DATABASE

# -------------------------------------------------------------

if "students" not in st.session_state:

    st.session_state.students = [

        {"name": "Alice", "skills": ["Python", "Backend"], "avail": ["Mon AM"]},

        {"name": "Bob", "skills": ["React", "Frontend"], "avail": ["Mon AM"]},

        {"name": "Charlie", "skills": ["Design", "UI/UX"], "avail": ["Tue PM"]},

        {"name": "Diana", "skills": ["Python", "Data Science"], "avail": ["Tue PM"]}

    ]



if "projects" not in st.session_state:

    st.session_state.projects = [

        {"title": "E-Commerce System", "req_skills": ["Frontend", "Backend"]},

        {"title": "AI Dashboard", "req_skills": ["Python", "Data Science"]}

    ]



if "assignments" not in st.session_state:

    st.session_state.assignments = {}



# -------------------------------------------------------------

# LLM HELPER FUNCTION

# -------------------------------------------------------------

def parse_profile_with_llm(raw_text):

    """Uses LLM to clean messy natural language into structured skill arrays"""

    prompt = f"""

    Analyze the following student profile biography text and extract their primary core skills as a clean JSON list.

    Choose ONLY from these standard tags: ["Python", "Backend", "React", "Frontend", "Design", "UI/UX", "Data Science"].

   

    Profile Text: "{raw_text}"

   

    Return ONLY a valid JSON array of strings, like this: ["Python", "Backend"]. Do not include markdown formatting or extra text.

    """

    if not api_key:

        # Fallback Mock LLM Logic for demonstration if no API key is provided

        if "data" in raw_text.lower() or "python" in raw_text.lower():

            return ["Python", "Data Science"]

        return ["React", "Frontend"]

       

    try:

        model = genai.GenerativeModel('gemini-2.5-flash')

        response = model.generate_content(prompt)

        # Clean potential markdown wrappers from the LLM output

        clean_text = response.text.replace("```json", "").replace("```", "").strip()

        return json.loads(clean_text)

    except Exception as e:

        st.error(f"LLM Error: {e}. Using fallback tagger.")

        return ["Backend"]



# -------------------------------------------------------------

# 3. INTERFACE 1: STUDENT REGISTRATION (WITH LLM PARSING)

# -------------------------------------------------------------

if role == "Student Registration":

    st.header("LLM-Powered Student Profile Setup")

    st.write("Type your profile organically. The LLM will extract structured skills automatically.")

   

    with st.form("student_form", clear_on_submit=True):

        name = st.text_input("Full Name")

        bio = st.text_area("Tell us about yourself (e.g., 'I built a website using React last year and love UI/UX design')")

        avail = st.multiselect("Select Available Slots", ["Mon AM", "Mon PM", "Tue AM", "Tue PM"])

       

        submit = st.form_submit_button("Submit Profile to AI Analyzer")

        if submit and name and bio:

            with st.spinner("LLM is parsing your skills..."):

                extracted_skills = parse_profile_with_llm(bio)

                st.session_state.students.append({

                    "name": name,

                    "skills": extracted_skills,

                    "avail": avail

                })

            st.success(f"Profile saved! LLM extracted these skills: {extracted_skills}")



    st.subheader("Currently Registered Students")

    st.dataframe(pd.DataFrame(st.session_state.students))



# -------------------------------------------------------------

# 4. INTERFACE 2: FACULTY DASHBOARD

# -------------------------------------------------------------

elif role == "Faculty Dashboard":

    st.header("Faculty Project Management")

   

    with st.form("project_form", clear_on_submit=True):

        title = st.text_input("Project Title")

        req_skills = st.multiselect("Required Core Skills", ["Python", "Backend", "React", "Frontend", "Design", "UI/UX", "Data Science"])

       

        submit = st.form_submit_button("Create Project")

        if submit and title:

            st.session_state.projects.append({"title": title, "req_skills": req_skills})

            st.success(f"Project '{title}' added!")



    st.subheader("Current Projects")

    st.dataframe(pd.DataFrame(st.session_state.projects))



# -------------------------------------------------------------

# 5. INTERFACE 3: MATCHING ENGINE (ALGORITHM LAYER)

# -------------------------------------------------------------

elif role == "Matching Engine":

    st.header("Hybrid Optimization & Matcher Engine")

   

    if st.button("Run Team Optimization Matcher", type="primary"):

        students = st.session_state.students

        projects = st.session_state.projects

       

        if not students or not projects:

            st.error("Please add data first.")

        else:

            model = cp_model.CpModel()

            x = {}

            for s in range(len(students)):

                for p in range(len(projects)):

                    x[s, p] = model.NewBoolVar(f'x_{s}_{p}')

           

            for s in range(len(students)):

                model.Add(sum(x[s, p] for p in range(len(projects))) == 1)

               

            ideal_size = len(students) // len(projects)

            for p in range(len(projects)):

                model.Add(sum(x[s, p] for s in range(len(students))) >= ideal_size)

                model.Add(sum(x[s, p] for s in range(len(students))) <= ideal_size + 1)



            objective_terms = []

            for s in range(len(students)):

                for p in range(len(projects)):

                    skill_match = len(set(students[s]["skills"]) & set(projects[p]["req_skills"]))

                    weight = skill_match * 10 + 1

                    objective_terms.append(x[s, p] * weight)

                   

            model.Maximize(sum(objective_terms))

            solver = cp_model.CpSolver()

            status = solver.Solve(model)

           

            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:

                st.session_state.assignments = {p["title"]: [] for p in projects}

                for s in range(len(students)):

                    for p in range(len(projects)):

                        if solver.Value(x[s, p]) == 1:

                            st.session_state.assignments[projects[p]["title"]].append(students[s]["name"])

                st.success("Teams optimized successfully!")

            else:

                st.error("Optimization failed.")



    if st.session_state.assignments:

        st.subheader("Generated Teams Matrix")

        for proj_title, team_members in st.session_state.assignments.items():

            with st.expander(f"📋 {proj_title}", expanded=True):

                for member in team_members:

                    st.write(f"- {member}") 

