import os
import json
from datetime import date
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

import pdfplumber
import docx
from langchain.schema import SystemMessage, HumanMessage
from ai_utils import llm_reasoning, extract_json

resume_bp = Blueprint("resume_bp", __name__)

def extract_resume_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    elif ext == '.docx':
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()
    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read().strip()
    else:
        raise ValueError("Unsupported file format. Only .pdf, .docx, and .txt are supported.")

def get_resume_summary_and_ats_score(jd_text, resume_text, llm):
    system_prompt = (
        "You are an AI assistant that parses resume text and evaluates its relevance against a job description (JD)."
    )

    user_prompt = f"""
    [Today's Date: {date.today()}]

    [Job Description (JD)]:
    {jd_text}

    [Resume Text]:
    {resume_text}

    Instructions:
    1. Extract and summarize key candidate information into `user_info`:
    - Name
    - Email (if available)
    - Phone (if available; must include country code, e.g., +917896577010. If location is not provided, assume India for default country code)
    - Location
    - Total Experience (years, inferred)
    - Relevant Experience (to JD)
    - Current/Last Role and Company
    - Skills (key technical and soft skills)
    - Education (highest degree and university)
    - Notable achievements (if any)

    Key IDs (user_info): Name, Email, Phone, Location, Total Experience, Relevant Experience, Current/Last Role and Company, Skills, Education, Notable achievements
    2. Compute an `ats_score` (0–100) evaluating the match between resume and JD. Consider:
    - Skill overlap
    - Experience relevance
    - Role alignment
    - Education fit

    Return only a JSON object with two keys:
    - `user_info`: summarized candidate data
    - `ats_score`: integer between 0–100 indicating resume fit
""".strip()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    try:
        # print('messages: ', messages)
        response = llm.invoke(messages).content.strip()
        # print(response)
        return extract_json(response)
    except Exception as e:
        print(f"Error: {e}")
        return {"user_info": {}, "ats_score": 0}


@resume_bp.route("/parse_resume", methods=["POST"])
@jwt_required()
def parse_resume():
    current_user_id = get_jwt_identity()

    if "resume_file" not in request.files:
        return jsonify({"error": "Missing resume_file"}), 400
    if "job_description" not in request.form:
        return jsonify({"error": "Missing job_description"}), 400

    resume_file = request.files["resume_file"]
    job_description = request.form["job_description"]

    # Save file temporarily
    temp_path = f"/tmp/{resume_file.filename}"
    resume_file.save(temp_path)

    try:
        resume_text = extract_resume_text(temp_path)
        # print(resume_text)
        result = get_resume_summary_and_ats_score(job_description, resume_text, llm_reasoning)
        return jsonify(result), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to parse resume: {e}"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
