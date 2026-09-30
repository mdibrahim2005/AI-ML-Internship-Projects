from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename

from resume_parser import extract_text_from_pdf
from ranker import analyze_resume

from database import (
    initialize_database,
    create_session,
    save_candidate_result,
    get_all_sessions,
    get_session,
    get_session_results
)

import os
import uuid
import csv
import io
import json


app = Flask(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

UPLOAD_FOLDER = "resumes"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Maximum total upload size
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# DATABASE
# ============================================================

initialize_database()


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def is_real_pdf(filepath):

    try:

        with open(filepath, "rb") as file:

            header = file.read(5)

        return header == b"%PDF-"

    except Exception:

        return False


# ============================================================
# HOME / RESUME RANKING
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    results = []
    message = None

    if request.method == "POST":

        files = request.files.getlist("resumes")

        job_description = request.form.get(
            "job_description",
            ""
        ).strip()


        # ----------------------------------------------------
        # CHECK RESUMES
        # ----------------------------------------------------

        if not files or all(
            file.filename == ""
            for file in files
        ):

            message = "Please upload at least one resume PDF."

            return render_template(
                "index.html",
                results=[],
                message=message
            )


        # ----------------------------------------------------
        # CHECK JOB DESCRIPTION
        # ----------------------------------------------------

        if not job_description:

            message = "Please enter a job description."

            return render_template(
                "index.html",
                results=[],
                message=message
            )


        # ----------------------------------------------------
        # ANALYZE RESUMES
        # ----------------------------------------------------

        for file in files:

            if file.filename == "":
                continue


            # Only PDF files
            if not allowed_file(file.filename):

                print(
                    f"Skipped non-PDF file: {file.filename}"
                )

                continue


            # Secure original filename
            original_filename = secure_filename(
                file.filename
            )


            if not original_filename:
                continue


            # Create unique temporary filename
            unique_filename = (
                f"{uuid.uuid4().hex}_{original_filename}"
            )


            filepath = os.path.join(
                UPLOAD_FOLDER,
                unique_filename
            )


            try:

                # Save uploaded file temporarily
                file.save(filepath)


                # Check actual PDF signature
                if not is_real_pdf(filepath):

                    print(
                        f"Skipped invalid PDF: "
                        f"{original_filename}"
                    )

                    continue


                # Extract resume text
                resume_text = extract_text_from_pdf(
                    filepath
                )


                # Check extracted text
                if not resume_text.strip():

                    print(
                        f"Skipped empty PDF: "
                        f"{original_filename}"
                    )

                    continue


                # AI resume analysis
                result = analyze_resume(
                    resume_text,
                    job_description
                )


                # Keep original filename
                result["filename"] = original_filename

                results.append(result)


            except Exception as e:

                print(
                    f"Error analyzing "
                    f"{original_filename}: {e}"
                )


            finally:

                # Delete temporary uploaded file
                if os.path.exists(filepath):

                    try:

                        os.remove(filepath)

                    except Exception as e:

                        print(
                            f"Could not delete file: {e}"
                        )


        # ----------------------------------------------------
        # SORT RESULTS
        # ----------------------------------------------------

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )


        # ----------------------------------------------------
        # SAVE RESULTS TO DATABASE
        # ----------------------------------------------------

        if results:

            session_id = create_session(
                job_description,
                len(results)
            )


            for result in results:

                save_candidate_result(
                    session_id,
                    result
                )


        else:

            message = (
                "No valid PDF resumes could be analyzed."
            )


    return render_template(
        "index.html",
        results=results,
        message=message
    )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    sessions = get_all_sessions()

    return render_template(
        "history.html",
        sessions=sessions
    )


# ============================================================
# SAVED SESSION
# ============================================================

@app.route("/history/<int:session_id>")
def view_session(session_id):

    session = get_session(session_id)

    if not session:

        return "Ranking session not found.", 404


    results = get_session_results(
        session_id
    )


    return render_template(
        "session.html",
        session=session,
        results=results
    )


# ============================================================
# CSV EXPORT
# ============================================================

@app.route("/export_csv", methods=["POST"])
def export_csv():

    results_json = request.form.get(
        "results_json",
        ""
    )


    if not results_json:

        return "No ranking results available.", 400


    try:

        results = json.loads(
            results_json
        )

    except Exception:

        return "Unable to process ranking results.", 400


    output = io.StringIO()

    writer = csv.writer(output)


    # CSV headings
    writer.writerow([
        "Rank",
        "Resume",
        "Overall Score",
        "Recommendation",
        "Skill Match",
        "AI Relevance",
        "Job Title Match",
        "Experience Match",
        "Matching Skills",
        "Missing Skills"
    ])


    # CSV data
    for index, result in enumerate(
        results,
        start=1
    ):

        writer.writerow([
            index,
            result.get(
                "filename",
                ""
            ),
            result.get(
                "score",
                ""
            ),
            result.get(
                "recommendation",
                ""
            ),
            result.get(
                "skill_score",
                ""
            ),
            result.get(
                "text_score",
                ""
            ),
            result.get(
                "title_score",
                ""
            ),
            result.get(
                "experience_score",
                ""
            ),
            ", ".join(
                result.get(
                    "matching_skills",
                    []
                )
            ),
            ", ".join(
                result.get(
                    "missing_skills",
                    []
                )
            )
        ])


    csv_data = output.getvalue()

    output.close()


    return Response(
        csv_data,
        mimetype="text/csv",
        headers={
            "Content-Disposition":
                "attachment; "
                "filename=resume_ranking_results.csv"
        }
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run()