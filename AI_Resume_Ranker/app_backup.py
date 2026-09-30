from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from resume_parser import extract_text_from_pdf
from ranker import analyze_resume
import os


app = Flask(__name__)

UPLOAD_FOLDER = "resumes"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def home():

    results = []
    message = None

    if request.method == "POST":

        # Get uploaded resumes
        files = request.files.getlist("resumes")

        # Get job description
        job_description = request.form.get(
            "job_description",
            ""
        ).strip()


        # Check resumes
        if not files or all(
            file.filename == ""
            for file in files
        ):

            message = "Please upload at least one resume PDF."

            return render_template(
                "index.html",
                message=message
            )


        # Check job description
        if not job_description:

            message = "Please enter a job description."

            return render_template(
                "index.html",
                message=message
            )


        # Analyze every resume
        for file in files:

            if file.filename == "":
                continue


            # Allow PDF files only
            if not file.filename.lower().endswith(".pdf"):

                continue


            filename = secure_filename(
                file.filename
            )


            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )


            try:

                # Save resume
                file.save(filepath)


                # Extract resume text
                resume_text = extract_text_from_pdf(
                    filepath
                )


                if not resume_text.strip():

                    continue


                # Analyze resume
                result = analyze_resume(
                    resume_text,
                    job_description
                )


                # Add filename
                result["filename"] = filename


                results.append(result)


            except Exception as e:

                print(
                    f"Error analyzing {file.filename}: {e}"
                )


        # Sort candidates by score
        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )


        # No successful results
        if not results:

            message = (
                "No valid resumes could be analyzed."
            )


    return render_template(
        "index.html",
        results=results,
        message=message
    )


if __name__ == "__main__":

    app.run(debug=True)