import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer


# ============================================================
# AI SEMANTIC MODEL
# ============================================================

semantic_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# SKILL ALIASES
# ============================================================

SKILL_ALIASES = {

    # Programming
    "python": [
        "python"
    ],

    "java": [
        "java"
    ],

    "c++": [
        "c++",
        "cpp",
        "cplusplus"
    ],

    "sql": [
        "sql"
    ],

    # AI / ML
    "artificial intelligence": [
        "artificial intelligence",
        "ai"
    ],

    "machine learning": [
        "machine learning",
        "machine-learning",
        "ml"
    ],

    "deep learning": [
        "deep learning",
        "deep-learning",
        "dl"
    ],

    "data science": [
        "data science"
    ],

    "data analysis": [
        "data analysis",
        "data analytics"
    ],

    # Libraries / Frameworks
    "pandas": [
        "pandas"
    ],

    "numpy": [
        "numpy"
    ],

    "scikit-learn": [
        "scikit-learn",
        "scikit learn",
        "sklearn"
    ],

    "tensorflow": [
        "tensorflow"
    ],

    "pytorch": [
        "pytorch"
    ],

    "flask": [
        "flask"
    ],

    "django": [
        "django"
    ],

    # Web
    "html": [
        "html",
        "html5"
    ],

    "css": [
        "css",
        "css3"
    ],

    "javascript": [
        "javascript",
        "java script",
        "js"
    ],

    "react": [
        "react",
        "reactjs",
        "react.js"
    ],

    # Tools
    "git": [
        "git"
    ],

    "github": [
        "github"
    ],

    # Microsoft Office
    "microsoft office": [
        "microsoft office",
        "ms office"
    ],

    "ms word": [
        "ms word",
        "microsoft word",
        "word processing"
    ],

    "ms excel": [
        "ms excel",
        "microsoft excel",
        "spreadsheet"
    ],

    "ms powerpoint": [
        "ms powerpoint",
        "microsoft powerpoint",
        "presentation software"
    ],

    # Google Workspace
    "google docs": [
        "google docs",
        "google doc",
        "google documents",
        "google doc sheet and slides",
        "google docs sheets slides",
        "google docs sheets and slides"
    ],

    "google sheets": [
        "google sheets",
        "google sheet",
        "google spreadsheets",
        "google doc sheet and slides",
        "google docs sheets slides",
        "google docs sheets and slides"
    ],

    "google slides": [
        "google slides",
        "google slide",
        "google presentations",
        "google doc sheet and slides",
        "google docs sheets slides",
        "google docs sheets and slides"
    ],

    # Office skills
    "data entry": [
        "data entry",
        "data-entry",
        "dataentry"
    ],

    "documentation": [
        "documentation",
        "document management",
        "document handling"
    ],

    "file management": [
        "file management",
        "manage files",
        "managing files",
        "file organization",
        "file handling"
    ],

    "office administration": [
        "office administration",
        "office management",
        "administrative work",
        "administration"
    ],

    # Creative
    "social media management": [
        "social media management",
        "social media"
    ],

    "video editing": [
        "video editing",
        "video editor",
        "video-editing"
    ],

    "graphic design": [
        "graphic design",
        "graphic designing",
        "designing"
    ],

    "communication": [
        "communication",
        "communication skills",
        "good communication",
        "effective communication"
    ]
}


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    if not text:
        return ""

    text = text.lower()

    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = text.replace("\\", " ")

    text = text.replace("c++", "cplusplus")

    text = text.replace("-", " ")

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# EXTRACT SKILLS
# ============================================================

def extract_skills(text):

    text = normalize_text(text)

    found_skills = []

    for skill, aliases in SKILL_ALIASES.items():

        for alias in aliases:

            alias = normalize_text(alias)

            if not alias:
                continue

            pattern = (
                r"(?<!\w)"
                + re.escape(alias)
                + r"(?!\w)"
            )

            if re.search(pattern, text):

                found_skills.append(skill)

                break

    return found_skills


# ============================================================
# WORD TF-IDF SIMILARITY
# ============================================================

def word_similarity(
    resume_text,
    job_description
):

    documents = [
        resume_text,
        job_description
    ]

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        matrix = vectorizer.fit_transform(
            documents
        )

        score = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        return score * 100

    except ValueError:

        return 0


# ============================================================
# CHARACTER SIMILARITY
# ============================================================

def character_similarity(
    resume_text,
    job_description
):

    documents = [
        resume_text,
        job_description
    ]

    try:

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True
        )

        matrix = vectorizer.fit_transform(
            documents
        )

        score = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        return score * 100

    except ValueError:

        return 0


# ============================================================
# AI SEMANTIC SIMILARITY
# ============================================================

def semantic_similarity(
    resume_text,
    job_description
):

    try:

        embeddings = semantic_model.encode(
            [
                resume_text,
                job_description
            ],
            normalize_embeddings=True
        )

        score = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]

        return max(
            0,
            min(score * 100, 100)
        )

    except Exception:

        return 0


# ============================================================
# COMBINED TEXT RELEVANCE
# ============================================================

def calculate_text_similarity(
    resume_text,
    job_description
):

    resume_text = normalize_text(
        resume_text
    )

    job_description = normalize_text(
        job_description
    )

    if not resume_text or not job_description:

        return 0

    word_score = word_similarity(
        resume_text,
        job_description
    )

    character_score = character_similarity(
        resume_text,
        job_description
    )

    semantic_score = semantic_similarity(
        resume_text,
        job_description
    )

    final_text_score = (

        semantic_score * 0.60

        +

        word_score * 0.30

        +

        character_score * 0.10

    )

    return min(
        final_text_score,
        100
    )


# ============================================================
# RELATED SKILL GROUPS
# ============================================================

SKILL_GROUPS = {

    "artificial intelligence": [
        "artificial intelligence",
        "machine learning",
        "deep learning"
    ],

    "machine learning": [
        "machine learning",
        "artificial intelligence",
        "deep learning"
    ],

    "deep learning": [
        "deep learning",
        "machine learning",
        "artificial intelligence"
    ],

    "data science": [
        "data science",
        "data analysis"
    ],

    "data analysis": [
        "data analysis",
        "data science"
    ],

    "microsoft office": [
        "microsoft office",
        "ms word",
        "ms excel",
        "ms powerpoint"
    ],

    "ms word": [
        "ms word",
        "microsoft office"
    ],

    "ms excel": [
        "ms excel",
        "microsoft office"
    ],

    "ms powerpoint": [
        "ms powerpoint",
        "microsoft office"
    ]
}


# ============================================================
# CHECK RELATED SKILLS
# ============================================================

def skills_are_related(
    resume_skill,
    job_skill
):

    if resume_skill == job_skill:

        return True

    if job_skill in SKILL_GROUPS:

        if resume_skill in SKILL_GROUPS[job_skill]:

            return True

    if resume_skill in SKILL_GROUPS:

        if job_skill in SKILL_GROUPS[resume_skill]:

            return True

    return False


# ============================================================
# FIND MATCHING SKILLS
# ============================================================

def find_matching_skills(
    resume_skills,
    job_skills
):

    matching_skills = []

    for job_skill in job_skills:

        for resume_skill in resume_skills:

            if skills_are_related(
                resume_skill,
                job_skill
            ):

                if job_skill not in matching_skills:

                    matching_skills.append(
                        job_skill
                    )

                break

    return matching_skills


# ============================================================
# FIND MISSING SKILLS
# ============================================================

def find_missing_skills(
    resume_skills,
    job_skills
):

    missing_skills = []

    for job_skill in job_skills:

        found = False

        for resume_skill in resume_skills:

            if skills_are_related(
                resume_skill,
                job_skill
            ):

                found = True

                break

        if not found:

            missing_skills.append(
                job_skill
            )

    return missing_skills


# ============================================================
# JOB TITLE RELEVANCE
# ============================================================

def calculate_title_relevance(
    resume_text,
    job_description
):

    resume_text = normalize_text(
        resume_text
    )

    job_description = normalize_text(
        job_description
    )

    job_titles = [

        "data entry operator",
        "computer operator",
        "data entry",

        "software developer",
        "python developer",
        "web developer",

        "frontend developer",
        "backend developer",
        "full stack developer",

        "data analyst",
        "data scientist",

        "machine learning engineer",
        "ai engineer",

        "software engineer"
    ]

    found_titles = []

    for title in job_titles:

        title_normalized = normalize_text(
            title
        )

        if title_normalized in job_description:

            found_titles.append(
                title_normalized
            )

    if not found_titles:

        return 0

    for title in found_titles:

        if title in resume_text:

            return 100

    return 0


# ============================================================
# EXPERIENCE ANALYSIS
# ============================================================

def calculate_experience_relevance(
    resume_text,
    job_description
):

    resume_text = normalize_text(
        resume_text
    )

    job_description = normalize_text(
        job_description
    )

    experience_patterns = [

        r"(\d+)\+?\s+years?\s+of\s+experience",

        r"(\d+)\+?\s+years?\s+experience",

        r"minimum\s+(\d+)\s+years?",

        r"at least\s+(\d+)\s+years?"
    ]

    required_years = 0

    for pattern in experience_patterns:

        match = re.search(
            pattern,
            job_description
        )

        if match:

            required_years = int(
                match.group(1)
            )

            break

    # No experience requirement
    # means experience should not
    # reduce the score.

    if required_years == 0:

        return 100

    resume_years = []

    for pattern in experience_patterns:

        matches = re.findall(
            pattern,
            resume_text
        )

        for value in matches:

            resume_years.append(
                int(value)
            )

    if not resume_years:

        return 0

    candidate_years = max(
        resume_years
    )

    if candidate_years >= required_years:

        return 100

    experience_score = (

        candidate_years
        /
        required_years

    ) * 100

    return min(
        experience_score,
        100
    )


# ============================================================
# ANALYZE RESUME
# ============================================================

def analyze_resume(
    resume_text,
    job_description
):

    resume_text = normalize_text(
        resume_text
    )

    job_description = normalize_text(
        job_description
    )

    # ========================================================
    # SKILLS
    # ========================================================

    resume_skills = extract_skills(
        resume_text
    )

    job_skills = extract_skills(
        job_description
    )

    matching_skills = find_matching_skills(
        resume_skills,
        job_skills
    )

    missing_skills = find_missing_skills(
        resume_skills,
        job_skills
    )

    # ========================================================
    # SKILL SCORE
    # ========================================================

    if job_skills:

        skill_score = (

            len(matching_skills)
            /
            len(job_skills)

        ) * 100

    else:

        skill_score = 0

    # ========================================================
    # AI TEXT RELEVANCE
    # ========================================================

    text_score = calculate_text_similarity(
        resume_text,
        job_description
    )

    # ========================================================
    # JOB TITLE RELEVANCE
    # ========================================================

    title_score = calculate_title_relevance(
        resume_text,
        job_description
    )

    # ========================================================
    # EXPERIENCE RELEVANCE
    # ========================================================

    experience_score = calculate_experience_relevance(
        resume_text,
        job_description
    )

    # ========================================================
    # FINAL SCORE
    #
    # Skills              = 50%
    # AI text relevance   = 30%
    # Job title           = 10%
    # Experience          = 10%
    # ========================================================

    final_score = (

        skill_score * 0.50

        +

        text_score * 0.30

        +

        title_score * 0.10

        +

        experience_score * 0.10

    )

    final_score = round(
        min(final_score, 100),
        2
    )

    # ========================================================
    # RECOMMENDATION
    # ========================================================

    if final_score >= 80:

        recommendation = "Excellent match"

    elif final_score >= 65:

        recommendation = "Strong match"

    elif final_score >= 50:

        recommendation = "Moderate match"

    elif final_score >= 35:

        recommendation = "Weak match"

    else:

        recommendation = "Low match"

    # ========================================================
    # RESULT
    # ========================================================

    return {

        "score": final_score,

        "text_score": round(
            text_score,
            2
        ),

        "skill_score": round(
            skill_score,
            2
        ),

        "experience_score": round(
            experience_score,
            2
        ),

        "title_score": round(
            title_score,
            2
        ),

        "resume_skills": resume_skills,

        "job_skills": job_skills,

        "matching_skills": matching_skills,

        "missing_skills": missing_skills,

        "recommendation": recommendation
    }