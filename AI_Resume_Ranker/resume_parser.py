import pdfplumber


def extract_text_from_pdf(pdf_path):

    text = ""

    try:

        with pdfplumber.open(pdf_path) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text(
                    x_tolerance=2,
                    y_tolerance=2
                )

                if page_text:

                    text += page_text + "\n"


    except Exception as e:

        raise Exception(
            f"Unable to read PDF: {str(e)}"
        )


    # Clean extracted text
    text = text.replace("\x00", " ")

    text = text.replace("\ufeff", " ")

    # Fix common PDF extraction problems
    text = text.replace(
        "Google Docs/Sheets/Slides",
        "Google Docs Google Sheets Google Slides"
    )

    text = text.replace(
        "Google Docs, Sheets, Slides",
        "Google Docs Google Sheets Google Slides"
    )

    text = text.replace(
        "Google Docs / Sheets / Slides",
        "Google Docs Google Sheets Google Slides"
    )

    text = text.replace(
        "Microsoft Office (Word, Excel, PowerPoint)",
        "Microsoft Office MS Word MS Excel MS PowerPoint"
    )


    return text