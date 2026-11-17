import re
import html
import os
from pathlib import Path

import streamlit as st
import pymupdf
from docx import Document
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# INSTELLINGEN
# --------------------------------------------------

BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "kennisbank"

load_dotenv(BASE_DIR / ".env")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MAX_RESULTS = 3

SYSTEM_INSTRUCTIONS = """
Je bent de digitale museumassistent van het Maritiem Museum Rotterdam.

Je helpt bezoekers met vragen over de collectie, tentoonstellingen,
activiteiten en hun museumbezoek.

Regels:
- Antwoord in het Nederlands, tenzij de bezoeker een andere taal gebruikt.
- Gebruik de aangeleverde museumdocumenten als belangrijkste informatiebron.
- Verzin geen feiten, openingstijden, prijzen of activiteiten.
- Staat het antwoord niet in de documenten? Zeg dat eerlijk.
- Schrijf vriendelijk, duidelijk en begrijpelijk.
- Verwijs waar mogelijk naar de naam van het document waarin informatie staat.
- Maak duidelijk onderscheid tussen informatie uit de documenten en algemene kennis.
"""


# --------------------------------------------------
# DOCUMENTEN INLEZEN
# --------------------------------------------------

def read_pdf(path):
    """Lees tekst uit een PDF-bestand."""
    parts = []

    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                parts.append(
                    f"[Pagina {page_number}]\n{text}"
                )

    return "\n\n".join(parts)


def read_docx(path):
    """Lees tekst uit een Word-document, inclusief tabellen."""
    doc = Document(path)
    parts = []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()

        if text:
            parts.append(text)

    for table in doc.tables:
        for row in table.rows:
            cells = [
                cell.text.strip().replace("\n", " ")
                for cell in row.cells
            ]
            parts.append(" | ".join(cells))

    return "\n".join(parts)


def read_document(path):
    """Kies de juiste methode op basis van de bestandsextensie."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return read_pdf(path)

    if suffix == ".docx":
        return read_docx(path)

    return ""


# --------------------------------------------------
# TEKST OPSPLITSEN IN KLEINERE STUKKEN
# --------------------------------------------------

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Verdeel lange documenten in overlappende tekstfragmenten."""
    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Probeer niet midden in een woord te eindigen.
        if end < len(text):
            space_position = text.rfind(" ", start, end)

            if space_position > start:
                end = space_position

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def build_knowledge_base():
    """Lees alle ondersteunde bestanden en maak zoekbare fragmenten."""
    if not KNOWLEDGE_DIR.exists():
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        path
        for path in KNOWLEDGE_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".pdf", ".docx"}
        and not path.name.startswith("~$")
    )

    documents = []
    errors = []

    for path in files:
        try:
            text = read_document(path)

            if not text.strip():
                errors.append(
                    f"{path.name}: geen leesbare tekst gevonden."
                )
                continue

            fragments = split_text(text)

            for fragment in fragments:
                documents.append(
                    {
                        "source": path.name,
                        "text": fragment,
                    }
                )

        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    vectorizer = None
    matrix = None

    if documents:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        matrix = vectorizer.fit_transform(
            [item["text"] for item in documents]
        )

    return {
        "documents": documents,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "files": files,
        "errors": errors,
    }


# --------------------------------------------------
# RELEVANTE INFORMATIE ZOEKEN
# --------------------------------------------------

def find_relevant_fragments(question, knowledge_base):
    """Zoek de tekstfragmenten die het beste bij de vraag passen."""
    documents = knowledge_base["documents"]
    vectorizer = knowledge_base["vectorizer"]
    matrix = knowledge_base["matrix"]

    if not documents or vectorizer is None or matrix is None:
        return []

    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, matrix).ravel()

    best_indices = scores.argsort()[::-1][:MAX_RESULTS]

    results = []

    for index in best_indices:
        score = float(scores[index])

        # Sla resultaten zonder enige overeenkomst over.
        if score <= 0:
            continue

        item = documents[index]

        results.append(
            {
                "source": item["source"],
                "text": item["text"],
                "score": score,
            }
        )

    return results


# --------------------------------------------------
# SCHELDWOORDEN HERKENNEN EN AFDEKKEN
# --------------------------------------------------

BAD_WORDS = [
    # Nederlandse scheldwoorden
    "klootzak", "klootzakken", "lul", "lullen", "eikel", "eikels",
    "sukkel", "sukkels", "idioot", "idioten", "debiel", "mongool",
    "achterlijke", "mafkees", "tering", "tyfus", "godverdomme",
    "godver", "kanker", "kankerlijer", "kankerhoer", "kankerkop",
    "kut", "kuthoer", "kutwijf", "kutkind", "kutzooi", "bitch",
    "wijf", "hoer", "slet", "trut", "hufter", "etter", "rotzak",
    "smeerlap", "flikker", "teringlijer", "klootviool", "sodemieter",

    # Engelse scheldwoorden
    "faggot", "motherfucker", "asshole", "fuck", "fucking", "shit",
    "bullshit", "bastard", "dickhead", "dumbass", "moron", "prick",
    "son of a bitch", "piece of shit", "bitch", "bastard",

    # Racistische en discriminerende beledigingen
    "nigger", "nigga", "kaffer", "nikker",

    # Veelvoorkomende alternatieve schrijfwijzen
    "k4nker", "k@nker", "k anker", "k.u.t", "k u t",
    "f u c k", "sh1t", "n1gger", "n!gger",
]


def contains_bad_word(text):
    """Controleer of tekst een term uit de blokkeerlijst bevat."""
    for word in sorted(BAD_WORDS, key=len, reverse=True):
        pattern = r"(?<!\\w)" + re.escape(word) + r"(?!\\w)"
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def blur_bad_words(text):
    """Vervang herkende termen door blokjes."""
    for word in sorted(BAD_WORDS, key=len, reverse=True):
        pattern = r"(?<!\\w)" + re.escape(word) + r"(?!\\w)"
        text = re.sub(
            pattern,
            lambda match: "█" * len(match.group(0)),
            text,
            flags=re.IGNORECASE,
        )
    return text


# --------------------------------------------------
# GEMINI ANTWOORD LATEN GENEREREN
# --------------------------------------------------

def generate_answer(question, fragments, client):
    """Laat Gemini antwoorden op basis van gevonden museuminformatie."""

    # Extra controle voordat de vraag naar Gemini gaat.
    if contains_bad_word(question):
        return "Je bericht bevat ongepaste taal. Daar kan ik geen antwoord op geven."

    # Herken eenvoudige begroetingen.
    if question.strip().lower() in [
        "hoi", "hallo", "hey", "goedemorgen",
        "goedemiddag", "goedenavond"
    ]:
        return (
            "Hoi! Welkom bij het Maritiem Museum Rotterdam. "
            "Ik kan je helpen met vragen over de collectie, "
            "tentoonstellingen, activiteiten en je museumbezoek. "
            "Wat wil je weten?"
        )

    if not fragments:
        return (
            "Ik kan het antwoord niet terugvinden in de ingelezen "
            "museumdocumenten. Probeer je vraag anders te formuleren "
            "of raadpleeg het museum voor meer informatie."
        )

    context_parts = []
    for fragment in fragments:
        context_parts.append(
            f"Bron: {fragment['source']}\\n"
            f"Inhoud:\\n{fragment['text']}"
        )

    context = "\\n\\n---\\n\\n".join(context_parts)

    prompt = f"""
Beantwoord de vraag van de bezoeker met behulp van de informatie hieronder.

Als de informatie onvoldoende is, geef dat eerlijk aan.
Verzin geen ontbrekende informatie.

MUSEUMINFORMATIE:
{context}

VRAAG VAN DE BEZOEKER:
{question}
"""

    print("TEST: Gemini-aanvraag wordt gestart...")

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTIONS,
                temperature=0.2,
                max_output_tokens=800,
            ),
        )
        answer = response.text or ""
        print(f"TEST: Gemini-antwoord ontvangen: {answer}")

    except Exception as e:
        print(f"Fout bij Gemini: {e}")
        return (
            "Het lukt momenteel niet om een antwoord van de AI te krijgen. "
            "Probeer het over een minuut opnieuw."
        )

    if not answer:
        return "Ik kon helaas geen antwoord maken. Probeer het opnieuw."

    sources = list(dict.fromkeys(
        fragment["source"] for fragment in fragments
    ))
    source_text = "\\n".join(f"- {source}" for source in sources)

    return f"{answer}\\n\\n**Gebruikte bronnen:**\\n{source_text}"


# --------------------------------------------------
# STREAMLIT-WEBSITE
# --------------------------------------------------

st.set_page_config(
    page_title="Maritiem Museum Rotterdam",
    page_icon="⚓",
    layout="centered",
)

st.title("⚓ Maritiem Museum Rotterdam")

st.write(
    "Stel een vraag over de collectie, tentoonstellingen, "
    "activiteiten of je museumbezoek."
)

api_key = os.getenv("GEMINI_API_KEY", "").strip()

if not api_key:
    st.error(
        "De Gemini API-sleutel ontbreekt. Controleer of je in het "
        ".env-bestand GEMINI_API_KEY hebt ingesteld."
    )
    st.code("GEMINI_API_KEY=JOUW_EIGEN_SLEUTEL")
    st.stop()

client = genai.Client(
    api_key=api_key,
    http_options={"timeout": 20000},
)


# Maak de kennisbank bij de eerste start.
if "knowledge_base" not in st.session_state:
    with st.spinner("De museumdocumenten worden ingelezen..."):
        st.session_state.knowledge_base = build_knowledge_base()


knowledge_base = st.session_state.knowledge_base

with st.sidebar:
    st.header("Kennisbank")

    st.write(
        f"**Documenten gevonden:** "
        f"{len(knowledge_base['files'])}"
    )

    st.write(
        f"**Tekstfragmenten:** "
        f"{len(knowledge_base['documents'])}"
    )

    if knowledge_base["files"]:
        with st.expander("Ingelezen bestanden"):
            for path in knowledge_base["files"]:
                st.write(f"- {path.name}")

    if knowledge_base["errors"]:
        with st.expander("Meldingen bij het inlezen"):
            for error in knowledge_base["errors"]:
                st.warning(error)

    if st.button("Kennisbank opnieuw inlezen", use_container_width=True):
        with st.spinner("Documenten opnieuw inlezen..."):
            st.session_state.knowledge_base = build_knowledge_base()

        st.rerun()

    st.caption(f"AI-model: {MODEL_NAME}")


if not knowledge_base["documents"]:
    st.warning(
        "Er zijn nog geen leesbare documenten gevonden. "
        "Plaats PDF- of DOCX-bestanden in de map 'kennisbank' "
        "en klik op 'Kennisbank opnieuw inlezen'."
    )


# Bewaar de chat tijdens de huidige sessie.
if "messages" not in st.session_state:
    st.session_state.messages = []


# Toon eerdere berichten.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Vraag invoeren.
question = st.chat_input("Stel je vraag aan het museum...")


if question:
    # Controleer vóór we het bericht tonen of opslaan.
    if contains_bad_word(question):
        blurred_question = blur_bad_words(question)
        refusal = (
            "Je bericht bevat ongepaste taal. "
            "Daar kan ik geen antwoord op geven."
        )

        st.session_state.messages.append(
            {"role": "user", "content": blurred_question}
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": refusal}
        )

        with st.chat_message("user"):
            st.markdown(blurred_question)

        with st.chat_message("assistant"):
            st.warning(refusal)

    else:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Ik zoek de juiste museuminformatie erbij..."):
                    fragments = find_relevant_fragments(
                        question,
                        st.session_state.knowledge_base,
                    )

                    answer = generate_answer(
                        question,
                        fragments,
                        client,
                    )

                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as exc:
                error_message = str(exc)

                st.error(
                    "Er ging iets mis bij het maken van het antwoord. "
                    "Controleer de foutmelding hieronder."
                )
                st.code(error_message)

                st.caption(
                    "De API-sleutel is niet opgenomen in deze "
                    "foutweergave door de applicatiecode."
                )
