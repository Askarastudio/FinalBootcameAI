import pickle
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

st.set_page_config(
    page_title="Smart Expense Categorizer",
    page_icon="💰",
    layout="wide",
)

CATEGORY_ICON_MAP = {
    "Makanan": "🍜",
    "Transport": "🛵",
    "Tagihan": "💡",
    "Belanja": "🛍",
    "Hiburan": "🎮",
    "Kesehatan": "💊",
    "Pendidikan": "📚",
}

CATEGORY_COLOR_MAP = {
    "Makanan": "#FF7A59",
    "Transport": "#00A6A6",
    "Tagihan": "#F4B400",
    "Belanja": "#8E6CFF",
    "Hiburan": "#EC4899",
    "Kesehatan": "#10B981",
    "Pendidikan": "#3B82F6",
}

EXAMPLE_TRANSACTIONS = [
    "beli nasi goreng 20rb",
    "isi bensin motor 30rb",
    "bayar listrik PLN 100rb",
    "beli vitamin c di apotek",
    "bayar uang kuliah semester ini",
]


@st.cache_resource
def load_model():
    """Load model and vectorizer from local pickle files."""
    base_path = Path(__file__).resolve().parent
    model_path = base_path / "model.pkl"
    vectorizer_path = base_path / "vectorizer.pkl"

    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            "Pastikan file model.pkl dan vectorizer.pkl ada di folder project."
        )

    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    with open(vectorizer_path, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return model, vectorizer


def clean_text(text: str) -> str:
    """Lowercase, remove numbers, punctuation, and extra spaces with regex."""
    cleaned = text.lower()
    cleaned = re.sub(r"\d+", " ", cleaned)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"_", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def get_category_icon(category: str) -> str:
    return CATEGORY_ICON_MAP.get(category, "🏷")


def get_category_color(category: str) -> str:
    return CATEGORY_COLOR_MAP.get(category, "#2563EB")


def predict_category(text: str, model, vectorizer):
    """Run preprocessing, vectorization, and model prediction."""
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Teks transaksi tidak valid setelah preprocessing.")

    vector = vectorizer.transform([cleaned])
    category = str(model.predict(vector)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)
        confidence = float(np.max(probabilities) * 100)

    return category, confidence, cleaned


def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "transaction_input" not in st.session_state:
        st.session_state.transaction_input = ""
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    if "pending_example" not in st.session_state:
        st.session_state.pending_example = None


def inject_custom_css():
    # Global style for fintech-like dashboard look.
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at 10% 20%, #f5fbff 0%, #f8f4ff 35%, #eef8f6 100%);
                color: #112233;
                font-family: 'Poppins', 'Trebuchet MS', 'Verdana', sans-serif;
            }

            .hero-card {
                background: linear-gradient(135deg, #0f766e 0%, #2563eb 55%, #9333ea 100%);
                border-radius: 20px;
                padding: 28px;
                box-shadow: 0 18px 35px rgba(37, 99, 235, 0.24);
                margin-bottom: 20px;
                color: white;
            }

            .hero-title {
                font-size: 2.1rem;
                font-weight: 800;
                margin-bottom: 6px;
                letter-spacing: 0.4px;
            }

            .hero-subtitle {
                font-size: 1.02rem;
                opacity: 0.95;
            }

            .section-card {
                background: white;
                border-radius: 18px;
                padding: 20px;
                box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
                border: 1px solid rgba(148, 163, 184, 0.25);
                margin-bottom: 16px;
            }

            .prediction-card {
                border-radius: 18px;
                padding: 18px;
                box-shadow: 0 10px 24px rgba(0, 0, 0, 0.08);
                border-left: 8px solid #2563eb;
                background: #ffffff;
                margin-top: 14px;
                margin-bottom: 12px;
            }

            .prediction-title {
                font-weight: 700;
                font-size: 1.25rem;
                margin-bottom: 8px;
            }

            .prediction-meta {
                font-size: 0.96rem;
                color: #334155;
            }

            .footer-note {
                text-align: center;
                color: #334155;
                margin-top: 30px;
                margin-bottom: 10px;
                font-size: 0.93rem;
            }

            div[data-testid="metric-container"] {
                border-radius: 14px;
                padding: 12px;
                background: linear-gradient(145deg, #ffffff 0%, #f7fbff 100%);
                border: 1px solid rgba(148, 163, 184, 0.35);
                box-shadow: 0 6px 16px rgba(15, 23, 42, 0.07);
            }

            .stButton > button {
                border-radius: 12px;
                border: none;
                padding: 0.6rem 0.9rem;
                font-weight: 700;
                background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
                color: white;
                box-shadow: 0 8px 18px rgba(37, 99, 235, 0.28);
                transition: all 0.2s ease;
            }

            .stButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 10px 20px rgba(37, 99, 235, 0.35);
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            }

            [data-testid="stSidebar"] * {
                color: #f8fafc;
            }

            .sidebar-box {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 14px;
                padding: 14px;
                margin-bottom: 12px;
            }

            @media (max-width: 768px) {
                .hero-title {
                    font-size: 1.55rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown("## Tentang Aplikasi")
        st.markdown(
            """
            <div class="sidebar-box">
            Aplikasi ini menggunakan Machine Learning untuk mengklasifikasikan transaksi
            keuangan secara otomatis menggunakan model Naive Bayes dan TF-IDF.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Teknologi")
        st.markdown(
            """
            - Python
            - Streamlit
            - Scikit-learn
            - TF-IDF
            - Naive Bayes
            """
        )

        st.markdown("### Kategori yang didukung")
        st.markdown(
            """
            - 🍜 Makanan
            - 🛵 Transport
            - 💡 Tagihan
            - 🛍 Belanja
            - 🎮 Hiburan
            - 💊 Kesehatan
            - 📚 Pendidikan
            """
        )

        st.markdown("---")
        st.caption("Created by Rifqi Kurniansyah")


def render_header():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">SMART EXPENSE CATEGORIZER 💰 🤖 📊</div>
            <div class="hero-subtitle">
                Klasifikasi Pengeluaran Otomatis Berbasis Machine Learning
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_example_buttons():
    st.markdown("### Contoh Transaksi Cepat")
    cols = st.columns(2)
    for idx, example in enumerate(EXAMPLE_TRANSACTIONS):
        with cols[idx % 2]:
            if st.button(example, key=f"example_{idx}", width="stretch"):
                # Delay text_input state update to the next rerun before widget creation.
                st.session_state.pending_example = example
                st.rerun()


def render_prediction_result(result: dict):
    category = result["category"]
    icon = get_category_icon(category)
    color = get_category_color(category)

    confidence_text = "N/A"
    if result["confidence"] is not None:
        confidence_text = f"{result['confidence']:.0f}%"

    st.markdown(
        f"""
        <div class="prediction-card" style="border-left-color: {color};">
            <div class="prediction-title">Kategori: {category} {icon}</div>
            <div class="prediction-meta"><b>Confidence:</b> {confidence_text}</div>
            <div class="prediction-meta"><b>Transaksi:</b> {result['original_text']}</div>
            <div class="prediction-meta"><b>Text Cleaned:</b> {result['cleaned_text']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_history_table():
    st.markdown("### Riwayat Prediksi")

    if not st.session_state.history:
        st.info("Belum ada riwayat prediksi pada sesi ini.")
        return

    history_df = pd.DataFrame(st.session_state.history)
    history_df.insert(0, "No", range(1, len(history_df) + 1))

    st.dataframe(
        history_df[["No", "Waktu", "Transaksi", "Kategori"]],
        width="stretch",
        hide_index=True,
    )


def main():
    init_session_state()

    # Apply selected quick example before transaction_input widget is instantiated.
    if st.session_state.pending_example is not None:
        st.session_state.transaction_input = st.session_state.pending_example
        st.session_state.pending_example = None

    inject_custom_css()
    render_sidebar()
    render_header()

    try:
        model, vectorizer = load_model()
    except Exception as error:
        st.error(f"Gagal memuat model: {error}")
        st.stop()

    metric_col_1, metric_col_2 = st.columns(2)
    with metric_col_1:
        st.metric("Jumlah prediksi dalam sesi", len(st.session_state.history))
    with metric_col_2:
        unique_categories = len({item["Kategori"] for item in st.session_state.history})
        st.metric("Jumlah kategori unik", unique_categories)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Input Transaksi")

    transaction_text = st.text_input(
        "Masukkan transaksi",
        placeholder="Contoh: beli kopi susu 25rb",
        key="transaction_input",
    )

    predict_clicked = st.button("Prediksi Kategori", width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    render_example_buttons()

    if predict_clicked:
        if not transaction_text.strip():
            st.warning("Masukkan transaksi terlebih dahulu.")
        else:
            try:
                category, confidence, cleaned_text = predict_category(
                    transaction_text, model, vectorizer
                )
                prediction_result = {
                    "category": category,
                    "confidence": confidence,
                    "cleaned_text": cleaned_text,
                    "original_text": transaction_text,
                }
                st.session_state.last_prediction = prediction_result
                st.session_state.history.append(
                    {
                        "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Transaksi": transaction_text,
                        "Kategori": category,
                    }
                )
            except ValueError as error:
                st.warning(str(error))

    if st.session_state.last_prediction:
        render_prediction_result(st.session_state.last_prediction)

    render_history_table()

    if st.button("Clear History", width="stretch"):
        st.session_state.history = []
        st.session_state.last_prediction = None
        st.success("Riwayat prediksi berhasil dihapus.")

    st.markdown(
        """
        <div class="footer-note">
            Created by Rifqi Kurniansyah<br>
            Smart Expense Categorizer Project
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
