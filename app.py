import hashlib
import io
import pickle
import re
import sqlite3
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

st.set_page_config(
    page_title="Finance Saku",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = Path(__file__).resolve().parent / "finance_saku.db"

EXPENSE_CATEGORIES = [
    "Makanan",
    "Transport",
    "Tagihan",
    "Belanja",
    "Hiburan",
    "Kesehatan",
    "Pendidikan",
    "Lainnya",
]
INCOME_CATEGORIES = ["Gaji", "Bonus", "Investasi", "Freelance", "Lainnya"]


def init_session_state():
    defaults = {
        "is_authenticated": False,
        "user_id": None,
        "user_email": "",
        "active_page": "Dashboard",
        "txn_editor_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def inject_custom_css():
    st.markdown(
        """
        <style>
            /* Hero banner styling */
            .hero {
                background: linear-gradient(115deg, #14532d 0%, #1d4ed8 45%, #0f766e 100%);
                border-radius: 12px;
                padding: 24px;
                color: #ffffff;
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
                margin-bottom: 24px;
            }
            .hero h1 {
                margin: 0;
                font-size: 2rem;
                color: #ffffff;
            }
            .hero p {
                margin-top: 8px;
                margin-bottom: 0;
                font-size: 1rem;
                opacity: 0.95;
                color: #ffffff;
            }
            
            /* Sidebar styling */
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                transaction_type TEXT NOT NULL CHECK(transaction_type IN ('expense', 'income')),
                amount REAL NOT NULL,
                description TEXT NOT NULL,
                category TEXT NOT NULL,
                transaction_date TEXT NOT NULL,
                confidence REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_transactions_user_date ON transactions(user_id, transaction_date)"
        )
        connection.commit()


def hash_password(password: str) -> str:
    salt = hashlib.sha256(f"{datetime.utcnow().timestamp()}_{password}".encode()).hexdigest()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120000)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    salt, digest_hex = password_hash.split("$", 1)
    check_digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120000)
    return check_digest.hex() == digest_hex


def register_user(email: str, password: str) -> tuple[bool, str]:
    email = email.strip().lower()
    if "@" not in email or len(email) < 5:
        return False, "Email tidak valid."
    if len(password) < 6:
        return False, "Password minimal 6 karakter."

    try:
        with get_connection() as connection:
            connection.execute(
                "INSERT INTO users(email, password_hash, created_at) VALUES(?, ?, ?)",
                (email, hash_password(password), datetime.now().isoformat()),
            )
            connection.commit()
    except sqlite3.IntegrityError:
        return False, "Email sudah terdaftar."

    return True, "Registrasi berhasil. Silakan login."


def login_user(email: str, password: str) -> tuple[bool, str]:
    with get_connection() as connection:
        row = connection.execute("SELECT id, email, password_hash FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()

    if row is None:
        return False, "Akun tidak ditemukan."

    if not verify_password(password, row["password_hash"]):
        return False, "Password salah."

    st.session_state.is_authenticated = True
    st.session_state.user_id = row["id"]
    st.session_state.user_email = row["email"]
    return True, "Login berhasil."


def logout_user():
    st.session_state.is_authenticated = False
    st.session_state.user_id = None
    st.session_state.user_email = ""
    st.session_state.active_page = "Dashboard"


@st.cache_resource
def load_model() -> tuple[Optional[object], Optional[object], Optional[str]]:
    base_path = Path(__file__).resolve().parent
    model_path = base_path / "model.pkl"
    vectorizer_path = base_path / "vectorizer.pkl"
    if not model_path.exists() or not vectorizer_path.exists():
        return None, None, "Model tidak ditemukan, auto category menggunakan fallback rule."

    try:
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        with open(vectorizer_path, "rb") as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
    except Exception as error:
        return None, None, f"Model gagal dimuat: {error}"

    return model, vectorizer, None


def clean_text(text: str) -> str:
    cleaned = text.lower()
    cleaned = re.sub(r"\d+", " ", cleaned)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"_", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def fallback_expense_category(description: str) -> str:
    text = clean_text(description)
    keyword_map = [
        ("Makanan", ["makan", "kopi", "nasi", "resto", "warung", "jajan"]),
        ("Transport", ["bensin", "tol", "parkir", "grab", "gojek", "kereta"]),
        ("Tagihan", ["listrik", "air", "internet", "pln", "tagihan", "bpjs"]),
        ("Belanja", ["belanja", "toko", "sepatu", "baju", "marketplace"]),
        ("Hiburan", ["film", "game", "karaoke", "wisata", "konser"]),
        ("Kesehatan", ["obat", "dokter", "apotek", "vitamin", "rumah sakit"]),
        ("Pendidikan", ["kursus", "kuliah", "sekolah", "buku", "ujian"]),
    ]
    for category, keywords in keyword_map:
        if any(keyword in text for keyword in keywords):
            return category
    return "Lainnya"


def predict_expense_category(description: str, model, vectorizer) -> tuple[str, Optional[float]]:
    cleaned = clean_text(description)
    if not cleaned:
        return "Lainnya", None

    if model is None or vectorizer is None:
        return fallback_expense_category(description), None

    vector = vectorizer.transform([cleaned])
    category = str(model.predict(vector)[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(vector)
        confidence = float(np.max(probability) * 100)
    return category, confidence


def format_rupiah(amount: float) -> str:
    return f"Rp {amount:,.0f}".replace(",", ".")


def add_transaction(
    user_id: int,
    transaction_type: str,
    amount: float,
    description: str,
    category: str,
    transaction_date: date,
    confidence: Optional[float] = None,
):
    now_iso = datetime.now().isoformat()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO transactions(
                user_id, transaction_type, amount, description, category,
                transaction_date, confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                transaction_type,
                amount,
                description.strip(),
                category,
                transaction_date.isoformat(),
                confidence,
                now_iso,
            ),
        )
        connection.commit()


def update_transaction(
    transaction_id: int,
    user_id: int,
    transaction_type: str,
    amount: float,
    description: str,
    category: str,
    transaction_date: date,
):
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE transactions
            SET transaction_type = ?, amount = ?, description = ?, category = ?,
                transaction_date = ?, updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (
                transaction_type,
                amount,
                description.strip(),
                category,
                transaction_date.isoformat(),
                datetime.now().isoformat(),
                transaction_id,
                user_id,
            ),
        )
        connection.commit()


def delete_transaction(transaction_id: int, user_id: int):
    with get_connection() as connection:
        connection.execute(
            "DELETE FROM transactions WHERE id = ? AND user_id = ?",
            (transaction_id, user_id),
        )
        connection.commit()


def fetch_transactions(
    user_id: int,
    start_date: date,
    end_date: date,
    tx_type_filter: str = "Semua",
    category_filter: str = "Semua",
    search_text: str = "",
) -> pd.DataFrame:
    query = [
        """
        SELECT id, transaction_type, amount, description, category, transaction_date,
               confidence, created_at
        FROM transactions
        WHERE user_id = ?
          AND date(transaction_date) BETWEEN date(?) AND date(?)
        """
    ]
    params: list = [user_id, start_date.isoformat(), end_date.isoformat()]

    if tx_type_filter == "Pengeluaran":
        query.append("AND transaction_type = 'expense'")
    elif tx_type_filter == "Pemasukan":
        query.append("AND transaction_type = 'income'")

    if category_filter != "Semua":
        query.append("AND category = ?")
        params.append(category_filter)

    if search_text.strip():
        query.append("AND lower(description) LIKE ?")
        params.append(f"%{search_text.strip().lower()}%")

    query.append("ORDER BY date(transaction_date) DESC, id DESC")

    with get_connection() as connection:
        df = pd.read_sql_query("\n".join(query), connection, params=params)
    return df


def fetch_monthly_summary(user_id: int, start_date: date, end_date: date) -> pd.DataFrame:
    query = """
        SELECT
            strftime('%Y-%m', transaction_date) AS bulan,
            SUM(CASE WHEN transaction_type = 'income' THEN amount ELSE 0 END) AS total_pemasukan,
            SUM(CASE WHEN transaction_type = 'expense' THEN amount ELSE 0 END) AS total_pengeluaran
        FROM transactions
        WHERE user_id = ? AND date(transaction_date) BETWEEN date(?) AND date(?)
        GROUP BY strftime('%Y-%m', transaction_date)
        ORDER BY bulan
    """
    with get_connection() as connection:
        return pd.read_sql_query(
            query,
            connection,
            params=[user_id, start_date.isoformat(), end_date.isoformat()],
        )


def fetch_category_breakdown(user_id: int, start_date: date, end_date: date) -> pd.DataFrame:
    query = """
        SELECT category, SUM(amount) AS total
        FROM transactions
        WHERE user_id = ?
          AND transaction_type = 'expense'
          AND date(transaction_date) BETWEEN date(?) AND date(?)
        GROUP BY category
        ORDER BY total DESC
    """
    with get_connection() as connection:
        return pd.read_sql_query(
            query,
            connection,
            params=[user_id, start_date.isoformat(), end_date.isoformat()],
        )


def render_header():
    st.markdown(
        """
        <div class="hero">
            <h1>Finance Saku</h1>
            <p>Kelola pemasukan, pengeluaran, auto-category, dan laporan dalam satu aplikasi.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_login_register_page():
    render_header()
    st.subheader("Login atau Registrasi")

    login_tab, register_tab = st.tabs(["Login", "Registrasi"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="email@contoh.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", width="stretch")
            if submitted:
                is_ok, message = login_user(email, password)
                if is_ok:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

    with register_tab:
        with st.form("register_form"):
            email = st.text_input("Email registrasi", placeholder="email@contoh.com")
            password = st.text_input("Password baru", type="password")
            password_confirm = st.text_input("Konfirmasi password", type="password")
            submitted = st.form_submit_button("Buat Akun", width="stretch")
            if submitted:
                if password != password_confirm:
                    st.error("Konfirmasi password tidak sama.")
                else:
                    is_ok, message = register_user(email, password)
                    if is_ok:
                        st.success(message)
                    else:
                        st.error(message)


def render_sidebar_nav(model_error: Optional[str]):
    with st.sidebar:
        st.markdown("## Menu Finance Saku")
        st.caption(f"User aktif: {st.session_state.user_email}")

        st.session_state.active_page = st.radio(
            "Navigasi",
            ["Dashboard", "Transaksi", "Laporan"],
            index=["Dashboard", "Transaksi", "Laporan"].index(st.session_state.active_page),
        )

        if model_error:
            st.warning(model_error)

        st.markdown("---")
        if st.button("Logout", width="stretch"):
            logout_user()
            st.rerun()


def render_dashboard(user_id: int):
    render_header()
    st.subheader("Dashboard Interaktif")

    filter_col_1, filter_col_2 = st.columns(2)
    with filter_col_1:
        start_date = st.date_input("Tanggal awal", date.today().replace(day=1), key="dash_start")
    with filter_col_2:
        end_date = st.date_input("Tanggal akhir", date.today(), key="dash_end")

    if start_date > end_date:
        st.error("Tanggal awal tidak boleh lebih besar dari tanggal akhir.")
        return

    df = fetch_transactions(user_id, start_date, end_date)
    if df.empty:
        st.info("Belum ada transaksi pada periode ini.")
        return

    total_income = float(df.loc[df["transaction_type"] == "income", "amount"].sum())
    total_expense = float(df.loc[df["transaction_type"] == "expense", "amount"].sum())
    balance = total_income - total_expense

    metric_cols = st.columns(4)
    metric_cols[0].metric("Total Pemasukan", format_rupiah(total_income))
    metric_cols[1].metric("Total Pengeluaran", format_rupiah(total_expense))
    metric_cols[2].metric("Saldo", format_rupiah(balance))
    metric_cols[3].metric("Jumlah Transaksi", int(df.shape[0]))

    monthly_df = fetch_monthly_summary(user_id, start_date, end_date)
    category_df = fetch_category_breakdown(user_id, start_date, end_date)

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        if not monthly_df.empty:
            trend_fig = px.line(
                monthly_df,
                x="bulan",
                y=["total_pemasukan", "total_pengeluaran"],
                markers=True,
                title="Tren Bulanan",
                labels={"value": "Nominal", "bulan": "Bulan", "variable": "Jenis"},
                template="plotly_white",
            )
            st.plotly_chart(trend_fig, use_container_width=True)

    with chart_col_2:
        if not category_df.empty:
            pie_fig = px.pie(
                category_df,
                names="category",
                values="total",
                title="Komposisi Kategori Pengeluaran",
                hole=0.45,
                template="plotly_white",
            )
            st.plotly_chart(pie_fig, use_container_width=True)

    preview_df = df.copy()
    preview_df["Jenis"] = preview_df["transaction_type"].map(
        {"expense": "Pengeluaran", "income": "Pemasukan"}
    )
    preview_df["Nominal"] = preview_df["amount"].apply(format_rupiah)
    preview_df["Tanggal"] = preview_df["transaction_date"]

    st.markdown("### Transaksi Terbaru")
    st.dataframe(
        preview_df[["Tanggal", "Jenis", "description", "category", "Nominal"]].rename(
            columns={"description": "Deskripsi", "category": "Kategori"}
        ),
        width="stretch",
        hide_index=True,
    )


def render_transaction_form(user_id: int, model, vectorizer):
    st.markdown("### Tambah Transaksi")
    with st.form("add_transaction_form"):
        transaction_type_label = st.radio("Jenis", ["Pengeluaran", "Pemasukan"], horizontal=True)
        amount = st.number_input("Nominal", min_value=1000.0, step=1000.0, value=10000.0)
        description = st.text_input("Deskripsi", placeholder="Contoh: bayar listrik PLN")
        transaction_date = st.date_input("Tanggal", date.today())

        if transaction_type_label == "Pengeluaran":
            category_mode = st.selectbox(
                "Kategori",
                ["Auto (prediksi model)"] + EXPENSE_CATEGORIES,
                index=0,
            )
        else:
            category_mode = st.selectbox("Kategori", INCOME_CATEGORIES, index=0)

        submitted = st.form_submit_button("Simpan Transaksi", width="stretch")

    if not submitted:
        return
    if not description.strip():
        st.warning("Deskripsi wajib diisi.")
        return

    transaction_type = "expense" if transaction_type_label == "Pengeluaran" else "income"
    confidence = None

    if transaction_type == "expense":
        if category_mode == "Auto (prediksi model)":
            category, confidence = predict_expense_category(description, model, vectorizer)
        else:
            category = category_mode
    else:
        category = category_mode

    add_transaction(
        user_id=user_id,
        transaction_type=transaction_type,
        amount=float(amount),
        description=description,
        category=category,
        transaction_date=transaction_date,
        confidence=confidence,
    )

    message = f"Transaksi berhasil disimpan dengan kategori: {category}."
    if confidence is not None:
        message += f" Confidence: {confidence:.0f}%"
    st.success(message)


def render_transaction_list_and_actions(user_id: int):
    st.markdown("### Daftar Transaksi")
    filter_cols = st.columns(5)
    with filter_cols[0]:
        start_date = st.date_input("Dari", date.today() - timedelta(days=30), key="tx_start")
    with filter_cols[1]:
        end_date = st.date_input("Sampai", date.today(), key="tx_end")
    with filter_cols[2]:
        tx_filter = st.selectbox("Jenis", ["Semua", "Pengeluaran", "Pemasukan"], key="tx_filter")
    with filter_cols[3]:
        category_filter = st.selectbox(
            "Kategori",
            ["Semua"] + sorted(EXPENSE_CATEGORIES + INCOME_CATEGORIES),
            key="tx_category_filter",
        )
    with filter_cols[4]:
        search_text = st.text_input("Cari deskripsi", key="tx_search")

    if start_date > end_date:
        st.error("Rentang tanggal tidak valid.")
        return

    df = fetch_transactions(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        tx_type_filter=tx_filter,
        category_filter=category_filter,
        search_text=search_text,
    )

    if df.empty:
        st.info("Tidak ada data transaksi sesuai filter.")
        return

    display_df = df.copy()
    display_df["Jenis"] = display_df["transaction_type"].map(
        {"expense": "Pengeluaran", "income": "Pemasukan"}
    )
    display_df["Nominal"] = display_df["amount"].apply(format_rupiah)
    display_df["Tanggal"] = display_df["transaction_date"]

    st.dataframe(
        display_df[["id", "Tanggal", "Jenis", "description", "category", "Nominal"]].rename(
            columns={"id": "ID", "description": "Deskripsi", "category": "Kategori"}
        ),
        width="stretch",
        hide_index=True,
    )

    options = {
        int(row["id"]): f"#{int(row['id'])} | {row['transaction_date']} | {row['description'][:35]}"
        for _, row in df.iterrows()
    }
    selected_id = st.selectbox(
        "Pilih transaksi untuk edit atau hapus",
        options=list(options.keys()),
        format_func=lambda item: options[item],
    )

    selected_row = df.loc[df["id"] == selected_id].iloc[0]
    with st.expander("Edit transaksi", expanded=False):
        with st.form("edit_transaction_form"):
            selected_type_label = (
                "Pengeluaran" if selected_row["transaction_type"] == "expense" else "Pemasukan"
            )
            type_label = st.radio(
                "Jenis transaksi",
                ["Pengeluaran", "Pemasukan"],
                index=0 if selected_type_label == "Pengeluaran" else 1,
                horizontal=True,
            )
            amount = st.number_input(
                "Nominal edit",
                min_value=1000.0,
                step=1000.0,
                value=float(selected_row["amount"]),
            )
            description = st.text_input("Deskripsi edit", value=str(selected_row["description"]))
            transaction_date = st.date_input(
                "Tanggal edit",
                value=datetime.fromisoformat(str(selected_row["transaction_date"])).date(),
            )
            category_pool = EXPENSE_CATEGORIES if type_label == "Pengeluaran" else INCOME_CATEGORIES
            default_category_index = (
                category_pool.index(selected_row["category"])
                if selected_row["category"] in category_pool
                else len(category_pool) - 1
            )
            category = st.selectbox(
                "Kategori edit",
                category_pool,
                index=default_category_index,
            )

            save_clicked = st.form_submit_button("Simpan Perubahan", width="stretch")
            if save_clicked:
                if not description.strip():
                    st.warning("Deskripsi tidak boleh kosong.")
                else:
                    update_transaction(
                        transaction_id=selected_id,
                        user_id=user_id,
                        transaction_type=("expense" if type_label == "Pengeluaran" else "income"),
                        amount=float(amount),
                        description=description,
                        category=category,
                        transaction_date=transaction_date,
                    )
                    st.success("Transaksi berhasil diperbarui.")
                    st.rerun()

    delete_col_1, delete_col_2 = st.columns([1, 2])
    with delete_col_1:
        confirm_delete = st.checkbox("Konfirmasi hapus", key=f"confirm_delete_{selected_id}")
    with delete_col_2:
        if st.button("Hapus transaksi terpilih", width="stretch"):
            if not confirm_delete:
                st.warning("Centang konfirmasi hapus terlebih dahulu.")
            else:
                delete_transaction(selected_id, user_id)
                st.success("Transaksi berhasil dihapus.")
                st.rerun()


def render_transactions(user_id: int, model, vectorizer):
    render_header()
    st.subheader("Transaksi Pemasukan dan Pengeluaran")

    col_1, col_2 = st.columns([1, 1.3])
    with col_1:
        render_transaction_form(user_id, model, vectorizer)
    with col_2:
        render_transaction_list_and_actions(user_id)


def create_pdf_report(df: pd.DataFrame, start_date: date, end_date: date) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 40

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, "Laporan Finance Saku")
    y -= 20
    pdf.setFont("Helvetica", 10)
    pdf.drawString(40, y, f"Periode: {start_date.isoformat()} s/d {end_date.isoformat()}")
    y -= 24

    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawString(40, y, "Tanggal")
    pdf.drawString(105, y, "Jenis")
    pdf.drawString(170, y, "Kategori")
    pdf.drawString(260, y, "Nominal")
    pdf.drawString(340, y, "Deskripsi")
    y -= 14

    table_rows = df.copy()
    table_rows["jenis"] = table_rows["transaction_type"].map(
        {"expense": "Pengeluaran", "income": "Pemasukan"}
    )
    table_rows["nominal"] = table_rows["amount"].apply(format_rupiah)

    pdf.setFont("Helvetica", 8)
    for _, row in table_rows.head(200).iterrows():
        if y < 40:
            pdf.showPage()
            y = height - 40
            pdf.setFont("Helvetica", 8)

        description = str(row["description"]).replace("\n", " ")[:38]
        pdf.drawString(40, y, str(row["transaction_date"])[:10])
        pdf.drawString(105, y, str(row["jenis"])[:11])
        pdf.drawString(170, y, str(row["category"])[:16])
        pdf.drawString(260, y, str(row["nominal"])[:16])
        pdf.drawString(340, y, description)
        y -= 12

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


def render_reports(user_id: int):
    render_header()
    st.subheader("Laporan")

    col_1, col_2, col_3 = st.columns(3)
    with col_1:
        start_date = st.date_input("Tanggal awal laporan", date.today() - timedelta(days=90), key="report_start")
    with col_2:
        end_date = st.date_input("Tanggal akhir laporan", date.today(), key="report_end")
    with col_3:
        type_filter = st.selectbox("Filter jenis", ["Semua", "Pengeluaran", "Pemasukan"], key="report_type")

    if start_date > end_date:
        st.error("Tanggal awal tidak boleh lebih besar dari tanggal akhir.")
        return

    df = fetch_transactions(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        tx_type_filter=type_filter,
    )
    if df.empty:
        st.info("Belum ada data untuk laporan pada periode ini.")
        return

    total_income = float(df.loc[df["transaction_type"] == "income", "amount"].sum())
    total_expense = float(df.loc[df["transaction_type"] == "expense", "amount"].sum())
    balance = total_income - total_expense

    sum_cols = st.columns(3)
    sum_cols[0].metric("Total Pemasukan", format_rupiah(total_income))
    sum_cols[1].metric("Total Pengeluaran", format_rupiah(total_expense))
    sum_cols[2].metric("Saldo", format_rupiah(balance))

    monthly_df = fetch_monthly_summary(user_id, start_date, end_date)
    if not monthly_df.empty:
        monthly_df["saldo"] = monthly_df["total_pemasukan"] - monthly_df["total_pengeluaran"]
        st.markdown("### Ringkasan Bulanan")
        monthly_display = monthly_df.rename(
            columns={
                "bulan": "Bulan",
                "total_pemasukan": "Total Pemasukan",
                "total_pengeluaran": "Total Pengeluaran",
                "saldo": "Saldo",
            }
        )
        st.dataframe(monthly_display, width="stretch", hide_index=True)

        chart = px.bar(
            monthly_df,
            x="bulan",
            y=["total_pemasukan", "total_pengeluaran", "saldo"],
            barmode="group",
            title="Perbandingan Bulanan",
            template="plotly_white",
        )
        st.plotly_chart(chart, use_container_width=True)

    category_df = fetch_category_breakdown(user_id, start_date, end_date)
    if not category_df.empty:
        pie = px.pie(category_df, names="category", values="total", title="Komposisi Kategori", template="plotly_white")
        st.plotly_chart(pie, use_container_width=True)

    export_df = df.copy()
    export_df["transaction_type"] = export_df["transaction_type"].map(
        {"expense": "Pengeluaran", "income": "Pemasukan"}
    )

    csv_data = export_df.to_csv(index=False).encode("utf-8")
    pdf_data = create_pdf_report(df, start_date, end_date)

    export_cols = st.columns(2)
    export_cols[0].download_button(
        label="Export CSV",
        data=csv_data,
        file_name=f"laporan_finance_saku_{start_date}_{end_date}.csv",
        mime="text/csv",
        width="stretch",
    )
    export_cols[1].download_button(
        label="Export PDF",
        data=pdf_data,
        file_name=f"laporan_finance_saku_{start_date}_{end_date}.pdf",
        mime="application/pdf",
        width="stretch",
    )


def main():
    init_db()
    init_session_state()
    inject_custom_css()
    model, vectorizer, model_error = load_model()

    if not st.session_state.is_authenticated:
        render_login_register_page()
        return

    render_sidebar_nav(model_error)
    if st.session_state.active_page == "Dashboard":
        render_dashboard(st.session_state.user_id)
    elif st.session_state.active_page == "Transaksi":
        render_transactions(st.session_state.user_id, model, vectorizer)
    elif st.session_state.active_page == "Laporan":
        render_reports(st.session_state.user_id)


if __name__ == "__main__":
    main()
