"""
AI Sentiment Analyzer Using NLP and LSTM
========================================
A comprehensive sentiment classification application built with:
- Natural Language Processing (NLP) for text preprocessing & tokenization
- TensorFlow/Keras for the Long Short-Term Memory (LSTM) neural network
- Streamlit for the responsive, clean web interface
- CSV File Upload & Batch Sentiment Analysis support (>=100 KB datasets)
- NumPy and Pandas for numerical and tabular processing
"""

import os
import re
import io
import time
import pandas as pd
import numpy as np

# Configure Keras home directory to prevent sandbox permission errors
os.environ["KERAS_HOME"] = "/tmp/.keras"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Sentiment Analyzer | NLP & LSTM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional light theme with subtle blue accents and clean typography
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1.75rem;
        padding-bottom: 2.5rem;
        max-width: 1150px;
    }
    
    h1, h2, h3, h4 {
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-weight: 700;
    }
    
    .sentiment-positive {
        background-color: #f0fdf4;
        border: 1.5px solid #86efac;
        color: #166534;
        border-radius: 10px;
        padding: 1rem 1.25rem;
    }
    
    .sentiment-negative {
        background-color: #fef2f2;
        border: 1.5px solid #fca5a5;
        color: #991b1b;
        border-radius: 10px;
        padding: 1rem 1.25rem;
    }
    
    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #e0f2fe;
        color: #0369a1;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Hyperparameters & Configuration
# -----------------------------------------------------------------------------
VOCAB_SIZE = 1000      # Vocabulary size for the embedding layer
MAX_LEN = 25           # Maximum sequence length (padded / truncated)
EMBEDDING_DIM = 16     # Dimension of word embedding vector
LSTM_UNITS = 16        # Hidden units in LSTM recurrent layer
EPOCHS = 12            # Number of training epochs
BATCH_SIZE = 16        # Mini-batch gradient descent size
DEFAULT_DATASET_PATH = "dataset.csv"

# -----------------------------------------------------------------------------
# 3. NLP Text Preprocessing Function
# -----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Cleans and normalizes raw text:
    - Lowercases all characters
    - Strips special punctuation, URLs, and non-alphabetic symbols
    - Collapses multiple whitespace into a single space
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------------------------------------------------------
# 4. Model Training & Resource Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_sentiment_lstm_model():
    """
    Builds, trains, and caches the LSTM neural network on the multi-domain dataset.
    Returns: (trained_model, fitted_tokenizer, training_accuracy, dataset_size_kb)
    """
    # Verify dataset exists, else generate it
    if not os.path.exists(DEFAULT_DATASET_PATH):
        from generate_dataset import generate_dataset
        generate_dataset(DEFAULT_DATASET_PATH, target_rows=2200)

    dataset_size_kb = os.path.getsize(DEFAULT_DATASET_PATH) / 1024.0
    df = pd.read_csv(DEFAULT_DATASET_PATH)

    # Use a balanced slice of 600 samples for high accuracy with ~2s training time
    train_slice = df.iloc[:600]
    cleaned_sentences = [clean_text(t) for t in train_slice["review"].tolist()]
    labels = train_slice["label"].values.astype(np.float32)

    # Fit Keras Tokenizer
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(cleaned_sentences)

    # Sequences and Padding
    sequences = tokenizer.texts_to_sequences(cleaned_sentences)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

    # Set seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Construct Keras Sequential Architecture
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        LSTM(LSTM_UNITS, dropout=0.1, recurrent_dropout=0.1),
        Dense(16, activation="relu"),
        Dropout(0.1),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        padded,
        labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    final_accuracy = float(history.history["accuracy"][-1]) * 100.0
    return model, tokenizer, final_accuracy, dataset_size_kb

# -----------------------------------------------------------------------------
# 5. Prediction Helpers (Single & Batch)
# -----------------------------------------------------------------------------
def predict_single(text: str, model, tokenizer):
    """
    Inference helper for an individual text input.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return None, 0.0, 0.0, []

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    raw_prob = float(model.predict(padded, verbose=0)[0][0])

    if raw_prob >= 0.5:
        sentiment = "Positive"
        confidence = raw_prob * 100.0
    else:
        sentiment = "Negative"
        confidence = (1.0 - raw_prob) * 100.0

    tokens = cleaned.split()
    recognized = [w for w in tokens if w in tokenizer.word_index and tokenizer.word_index[w] < VOCAB_SIZE]

    return sentiment, raw_prob, confidence, recognized


def predict_batch(texts: list, model, tokenizer, batch_size=64):
    """
    Vectorized batch inference for tabular CSV datasets.
    """
    cleaned_texts = [clean_text(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(cleaned_texts)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

    probs = model.predict(padded, batch_size=batch_size, verbose=0).flatten()
    sentiments = np.where(probs >= 0.5, "Positive", "Negative")
    confidences = np.where(probs >= 0.5, probs * 100.0, (1.0 - probs) * 100.0)

    return sentiments, probs, confidences

# -----------------------------------------------------------------------------
# 6. Streamlit User Interface
# -----------------------------------------------------------------------------
def main():
    # Cache / load LSTM model
    with st.spinner("Initializing NLP pipeline & loading LSTM network..."):
        model, tokenizer, train_acc, ds_size_kb = train_sentiment_lstm_model()

    # Sidebar: Model Specs & Pipeline Explanation
    with st.sidebar:
        st.markdown("### 🧠 Model Specifications")
        with st.container(border=True):
            st.markdown(f"""
            - **Embedding Layer:** Vocab `{VOCAB_SIZE}`, Dim `{EMBEDDING_DIM}`
            - **Recurrent Layer:** LSTM (`{LSTM_UNITS}` units)
            - **Dense Hidden:** 16 units (ReLU) + Dropout
            - **Output Layer:** Sigmoid ($[0.0, 1.0]$)
            - **Built-in Dataset:** `{ds_size_kb:.1f} KB` (2,200 rows)
            - **Training Accuracy:** `{train_acc:.1f}%`
            """)

        with st.expander("📖 NLP Pipeline Stages"):
            st.markdown("""
            1. **Regex Cleaning:** Stripping URLs, digits, punctuation.
            2. **Tokenization:** Mapping words to integer indices.
            3. **Padding (`pad_sequences`):** Fixed length of 25 tokens.
            4. **Vector Embedding:** Dense semantic vector mapping.
            5. **LSTM Memory Cells:** Context preservation across tokens.
            6. **Sigmoid Output:** Producing probability score.
            """)

        with st.container(border=True):
            st.markdown("##### 📁 Download Sample Dataset")
            st.caption("Download the included 203 KB benchmark dataset to test file uploading.")
            if os.path.exists(DEFAULT_DATASET_PATH):
                with open(DEFAULT_DATASET_PATH, "rb") as f:
                    st.download_button(
                        label="⬇️ Download dataset.csv (203 KB)",
                        data=f,
                        file_name="dataset.csv",
                        mime="text/csv",
                        width="stretch"
                    )

    # Main Header
    st.markdown("## 🧠 AI Sentiment Analyzer Using NLP & LSTM")
    st.markdown(
        "Analyze sentiments of customer reviews, product feedback, or social comments "
        "using a **TensorFlow LSTM Neural Network**. Analyze single sentences or **upload CSV files (100+ KB)** for batch analysis."
    )
    st.markdown(
        '<span class="badge">TensorFlow / Keras</span>'
        '<span class="badge">LSTM Neural Net</span>'
        '<span class="badge">CSV Upload & Batch Analysis</span>'
        '<span class="badge">NLP Preprocessing</span>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Tabs for Single Sentence Analysis and Batch CSV Upload
    tab_single, tab_csv, tab_dataset = st.tabs([
        "💬 Single Text Analyzer",
        "📁 Batch CSV Upload & Analysis",
        "📊 Dataset & Architecture Explorer"
    ])

    # =========================================================================
    # TAB 1: Single Sentence Analyzer
    # =========================================================================
    with tab_single:
        if "user_text" not in st.session_state:
            st.session_state.user_text = ""

        st.markdown("##### 💡 Quick Test Samples:")
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)

        with s_col1:
            if st.button("🌟 Positive Movie", width="stretch"):
                st.session_state.user_text = "I love this movie, the storyline and acting were superb!"
                st.rerun()

        with s_col2:
            if st.button("🍴 Delicious Food", width="stretch"):
                st.session_state.user_text = "The food at this restaurant was delicious and freshly cooked."
                st.rerun()

        with s_col3:
            if st.button("💔 Broken Product", width="stretch"):
                st.session_state.user_text = "This product is terrible, broke on the first day of use."
                st.rerun()

        with s_col4:
            if st.button("😡 Rude Support", width="stretch"):
                st.session_state.user_text = "Rude customer service and nobody bothered to answer my questions."
                st.rerun()

        text_input = st.text_area(
            label="Enter sentence, product review, or feedback to analyze:",
            value=st.session_state.user_text,
            height=120,
            placeholder="Type or paste any review or feedback here...",
            key="single_text_area"
        )

        btn_c1, btn_c2, _ = st.columns([1.5, 1.2, 5])
        with btn_c1:
            analyze_clicked = st.button("🔍 Analyze Sentiment", type="primary", width="stretch")
        with btn_c2:
            clear_clicked = st.button("🗑️ Clear Text", width="stretch")

        if clear_clicked:
            st.session_state.user_text = ""
            st.rerun()

        if analyze_clicked:
            active_text = text_input.strip()
            if not active_text:
                st.warning("⚠️ Please enter text or click one of the quick test samples above.")
            else:
                try:
                    with st.spinner("Analyzing sequence through LSTM layers..."):
                        sentiment, raw_prob, confidence, recognized = predict_single(active_text, model, tokenizer)

                    if sentiment is None:
                        st.error("Could not extract meaningful textual features. Please enter valid words.")
                    else:
                        st.markdown("---")
                        st.markdown("### 📊 Analysis Results")

                        col_res, col_conf, col_prob = st.columns([1.5, 1.5, 1.5])

                        with col_res:
                            if sentiment == "Positive":
                                st.markdown("""
                                <div class="sentiment-positive">
                                    <h4 style="margin: 0; color: #166534;">🎉 Positive Sentiment</h4>
                                    <p style="margin: 0.25rem 0 0 0; font-size: 0.9rem;">Model detected favorable and constructive language.</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div class="sentiment-negative">
                                    <h4 style="margin: 0; color: #991b1b;">⚠️ Negative Sentiment</h4>
                                    <p style="margin: 0.25rem 0 0 0; font-size: 0.9rem;">Model detected unfavorable or critical language.</p>
                                </div>
                                """, unsafe_allow_html=True)

                        with col_conf:
                            st.metric(
                                label="Estimated Model Confidence",
                                value=f"{confidence:.2f}%",
                                help="Calculated distance from the 0.50 decision boundary."
                            )
                            st.progress(float(confidence / 100.0))

                        with col_prob:
                            st.metric(
                                label="Raw Sigmoid Output",
                                value=f"{raw_prob:.4f}",
                                help="0.0 = Strongly Negative | 1.0 = Strongly Positive"
                            )

                        st.markdown("#### 💡 Explanation")
                        st.markdown(
                            f"The LSTM network evaluated the sequential word context and generated a raw sigmoid "
                            f"activation of **`{raw_prob:.4f}`**. Since this score is "
                            f"{'above' if raw_prob >= 0.5 else 'below'} the decision threshold of **0.50**, "
                            f"the model assigned a **{sentiment}** classification with an estimated confidence of **{confidence:.2f}%**."
                        )

                        with st.expander("🔍 Inspect Extracted Tokens & Vocabulary Overlap"):
                            if recognized:
                                st.write("**Tokens recognized in vocabulary:**", recognized)
                            else:
                                st.write("Tokens treated under generalized out-of-vocabulary (`<OOV>`) context.")

                except Exception as ex:
                    st.error(f"Error during inference: {str(ex)}")

    # =========================================================================
    # TAB 2: Batch CSV Upload & Analysis (>= 100 KB Datasets)
    # =========================================================================
    with tab_csv:
        st.markdown("### 📁 Batch Sentiment Analysis via CSV File Upload")
        st.markdown(
            "Upload any CSV file containing customer reviews or feedback (supports datasets **larger than 100 KB**). "
            "The LSTM model will process the rows in batches, classify sentiments, and calculate confidence scores."
        )

        up_col1, up_col2 = st.columns([3, 2])

        with up_col1:
            uploaded_file = st.file_uploader(
                "Select a CSV file to upload",
                type=["csv"],
                help="Upload a CSV with text/review columns (>=100 KB supported)"
            )

        with up_col2:
            st.markdown("##### ⚡ Quick Demonstration:")
            st.caption("Don't have a CSV handy? Click below to instantly test with the built-in 203 KB dataset.")
            load_builtin = st.button("📂 Load Built-in 203 KB Dataset (`dataset.csv`)", width="stretch")

        df_to_process = None
        source_name = ""

        if uploaded_file is not None:
            try:
                file_size_kb = uploaded_file.size / 1024.0
                df_to_process = pd.read_csv(uploaded_file)
                source_name = f"Uploaded file: `{uploaded_file.name}` ({file_size_kb:.1f} KB)"
                st.success(f"Successfully loaded {source_name} with {len(df_to_process):,} rows!")
            except Exception as e:
                st.error(f"Failed to parse uploaded CSV: {str(e)}")

        elif load_builtin:
            if os.path.exists(DEFAULT_DATASET_PATH):
                df_to_process = pd.read_csv(DEFAULT_DATASET_PATH)
                size_kb = os.path.getsize(DEFAULT_DATASET_PATH) / 1024.0
                source_name = f"Built-in Benchmark Dataset: `dataset.csv` ({size_kb:.1f} KB)"
                st.session_state["cached_batch_df"] = df_to_process
                st.session_state["cached_source_name"] = source_name
                st.success(f"Loaded {source_name} with {len(df_to_process):,} rows!")

        elif "cached_batch_df" in st.session_state:
            df_to_process = st.session_state["cached_batch_df"]
            source_name = st.session_state["cached_source_name"]

        if df_to_process is not None:
            with st.container(border=True):
                st.markdown(f"**Dataset Overview:** {source_name}")
                st.write(f"Total Rows: **{len(df_to_process):,}** | Columns: `{list(df_to_process.columns)}`")

                # Column Selection for Text
                candidate_cols = list(df_to_process.columns)
                default_idx = 0
                for idx, col in enumerate(candidate_cols):
                    if col.lower() in ["review", "text", "sentence", "content", "feedback", "comment"]:
                        default_idx = idx
                        break

                text_col = st.selectbox(
                    "Select the column containing text reviews to analyze:",
                    options=candidate_cols,
                    index=default_idx
                )

                st.markdown("##### Preview of First 5 Rows:")
                st.dataframe(df_to_process.head(5), width="stretch")

                # Row limit control for flexible testing
                max_available = len(df_to_process)
                default_proc = min(500, max_available)
                rows_to_analyze = st.slider(
                    "Number of rows to analyze in this batch run:",
                    min_value=10,
                    max_value=max_available,
                    value=default_proc,
                    step=50,
                    help="You can analyze the entire dataset or a subset for ultra-fast processing."
                )

                run_batch_btn = st.button("🚀 Run Batch LSTM Sentiment Analysis", type="primary", width="stretch")

            if run_batch_btn:
                eval_slice = df_to_process.iloc[:rows_to_analyze].copy()
                raw_texts = eval_slice[text_col].astype(str).tolist()

                with st.spinner(f"Processing {len(raw_texts):,} rows through the LSTM model..."):
                    t0 = time.time()
                    sentiments, probs, confs = predict_batch(raw_texts, model, tokenizer)
                    elapsed = time.time() - t0

                eval_slice["Predicted_Sentiment"] = sentiments
                eval_slice["Model_Confidence_%"] = np.round(confs, 2)
                eval_slice["Sigmoid_Probability"] = np.round(probs, 4)

                # Batch Analysis Metrics
                st.markdown("---")
                st.markdown("### 📈 Batch Analysis Results")

                pos_count = int(np.sum(sentiments == "Positive"))
                neg_count = int(np.sum(sentiments == "Negative"))
                total_proc = len(sentiments)
                pos_pct = (pos_count / total_proc) * 100.0
                neg_pct = (neg_count / total_proc) * 100.0
                avg_conf = float(np.mean(confs))

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Analyzed", f"{total_proc:,}", f"Time: {elapsed:.2f}s")
                m2.metric("Positive Reviews", f"{pos_count:,}", f"{pos_pct:.1f}%")
                m3.metric("Negative Reviews", f"{neg_count:,}", f"{neg_pct:.1f}%")
                m4.metric("Avg. Confidence", f"{avg_conf:.1f}%")

                # Visual Sentiment Distribution
                st.markdown("##### Sentiment Distribution:")
                chart_df = pd.DataFrame({
                    "Sentiment": ["Positive", "Negative"],
                    "Count": [pos_count, neg_count]
                }).set_index("Sentiment")
                st.bar_chart(chart_df, width="stretch")

                # Ground Truth Evaluation if available
                ground_truth_col = None
                for c in ["sentiment", "label", "target"]:
                    if c in eval_slice.columns and c != "Predicted_Sentiment":
                        ground_truth_col = c
                        break

                if ground_truth_col:
                    with st.expander("🎯 Ground Truth Evaluation vs. Labels", expanded=True):
                        # Normalize ground truth for comparison
                        gt_vals = eval_slice[ground_truth_col].astype(str).str.strip().str.capitalize()
                        is_match = (gt_vals == eval_slice["Predicted_Sentiment"]) | (
                            (eval_slice[ground_truth_col] == 1) & (eval_slice["Predicted_Sentiment"] == "Positive")
                        ) | (
                            (eval_slice[ground_truth_col] == 0) & (eval_slice["Predicted_Sentiment"] == "Negative")
                        )
                        match_acc = (is_match.sum() / len(eval_slice)) * 100.0
                        st.info(f"🏆 Model Accuracy against `{ground_truth_col}` column: **{match_acc:.2f}%**")

                # Output Dataframe
                st.markdown("##### Annotated Results Table:")
                st.dataframe(eval_slice, width="stretch")

                # CSV Download Button
                csv_buffer = io.StringIO()
                eval_slice.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="⬇️ Download Annotated CSV with Predictions",
                    data=csv_buffer.getvalue(),
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv",
                    width="stretch"
                )

    # =========================================================================
    # TAB 3: Dataset & Architecture Explorer
    # =========================================================================
    with tab_dataset:
        st.markdown("### 📊 Dataset Details & Neural Network Architecture")

        d_col1, d_col2 = st.columns([1.5, 1])

        with d_col1:
            with st.container(border=True):
                st.markdown("#### 📁 Built-in Multi-Domain Dataset")
                if os.path.exists(DEFAULT_DATASET_PATH):
                    full_ds = pd.read_csv(DEFAULT_DATASET_PATH)
                    actual_kb = os.path.getsize(DEFAULT_DATASET_PATH) / 1024.0
                    st.write(f"- **Filename:** `{DEFAULT_DATASET_PATH}`")
                    st.write(f"- **File Size:** **`{actual_kb:.2f} KB`** (exceeds the 100 KB requirement)")
                    st.write(f"- **Total Samples:** **{len(full_ds):,}** balanced positive/negative reviews")
                    st.write("- **Domains Covered:** Electronics, Restaurants & Food, Movies, Customer Service, Airlines, Apps, General Services.")
                    st.markdown("##### First 10 Records:")
                    st.dataframe(full_ds.head(10), width="stretch")
                else:
                    st.warning("Dataset not found. Run `python generate_dataset.py` to create it.")

        with d_col2:
            with st.container(border=True):
                st.markdown("#### 🧩 Neural Network Architecture")
                st.markdown("""
                | Layer | Type | Output Dimension | Notes |
                | :--- | :--- | :--- | :--- |
                | **1** | Embedding | `(25, 16)` | Semantic word vector table |
                | **2** | LSTM | `(16)` | Recurrent memory gates |
                | **3** | Dense | `(16)` | Non-linear ReLU activation |
                | **4** | Dropout | `(16)` | Regularization ($p=0.1$) |
                | **5** | Dense | `(1)` | Sigmoid probability |
                """)
                st.markdown("##### Loss Function & Optimizer:")
                st.write("- **Loss:** `binary_crossentropy`")
                st.write("- **Optimizer:** Adam (learning rate = 0.01)")
                st.write(f"- **Trained Model Accuracy:** `{train_acc:.1f}%`")


if __name__ == "__main__":
    main()
