# AI Sentiment Analyzer Using NLP and LSTM

An interactive, end-to-end Sentiment Analysis web application powered by **Natural Language Processing (NLP)**, a **Long Short-Term Memory (LSTM)** neural network, and **Streamlit**.

Supports both **real-time single text analysis** and **batch CSV dataset upload & analysis** (with included 203 KB / 2,200-row benchmark dataset).

---

## 📌 Project Overview
Sentiment Analysis is a fundamental Natural Language Processing task that classifies the emotional tone behind human language. Unlike traditional bag-of-words or Naive Bayes models, **Recurrent Neural Networks (RNNs)** like **LSTM** capture word order, syntactic context, and long-term sequential dependencies.

This project delivers:
1. **Interactive Single Text Analyzer:** Enter any review, select sample sentences, view predicted sentiment (Positive/Negative), confidence gauge, raw probability, and vocabulary tokens.
2. **Batch CSV Upload & Processing:** Upload any CSV file containing customer reviews or feedback (**supports datasets $\ge$ 100 KB**).
3. **Included Benchmark Dataset (`dataset.csv`):** A 203.24 KB balanced dataset containing 2,200 reviews across tech, movies, food, customer support, and travel.
4. **Dataset Generator (`generate_dataset.py`):** Script to reproduce or customize the 100+ KB dataset anytime.
5. **Batch Visualizations & Export:** Summary metrics (positive %, negative %, avg confidence), interactive sentiment bar chart, annotated data table, and one-click CSV export.

---

## 🛠️ Technology Stack
- **Python 3.12** — Core programming language
- **TensorFlow / Keras** — Neural network construction, training, and inference
- **Natural Language Processing (NLP)** — Tokenization, sequence transformation, and padding
- **Streamlit** — Web interface with tabbed navigation and real-time state management
- **Pandas & NumPy** — Tabular data manipulation, batch matrix preprocessing, and evaluation

---

## 📐 Model Architecture & Pipeline

```text
Raw Text / CSV Review Column
      │
      ▼
[NLP Preprocessing: Lowercase, URL/Regex Symbol Stripping]
      │
      ▼
[Keras Tokenizer: Integer Sequence Mapping with <OOV> Fallback]
      │
      ▼
[pad_sequences: Standardized Length = 25 Tokens]
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Keras Sequential Neural Network                        │
│ 1. Embedding Layer: Vocab Size = 1000, Vector Dim = 16 │
│ 2. LSTM Layer: 16 Memory Units (Dropout: 0.1)          │
│ 3. Dense Hidden Layer: 16 Units (ReLU Activation)      │
│ 4. Dropout Regularization: Rate = 0.1                  │
│ 5. Dense Output Layer: 1 Unit (Sigmoid Activation)     │
└────────────────────────────────────────────────────────┘
      │
      ▼
Sigmoid Probability [0.0 - 1.0]
      │
      ├── Prob >= 0.50 ➔ 🎉 POSITIVE Sentiment (Confidence: Prob × 100%)
      └── Prob <  0.50 ➔ ⚠️ NEGATIVE Sentiment (Confidence: (1 - Prob) × 100%)
```

---

## 📁 Dataset Details (`dataset.csv`)

| Property | Value |
| :--- | :--- |
| **File Path** | `dataset.csv` |
| **File Size** | **203.24 KB** (exceeds the 100 KB requirement) |
| **Total Rows** | **2,200 records** (1,100 Positive, 1,100 Negative) |
| **Columns** | `id`, `review`, `sentiment`, `label` |
| **Domains** | Electronics, Food & Dining, Movies, Customer Care, Hotels, Books, Apps |
| **Regeneration** | `python generate_dataset.py` |

---

## 🚀 Quickstart & Execution Steps

### 1. Prerequisites
Ensure Python 3.10, 3.11, or 3.12 is installed.

### 2. Set Up Virtual Environment (Recommended)
```bash
# Create virtual environment
python3.12 -m venv venv

# Activate virtual environment
# On macOS / Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Regenerate Dataset
```bash
python generate_dataset.py
```

### 5. Run the Streamlit Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🖥️ Application Features & Walkthrough

### Tab 1: Single Text Analyzer
- **Direct Input:** Enter any sentence or review.
- **Quick Test Buttons:** One-click samples for positive movies, delicious food, broken products, and rude support.
- **Visual Feedback:** Colored sentiment cards (Emerald Green for Positive, Crimson Red for Negative).
- **Metrics:** Model estimated confidence %, progress gauge, raw sigmoid probability, and recognized token inspector.
- **Clear Button:** Resets the input area.

### Tab 2: Batch CSV Upload & Analysis
- **CSV Uploader:** Upload any CSV file from your disk.
- **Built-in 203 KB Dataset Loader:** One-click button to load `dataset.csv` without needing to upload manually.
- **Column Selector:** Automatically detects text/review columns or lets you select any column.
- **Batch Processing:** Runs fast vectorized LSTM inference over hundreds or thousands of rows.
- **Analytics Dashboard:** Total analyzed, positive/negative count & percentage, average confidence.
- **Interactive Bar Chart:** Visual distribution of predicted sentiments.
- **Ground Truth Evaluation:** Compares predictions with true labels if available.
- **Export Results:** Download the predictions as an annotated CSV file.

### Tab 3: Dataset & Architecture Explorer
- Inspect dataset properties, sample rows, and layer-by-layer neural network parameters.

---

## 🧪 Sample Inputs and Expected Outputs

| Input Sentence / Review | Predicted Sentiment | Sigmoid Probability | Confidence |
| :--- | :---: | :---: | :---: |
| *"I love this product, it is amazing and works great!"* | **Positive** | `> 0.95` | `> 95%` |
| *"The food at this restaurant was delicious and freshly cooked."* | **Positive** | `> 0.95` | `> 95%` |
| *"This product is terrible, broke on the first day of use."* | **Negative** | `< 0.05` | `> 95%` |
| *"Terrible movie, absolute waste of time and horrible acting."* | **Negative** | `< 0.05` | `> 95%` |

---

## 🎓 College Viva / Practical Q&A

### Q1: What is an LSTM and why is it preferred over standard SimpleRNN for text classification?
> **Answer:** Standard Recurrent Neural Networks (SimpleRNN) suffer from the **vanishing and exploding gradient problem** over longer sequences, causing them to forget earlier context. An **LSTM (Long Short-Term Memory)** network overcomes this by introducing memory cells regulated by three gating mechanisms:
> 1. **Forget Gate:** Decides what information to discard from the cell state.
> 2. **Input Gate:** Determines which new information to store in the cell state.
> 3. **Output Gate:** Controls what information from the cell state is output as hidden state.
> This allows LSTMs to preserve sequential context across sentences effectively.

### Q2: What is the role of the Embedding Layer in this model?
> **Answer:** The Embedding Layer maps high-dimensional, sparse one-hot encoded word vectors into dense, continuous numerical vectors of fixed dimensions (here, 16 dimensions). Words with similar semantic meanings or contexts are positioned closer together in this continuous embedding space through backpropagation.

### Q3: Why is `pad_sequences` necessary before feeding text into an LSTM?
> **Answer:** Deep learning models, tensor math, and batch training require uniform input shapes. Since human sentences vary in length, `pad_sequences` ensures all input sequences have a standardized length (here, 25 tokens) by padding shorter sequences with zeros or truncating longer ones.

### Q4: Why is the `sigmoid` activation function used in the final layer?
> **Answer:** Sentiment analysis in this project is formulated as a binary classification task (Positive vs. Negative). The `sigmoid` function squashes any real-valued number into a probability range between $0$ and $1$. A threshold of $0.5$ is applied: values $\ge 0.5$ denote Positive sentiment, while values $< 0.5$ denote Negative sentiment.

### Q5: How does the application handle batch predictions on large uploaded CSV datasets?
> **Answer:** Rather than executing inference one row at a time in Python loops, the app cleans and tokenizes all reviews, standardizes them into a unified NumPy matrix using `pad_sequences`, and passes the batch tensor directly to `model.predict(padded, batch_size=64)`. This utilizes optimized vectorized C++ routines inside TensorFlow, allowing thousands of rows in a 100+ KB CSV to be classified in just 1–2 seconds.

---

## 📄 License
Open-source and intended for academic demonstration and laboratory practicals.
