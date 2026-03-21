# 🔤 Autocorrect System using Bidirectional LSTM

> A character-level autocorrect system built with Deep Learning.  
> Enter a misspelled word — get the correct one back.

**🚀 Live Demo:** [autocorrectgit-nzqgztbbhcktdwy7qorar9.streamlit.app](https://autocorrectgit-nzqgztbbhcktdwy7qorar9.streamlit.app/)

---

## 📌 Overview

This project implements a **character-level autocorrect system** using a Bidirectional LSTM neural network. It maps misspelled character sequences to their corrected forms using a **two-stage hybrid pipeline**:

1. **LSTM Stage** — Bidirectional LSTM predicts the corrected character sequence
2. **Dictionary Snap** — `difflib.get_close_matches` maps the output to the nearest real English word

This hybrid approach outperforms either stage alone — the LSTM handles phonetic/structural correction while the dictionary snap ensures the output is always a valid word.

---

## 🚀 Features

- Character-level autocorrection using a **Bidirectional LSTM**
- Handles 5 types of realistic spelling mistakes
- Supports both single words and multi-word phrases
- Length-capped decoding prevents runaway character repetition
- Dictionary refinement using `difflib` for guaranteed valid output
- Simple and interactive Streamlit UI (no button needed — corrects on type)
- Model and vocab cached with `@st.cache_resource` for fast inference

---

## 🧠 Model Architecture

```
Input  (18,)
  └── Embedding       vocab_size=41 → 64 dims   mask_zero=True
  └── Bidirectional   LSTM(128 units)            return_sequences=True
  └── Dropout         rate=0.20
  └── TimeDistributed Dense(41, softmax)
Output (18, 41)
```

| Layer | Output Shape | Parameters |
|---|---|---|
| Embedding | (None, 18, 64) | 2,624 |
| Bidirectional LSTM | (None, 18, 256) | 197,632 |
| Dropout | (None, 18, 256) | 0 |
| TimeDistributed Dense | (None, 18, 41) | 10,537 |
| **Total** | | **210,793** |

**Why Bidirectional?**  
A standard LSTM reads left-to-right only. A Bidirectional LSTM reads the word in both directions and concatenates the results — giving the model full left and right context at every character position. This is especially useful for corrections like `recieve → receive` where the error sits in the middle of the word.

---

## 📊 Training

| Setting | Value |
|---|---|
| Dataset size | 100,000 (misspelled, correct) pairs |
| Clean pairs (no noise) | 30% — model learns not to alter correct words |
| MAX_LEN | 18 — supports words up to 16 characters |
| Optimizer | Adam |
| Loss | Sparse Categorical Cross-Entropy |
| Batch size | 256 |
| Max epochs | 50 |
| Early stopping | patience=5, restores best weights |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Hardware | Google Colab T4 GPU (~5 min) |

---

## 🧪 Prediction Pipeline

```python
def correct_word(text):
    for token in text.split():
        x       = encode(token)                          # [SOS, c1, c2, ..., EOS, PAD...]
        pred    = model.predict(x)                       # (18, 41) probability matrix
        decoded = decode_sequence(pred,
                    max_output_len=len(token) + 3)       # length-capped greedy decode
        result  = dictionary_snap(decoded)               # snap to nearest real word
```

**Three bugs that were fixed in the pipeline:**

| # | Bug | Symptom | Fix |
|---|---|---|---|
| 1 | Missing SOS/EOS in `encode()` | All predictions random | Add `[SOS] + chars + [EOS]` |
| 2 | `continue` on PAD/EOS instead of `break` | Runaway repetition (`comturereeeee`) | `break` on PAD or EOS |
| 3 | No output length cap | Repetition even after fix #2 | Stop at `len(input) + 3` chars |

---

## ✅ Example Results

| Input | Output |
|---|---|
| `teh` | `the` |
| `enviroment` | `environment` |
| `recieve` | `receive` |
| `frend` | `friend` |
| `languge` | `language` |
| `comuter` | `computer` |
| `databse` | `database` |
| `algorythm` | `algorithm` |

---

## 🔡 Vocabulary & Tokenisation

- **41 tokens**: `<PAD>` (0), `<SOS>` (1), `<EOS>` (2), `a–z` (3–28), `0–9` (29–38), `'` (39), `-` (40)
- **MAX_LEN = 18**: each sequence is `[SOS, c1…c16, EOS, PAD…]`
- Unknown characters map to `<PAD>` (index 0)

---

## 🗂️ Project Structure

```
├── autocorrect_lstm.ipynb     # Training notebook (run on Google Colab)
├── app.py                     # Streamlit app
├── autocorrect_model.keras    # Trained Bidirectional LSTM (native Keras format)
├── vocab.pkl                  # (VOCAB, INV_VOCAB) character dictionaries
├── config.pkl                 # {max_len: 18, vocab_size: 41}
├── requirements.txt           # Python dependencies
├── Autocorrect_report.docx    # Full technical report
└── README.md
```

> **Note:** Model is saved as `.keras` (native Keras format, TF 2.16+).  
> Do **not** use `.h5` on TF 2.16+ — it is considered legacy and triggers deprecation warnings.

---

## ⚙️ Installation & Running Locally

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd <repo-folder>

# 2. Install dependencies
pip install tensorflow>=2.16 streamlit numpy wordfreq

# 3. Run the app
streamlit run app.py
# Open http://localhost:8501
```

---

## ⚠️ Limitations

- Corrects each word in isolation — no sentence-level context
- Words not in the top-10,000 English word list may not snap correctly
- Words longer than 16 characters are not supported
- Not as powerful as transformer-based models (e.g., BERT, T5)

---

## 🔮 Future Improvements

- [ ] Encoder–Decoder architecture with attention
- [ ] Beam search decoding (vs. greedy argmax)
- [ ] Sentence-level context awareness
- [ ] Transformer-based correction (T5 / BERT fine-tune)
- [ ] Faster dictionary lookup with BK-tree or trie

---

## 👨‍💻 Contributors

| Name | Roll Number |
|---|---|
| Chidumulla Saipavan Reddy | 23STUCHH010483 |
| K. Vishwanath | 23STUCHH010492 |

**Course:** Deep Learning Mini Project
