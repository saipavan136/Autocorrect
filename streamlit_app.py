import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from difflib import get_close_matches

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = "autocorrect_model.keras"
VOCAB_PATH  = "vocab.pkl"
CONFIG_PATH = "config.pkl"

# ── Load everything once ──────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with open(VOCAB_PATH, "rb") as f:
        vocab, inv_vocab = pickle.load(f)

    with open(CONFIG_PATH, "rb") as f:
        config = pickle.load(f)

    return model, vocab, inv_vocab, config["max_len"]

model, VOCAB, INV_VOCAB, MAX_LEN = load_all()

# ── Token constants (must match what the notebook used) ───────────────────────
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

# ── Encode ────────────────────────────────────────────────────────────────────
def encode(word):
    """
    BUG 1 FIX: original encode_word had no SOS/EOS tokens.
    The model was trained with [SOS, c1, c2, ..., EOS, PAD, PAD, ...]
    so the input MUST follow the same layout or predictions are garbage.
    """
    seq  = [VOCAB[SOS_TOKEN]]
    seq += [VOCAB.get(ch.lower(), VOCAB[PAD_TOKEN]) for ch in word]
    seq += [VOCAB[EOS_TOKEN]]
    seq  = seq[:MAX_LEN]
    seq += [VOCAB[PAD_TOKEN]] * (MAX_LEN - len(seq))
    return np.array(seq, dtype=np.int32)

# ── Decode ────────────────────────────────────────────────────────────────────
def decode_sequence(pred_probs, max_output_len):
    """
    BUG 2 FIX: original decode_output used 'continue' on PAD/EOS so it
    kept reading garbage from padded positions → runaway repetition.

    BUG 3 FIX: no length cap, so even if EOS isn't predicted cleanly the
    output is capped at len(input) + 3 characters.
    """
    result = []
    for step in pred_probs:
        idx   = int(np.argmax(step))
        token = INV_VOCAB.get(idx, "")

        # Hard stop — end of meaningful output
        if token in (EOS_TOKEN, PAD_TOKEN, ""):
            break

        # Skip start token
        if token == SOS_TOKEN:
            continue

        result.append(token)

        # Length cap — reliable fallback when model doesn't predict EOS cleanly
        if len(result) >= max_output_len:
            break

    return "".join(result)

# ── Dictionary snap ───────────────────────────────────────────────────────────
@st.cache_data
def load_base_words():
    """Load the same word list the notebook was trained on."""
    from wordfreq import top_n_list
    raw = top_n_list("en", 10_000)
    return list({
        w.lower() for w in raw
        if len(w) >= 3 and w.isalpha() and len(w) <= MAX_LEN - 2
    })

BASE_WORDS = load_base_words()

def dictionary_snap(word):
    matches = get_close_matches(word, BASE_WORDS, n=1, cutoff=0.6)
    return matches[0] if matches else word

# ── Predict ───────────────────────────────────────────────────────────────────
def predict_word(word):
    word  = word.strip().lower()
    x     = encode(word)[np.newaxis, :]          # shape (1, MAX_LEN)
    pred  = model.predict(x, verbose=0)[0]       # shape (MAX_LEN, vocab_size)

    decoded = decode_sequence(pred, max_output_len=len(word) + 3)

    base  = decoded if decoded else word
    return dictionary_snap(base)

# ── UI (unchanged) ────────────────────────────────────────────────────────────
st.title("🔤 Autocorrect LSTM App")

user_input = st.text_input("Enter a misspelled word:")

if user_input:
    corrected = predict_word(user_input)
    st.write("### ✅ Corrected Word:")
    st.success(corrected)