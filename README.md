# 🔤 Autocorrect System using LSTM

## 📌 Overview

This project implements a **character-level autocorrect system** using a
Deep Learning model (LSTM).\
It takes misspelled words or sentences as input and predicts the
corrected output.

To improve accuracy, the system uses a **hybrid approach**: - Deep
Learning (LSTM) - Dictionary-based correction

------------------------------------------------------------------------

## 🚀 Features

-   Character-level autocorrection using LSTM\
-   Handles common spelling mistakes\
-   Supports both words and sentences\
-   Dictionary-based refinement for better accuracy\
-   Simple and interactive Streamlit UI

------------------------------------------------------------------------

## 🧠 Model Details

-   Input: Misspelled word (character sequence)\
-   Output: Corrected word\
-   Architecture:
    -   Embedding Layer\
    -   LSTM Layers\
    -   Dense (Softmax) Output\
-   Training Data:
    -   Generated from English word lists\
    -   Includes noise like:
        -   Character swap\
        -   Character deletion\
        -   Character insertion\
        -   Character replacement

------------------------------------------------------------------------

## 📂 Project Structure

    ├── autocorrect_lstm.ipynb
    ├── autocorrect_model.h5
    ├── vocab.pkl
    ├── config.pkl
    ├── base_words.pkl
    ├── requirements.txt
    ├── Autocorrect.pdf
    └── README.md

------------------------------------------------------------------------

## ⚙️ Installation

pip install tensorflow numpy

------------------------------------------------------------------------

## 🧪 Example

Input → Output\
recieve → receive\
frend → friend\
databse → database

------------------------------------------------------------------------

## ⚠️ Limitations

-   Character-level only (no sentence context)
-   May struggle with rare words
-   Not as powerful as transformer models

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Encoder--Decoder architecture\
-   Beam search decoding\
-   Context-aware correction\
-   Faster dictionary lookup

------------------------------------------------------------------------

## 👨‍💻 Contributors

-   Chidumulla Saipavan Reddy
-   23STUCHH010483
-   K.Vishwanath
-   23STUCHH010492
