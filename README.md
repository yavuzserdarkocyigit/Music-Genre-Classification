# 🎵 Music Genre Classification with Traditional and Deep Learning Approaches

This project explores automatic music genre classification using both handcrafted audio features and deep learning-based representations. We compare multiple machine learning models—including Random Forest, KNN, SVM, and MLP—with a fine-tuned **Wav2Vec2** deep learning model to evaluate genre prediction performance on the GTZAN dataset.

## Dataset

- **Name**: GTZAN Music Genre Classification
- **Genres**: `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`
- **Samples**: 1000 `.wav` files (30 seconds each)
- **Balance**: 100 tracks per genre

Source: [GTZAN Dataset on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

---

## Feature Extraction

We used `Librosa` to extract a rich set of audio descriptors, including:

- **MFCCs** (20 coefficients → mean & var = 40 features)
- **Spectral Features**: centroid, bandwidth, rolloff
- **Chroma STFT**
- **Zero-Crossing Rate**
- **RMS Energy**
- **Tempo**

These were used to train classical ML models. Additionally, raw audio waveforms were used as input for Wav2Vec2.

---

## Models & Methodology

### Traditional ML Models 

| Model                    | Accuracy |
|-------------------------|----------|
| K-Nearest Neighbors     | 71%      |
| Random Forest (tuned)   | 73%      |
| MLP Classifier           | 68.5%    |
| Multinomial Logistic Reg| 69%      |
| One-vs-Rest Logistic Reg| 66%      |

✅ Feature importance analysis revealed Chroma STFT and MFCCs as key contributors.

### Deep Learning Model (Wav2Vec2)

- Pretrained: `facebook/wav2vec2-base`
- Input: Raw `.wav` → resampled to 16kHz
- Fine-tuned classifier on top of encoder (with dropout + pooling)
- Mean pooling applied across time axis
- Training: 20 epochs, batch size = 8

| Metric      | Value  |
|-------------|--------|
| **Accuracy**| **0.79** |
| Top Genres  | Classical, Jazz, Metal |
| Challenge   | Disco, Reggae (overlap with other genres) |

---

## 📈 Evaluation

- All models evaluated with:
  - Stratified 80-20 train-test split
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix (per genre)

- Wav2Vec2 provided highest generalization, especially for complex or ambiguous genres.

---

## 🧑‍💻 Authors

- **Yavuz Serdar Koçyiğit**   
- **Ozan Utku Çay** 
Özyeğin University – CS552

---

> 🎧 *Combining traditional audio descriptors and modern transformer architectures offers a powerful approach for music understanding and genre prediction.*
