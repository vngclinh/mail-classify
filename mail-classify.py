import pandas as pd
import re
import nltk
from math import log
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Tải dữ liệu NLTK cần thiết ---
# nltk.download('punkt')

# --- Đọc file và ánh xạ nhãn ---
df = pd.read_csv(r"C:\Users\OS\Downloads\spam.csv", encoding='latin1')[['Category', 'Message']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# --- Tiền xử lý văn bản ---
def preprocess(text):
    tokens = nltk.word_tokenize(text)
    tokens = [t.lower() for t in tokens]
    tokens = [re.sub(r'[^\w\s]', '', t) for t in tokens]
    tokens = [t for t in tokens if t]
    return tokens

df['tokens'] = df['message'].apply(preprocess)

# --- Tách tập huấn luyện và kiểm tra ---
x_train, x_test, y_train, y_test = train_test_split(df['tokens'], df['label'], test_size=0.2, random_state=42)
x_train = x_train.reset_index(drop=True)
x_test = x_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# --- Xây dựng từ vựng, tập tất cả các từ vựng có trong train ---
def build_vocabulary(X):
    vocab = set()
    for tokens in X:
        vocab.update(tokens)
    return sorted(list(vocab))

vocab = build_vocabulary(x_train)

# --- Biểu diễn mail bằng các vector với trọng số là số lần xuất hiện của từ đấy hoặc theo có/ không xuất hiện nếu binary=true  ---
def vectorize(tokens, vocab, binary=False):
    vec = [0] * len(vocab)
    for t in tokens:
        if t in vocab:
            idx = vocab.index(t)
            vec[idx] = 1 if binary else vec[idx] + 1
    return vec
#trọng số là số lần xuất hiện trong túi từ nếu binary false, ngược lại là theo sự xuất hiện
X_train_vec = [vectorize(tokens, vocab) for tokens in x_train]
X_test_vec  = [vectorize(tokens, vocab) for tokens in x_test]

# --- Đếm số lần xuất hiện theo lớp ham/spam ---
def get_token_counts_vec(X_vec, y, vocab):
    spam_counts = dict.fromkeys(vocab, 0)
    ham_counts = dict.fromkeys(vocab, 0)

    for vec, label in zip(X_vec, y):
        for i, count in enumerate(vec):
            word = vocab[i]
            if label == 1:
                spam_counts[word] += count
            else:
                ham_counts[word] += count
    return spam_counts, ham_counts

s_counts, h_counts = get_token_counts_vec(X_train_vec, y_train, vocab)

# --- Tính xác suất có điều kiện P(w|class) cho từng từ w với Laplace smoothing ---
def get_token_probs(count_dict):
    prob_dict = {}
    sum_counts = sum(count_dict.values()) + len(count_dict)
    for key in count_dict:
        prob_dict[key] = log((count_dict[key] + 1) / sum_counts)
    default_prob = log(1 / sum_counts)
    return prob_dict, default_prob

spam_probs, default_prob_spam = get_token_probs(s_counts)
ham_probs, default_prob_ham = get_token_probs(h_counts)

# --- Tính xác suất tiên nghiệm P(ham) và P(spam) ---
def get_log_priors(y):
    n_doc = len(y)
    n_spam = sum(y == 1)
    n_ham = sum(y == 0)
    return log(n_spam / n_doc), log(n_ham / n_doc)

spam_prior, ham_prior = get_log_priors(y_train)

# --- Dự đoán nhãn cho dữ liệu ---
def predict_vectorized(X_vec, vocab, spam_probs, ham_probs, def_spam, def_ham, spam_prior, ham_prior):
    preds = []
    for vec in X_vec:
        log_spam = spam_prior
        log_ham = ham_prior
        for i, count in enumerate(vec):
            word = vocab[i]
            log_spam += count * spam_probs.get(word, def_spam)
            log_ham += count * ham_probs.get(word, def_ham)
        preds.append(1 if log_spam > log_ham else 0)
    return preds

# --- Đánh giá mô hình ---
train_preds = predict_vectorized(X_train_vec, vocab, spam_probs, ham_probs, default_prob_spam, default_prob_ham, spam_prior, ham_prior)
test_preds = predict_vectorized(X_test_vec, vocab, spam_probs, ham_probs, default_prob_spam, default_prob_ham, spam_prior, ham_prior)

print("Training Accuracy:", round(accuracy_score(y_train, train_preds) * 100, 2), "%")
print("Test Accuracy:", round(accuracy_score(y_test, test_preds) * 100, 2), "%")

