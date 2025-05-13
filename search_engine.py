import re
import numpy as np
from collections import defaultdict, Counter
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet


def load_stop_words(path='common_words.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        return set(word.strip().lower() for word in f if word.strip())


def preprocess(text, use_stemming=True, remove_stopwords=True, stop_words=None, stemmer=None):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text) # Remove punctuation
    tokens = text.split()
    if remove_stopwords and stop_words:
        tokens = [word for word in tokens if word not in stop_words]
    if use_stemming and stemmer:
        tokens = [stemmer.stem(word) for word in tokens]
    return tokens


def load_documents(path, use_stemming, remove_stopwords, stop_words, stemmer):
    docs = []
    titles = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                title, content = line.split(':', 1)
                titles.append(title.strip())
                tokens = preprocess(content.strip(), use_stemming, remove_stopwords, stop_words, stemmer)
                docs.append(tokens)
    return titles, docs


def compute_tf(documents):
    df = defaultdict(int)
    vocab = set()

    for doc in documents:
        seen = set()
        for term in doc:
            vocab.add(term)
            if term not in seen:
                df[term] += 1
                seen.add(term)
    return sorted(vocab), df



def compute_tfidf(documents, vocab, df, total_docs):
    idf = {}
    for term in vocab:
        document_frequency_of_term = df.get(term, 0) 
        numerator = total_docs
        denominator = 1 + document_frequency_of_term
        idf_value = np.log(numerator / denominator)
        idf[term] = idf_value
        
    tfidf_vectors = []
    for doc in documents:
        tf = Counter(doc)
        doc_len = len(doc)
        vector = []
        for term in vocab:
            tf_val = tf[term] / doc_len if doc_len > 0 else 0
            vector.append(tf_val * idf[term])
        tfidf_vectors.append(np.array(vector))
    return tfidf_vectors


def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0.0


def search_images(image_titles, caption_vecs, query_vecs, top_n=5):
    results = []
    for query_vec in query_vecs:
        scores = [cosine_similarity(query_vec, cap_vec) for cap_vec in caption_vecs]
        top_indices = np.argsort(scores)[::-1][:top_n]
        top_images = [(image_titles[i], scores[i]) for i in top_indices]
        results.append(top_images)
    return results


def expand_query(query):
    expanded_query = set()
    for word in query:
        expanded_query.add(word) 
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ")
                expanded_query.add(synonym)
    return list(expanded_query) 


def run_experiment(permutation, use_stemming, remove_stopwords, use_thesaurus, output_file):
    output_file.write(f"\n=== {permutation} ===\n")

    stop_words = load_stop_words() if remove_stopwords else None
    stemmer = SnowballStemmer('english') if use_stemming else None

    img_titles, captions = load_documents('captions.txt', use_stemming, remove_stopwords, stop_words, stemmer)
    qry_titles, queries = load_documents('queries.txt', use_stemming, remove_stopwords, stop_words, stemmer)

    origninal_quereies = queries

    if use_thesaurus:
        captions = [expand_query(doc) for doc in captions]
        queries = [expand_query(doc) for doc in queries]

    all_docs = captions + queries
    vocab, df = compute_tf(all_docs)
    total_docs = len(all_docs)

    caption_vecs = compute_tfidf(captions, vocab, df, total_docs)
    query_vecs = compute_tfidf(queries, vocab, df, total_docs)

    results = search_images(img_titles, caption_vecs, query_vecs)

    for i, (query, top) in enumerate(zip(qry_titles, results)):
        textQuery = (" ".join(origninal_quereies[i]))
        output_file.write(f"\n{query}: {textQuery} \n")
        output_file.write("Top Matches:\n")
        for title, score in top:
            output_file.write(f"{title} (Score: {score:.4f})\n")


def main():
    experiments = [
        ("No Preprocessing", False, False, False),
        ("Stemming Only", True, False, False),
        ("Stopwords Only", False, True, False),
        ("Stemming + Stopwords", True, True, False),
        ("Stemming + Stopwords + Thesaurus", True, True, True),
    ]

    with open("results.txt", "w", encoding="utf-8") as output_file:
        for permutation, stem, stop_words, thesaurus in experiments:
            run_experiment(permutation, stem, stop_words, thesaurus, output_file)


if __name__ == "__main__":
    main()
