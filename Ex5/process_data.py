import numpy as np
from matplotlib import pyplot as plt
import os
from collections import Counter

np.random.seed(7)

# This list was taken from the NLTK package.
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'also', "there's"]


def read_data():
    root_path = os.path.join('review_polarity', 'txt_sentoken')
    pos, neg = {}, {}
    for class_name in ['pos', 'neg']:
        base_path = os.path.join(root_path, class_name)
        for filename in os.listdir(base_path):
            with open(os.path.join(base_path, filename), 'r') as fd:
                tokens = []
                for line in fd.readlines():
                    tokens.extend(line.strip('\n').split(' '))
                if class_name == 'pos':
                    pos[filename] = tokens
                else:
                    neg[filename] = tokens

    return pos, neg


def create_vocab(pos, neg, thr=5000):
    vocab = Counter()
    for subset in [pos, neg]:
        for doc in subset:
            vocab.update(subset[doc])
    
    ignore = [token for token in vocab if token in stopwords or len(token) < 3]
    for token in ignore:
        del vocab[token]

    return {x[0]:i for i,x in enumerate(vocab.most_common(thr))}


def get_feature_vectors(examples, vocab):
    vecs = []
    for doc in examples:
        vec = np.zeros(len(vocab))
        counts = Counter(examples[doc]).most_common()
        for cnt in counts:
            if cnt[0] in vocab:
                vec[vocab[cnt[0]]] = cnt[1]
        vecs.append(vec)
    return np.vstack(vecs)


def get_train_test_data(pos, neg, vocab):
    pos_vec = get_feature_vectors(pos, vocab)
    neg_vec = get_feature_vectors(neg, vocab)
    assert len(pos_vec) == len(neg_vec)
    np.random.shuffle(pos_vec)
    np.random.shuffle(neg_vec)
    split_idx = int(len(pos_vec) * 0.7)
    X_train = np.concatenate([pos_vec[:split_idx,:], neg_vec[:split_idx,:]])
    X_test = np.concatenate([pos_vec[split_idx:,:], neg_vec[split_idx:,:]])
    y_train = np.concatenate([np.ones(split_idx), np.ones(split_idx) * -1])
    y_test = np.concatenate([np.ones(len(pos_vec) - split_idx), np.ones(len(pos_vec) - split_idx) * -1])
    return X_train, y_train, X_test, y_test


def parse_data():
    data_file_name = 'review_polarity.tar.gz'
    data_dir_path = os.path.join('review_polarity', 'txt_sentoken')
    if not os.path.exists(data_file_name) and not os.path.exists(data_dir_path):
        print('Cannot find the data file, please download the file {} and place it in the same directory as the python files.'.format(data_file_name))
        return None
    if not os.path.exists(data_dir_path):
        print('Please extract the content of the file {}.'.format(data_file_name))
        return None
    pos, neg = read_data()
    vocab = create_vocab(pos, neg)
    inverse_vocab = {vocab[x]:x for x in vocab}
    X_train, y_train, X_test, y_test = get_train_test_data(pos, neg, vocab)

    return X_train, y_train, X_test, y_test, inverse_vocab





