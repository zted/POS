"""
This file processes different datasets into the same format so that
in training and evaluations we can easily switch between them
"""


def genia_split(line, secondTag=False):
    """
    In genia corpus, some words have two tags associated like so:
    'acetyl/JJ|NN'. The default option is to take the first tag,
    To take the second tag, set argument to True
    :param secondTag:
    :param line:
    :return:
    """
    l = len(line) - 1
    idx1 = l + 1
    tag = None
    for n in range(l, -1, -1):
        if line[n] == '|':
            if secondTag:
                tag = line[n + 1:idx1]
            # ^uncomment line to take second tag
            idx1 = n
        if line[n] == '/':
            tag = line[n + 1:idx1] if tag is None else tag
            word = line[0:n]
            word = word.replace(' ', '_space_')
            # there are spaces in the corpus, which messes up some
            # other parsing and processing models downstream
            return word, tag
    return None, None


def process_genia_dataset(infile, outfile='data/genia_processed.txt'):
    """
    Processes genia dataset. Outputs a file with 3 columns:
    nth word in sentece, word, POS tag
    :param infile:
    :param outfile:
    :return:
    """
    sentences = []
    alltags = []
    with open(infile, 'r') as f:
        new_sentence = []
        new_tags = []
        for line in f:
            line = line.rstrip('\n')
            if line == '====================' or line == '\n':
                sentences.append(new_sentence)
                alltags.append(new_tags)
                new_sentence = []
                new_tags = []
                continue
            word, tag = genia_split(line)
            if word is None:
                continue
            new_sentence.append(word)
            new_tags.append(tag)
    f = open(outfile, 'w')
    for m, sent in enumerate(sentences):
        tags = alltags[m]
        for n, word in enumerate(sent):
            f.write('{}\t{}\t{}\n'.format(n + 1, word, tags[n]))
        f.write('\n')
    f.close()
    return


def process_conll_dataset(infile, outfile='data/conll_processed.txt'):
    """
    Processes conll dataset. Outputs a file with 3 columns:
    nth word in sentece, word, POS tag
    :param infile:
    :param outfile:
    :return:
    """
    sentences = []
    alltags = []
    with open(infile, 'r') as f:
        new_sentence = []
        new_tags = []
        for line in f:
            splits = line.split('\t')
            if not splits[0].isdigit():
                sentences.append(new_sentence)
                alltags.append(new_tags)
                new_sentence = []
                new_tags = []
                continue
            tag = splits[4]
            word = splits[1]
            new_tags.append(tag)
            new_sentence.append(word)
    f = open(outfile, 'w')
    for m, sent in enumerate(sentences):
        tags = alltags[m]
        for n, word in enumerate(sent):
            f.write('{}\t{}\t{}\n'.format(n + 1, word, tags[n]))
        f.write('\n')
    f.close()
    return


def retrieve_sentences_tags(infile, maxlen=1000, allowedtags=[]):
    """
    Loads files processed according to the formats in other
    functions of this file into variables
    :param maxlen: discard sentences longer than this
    :param allowedtags: discard sentences containing tags not in this list.
    leave empty if any tags are allowed. this ensures the validation/test set
    will only use sentences containing same tags as training set
    :param infile: file to process
    :return: sentences, tags corresponding to each word in each sentence,
    set of unique words used contained in dataset, and set of unique tags
    contained in dataset
    """
    all_tags_allowed = True if len(allowedtags) == 0 else False
    sents = []
    truths = []
    words = set([])
    tags = set([])
    discard = False
    with open(infile, 'r') as f:
        new_sentence = []
        new_tags = []
        for n, line in enumerate(f):
            splits = line.split('\t')
            if not splits[0].isdigit():
                if len(new_sentence) <= maxlen and not discard:
                    sents.append(new_sentence)
                    truths.append(new_tags)
                    for w in new_sentence:
                        words.add(w)
                    for t in new_tags:
                        tags.add(t)
                new_sentence = []
                new_tags = []
                discard = False
                continue
            splits = line.split('\t')
            tag = splits[2]
            if (not all_tags_allowed) and (tag not in allowedtags):
                discard = True
            word_lower = splits[1].lower()
            new_tags.append(tag)
            new_sentence.append(word_lower)
    return sents, truths, words, tags


def sparsevec_to_binary(infile, outfile):
    skip = True
    allvecs = []
    with open(infile, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            splits = line.rstrip('\n').split(' ')
            if skip:
                h1 = splits[0]
                h2 = splits[1]
                skip = False
            else:
                word = splits[0]
                vectors = splits[1:-1]
                # dense to vec conversion code annoyingly puts a space after
                # the last number on each line, creating an extra element
                for n in range(len(vectors)):
                    vectors[n] = '1' if vectors[n] != '0' else vectors[n]
                allvecs.append([word] + vectors)

    f = open(outfile, 'w')
    f.write('{} {}\n'.format(h1, h2))
    for vector in allvecs:
        f.write(' '.join(vector) + '\n')
    f.close()


def create_input_data(data_x, data_y, x_dict, y_dict, maxlen):
    """
    Creates data to be fed into neural network
    :param data_x: sentences in list form, like [['he', 'is', 'jolly'],['she',...]]
    :param data_y: tags corresponding to data_y
    :param x_dict: dictionary that maps words to indices (integers)
    :param y_dict: dictionary that maps tags to indices (integers)
    :param maxlen: maximum length of a sentence so we know how much padding to use
    :return: x, y that can be fed to the embedding layer of an LSTM
    """
    from keras.preprocessing import sequence
    import numpy as np
    X_train = []
    Y_train = []
    for n, sent in enumerate(data_x):
        input = [x_dict[word] for word in sent]
        input = sequence.pad_sequences([input], maxlen=maxlen)
        output = [y_dict[tag] for tag in data_y[n]]
        output = sequence.pad_sequences([output], maxlen=maxlen)
        X_train.append(input[0])
        Y_train.append(output[0])
    return np.array(X_train), np.array(Y_train)


def train_embeddings(filename='./data/testsave.txt',
                     files=[], vocab_dim=200):
    import gensim
    sentences = []
    for f in files:
        sentence, _, _, _ = retrieve_sentences_tags(f)
        sentences = sentences + sentence
    gsm_mod = gensim.models.Word2Vec(sentences=sentences,
                                     size=vocab_dim, window=5, min_count=1,
                                     workers=4, iter=10)
    gsm_mod.save_word2vec_format(filename)
    gsm_mod.init_sims(replace=True)
    return