#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Burcu Can
## Research Group in Computational Linguistics
## University of Wolverhampton

import dynet as dy
import codecs
from random import shuffle
import itertools



EOS = "<EOS>"
characters = list("aâbcçdefgğhıijklmnoöpqrsştuüvwxyzABCÇDEFGHI~u'İ'JKLMNOÖPRŞSTĞUÜVWYXZ@.+,;=_'\"?:!#()/-0123456789 ")
characters.append(EOS)

int2char = list(characters)
char2int = {c:i for i,c in enumerate(characters)}

VOCAB_SIZE = len(characters)

LSTM_NUM_OF_LAYERS = 1
EMBEDDINGS_SIZE = 128
STATE_SIZE = 128
ATTENTION_SIZE = 32


wordStems = {}
testSet = {}
zemberekwords = {}
allwords = []

def read_training_set():
    f = codecs.open("norm-data.txt", encoding='utf-8')
    for line in f:
        line = line.rstrip('\n')
        #line = line.encode('utf-8')
        #line = line.lower()
        noisy, norm = line.split('\t')
        #noisy = noisy.encode('utf-8')
        #norm = norm.encode('utf-8')
        zemberekwords[noisy] = norm


def read_training_set_conll():
    f = codecs.open("METUSABANCI_treebank_v-1.conll", encoding='utf-8')
    wordnum = 0
    word = ""
    stem = ""
    stembegins = 0
    for line in f:
        line = line.rstrip('\n')
        if len(line) > 1:
            line = line.lower()
            tokens = line.split('\t')
            if tokens[1] == '_' and stembegins == 0:
                stem = tokens[2]
                stembegins = 1
            if stembegins == 1 and tokens[1] != '_':
                wordStems[tokens[1]] = stem
                allwords.append(tuple((tokens[1], stem)))
                stembegins = 0
                wordnum += 1
            if tokens[1].isalnum() and tokens[2].isalnum():
                wordStems[tokens[1]] = tokens[2]
                allwords.append(tuple((tokens[1], tokens[2])))
                wordnum += 1
    print("Number of train words: " + str(wordnum))
    print("Number of train words: " + str(len(wordStems)))
    print("Number of test words: " + str(len(testSet)))


def embed_sentence(sentence, input_lookup):
    #print(sentence)
    sentence = [EOS] + list(sentence) + [EOS]
    #print(sentence)
    sentence = [char2int[c] for c in sentence]

    return [input_lookup[char] for char in sentence]



def run_lstm(init_state, input_vecs):
    s = init_state

    out_vectors = []
    for vector in input_vecs:
        s = s.add_input(vector)
        out_vector = s.output()
        out_vectors.append(out_vector)
    return out_vectors


def encode_sentence(enc_fwd_lstm, enc_bwd_lstm, sentence):
    sentence_rev = list(reversed(sentence))

    fwd_vectors = run_lstm(enc_fwd_lstm.initial_state(), sentence)
    bwd_vectors = run_lstm(enc_bwd_lstm.initial_state(), sentence_rev)
    bwd_vectors = list(reversed(bwd_vectors))
    vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

    return vectors


def attend(input_mat, state, w1dt, attention_w2, attention_v):
    w2 = dy.parameter(attention_w2)
    v = dy.parameter(attention_v)

    # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
    # w1dt: (attdim x seqlen)
    # w2dt: (attdim x attdim)
    w2dt = w2*dy.concatenate(list(state.s()))
    # att_weights: (seqlen,) row vector
    unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
    att_weights = dy.softmax(unnormalized)
    # context: (encoder_state)
    context = input_mat * att_weights
    return context


def decode(dec_lstm, vectors, output, decoder_w, decoder_b, attention_w1, output_lookup, attention_w2, attention_v):
    output = [EOS] + list(output) + [EOS]
    output = [char2int[c] for c in output]

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(vectors)
    w1dt = None

    last_output_embeddings = output_lookup[char2int[EOS]]
    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE*2), last_output_embeddings]))
    loss = []

    for char in output:
        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(input_mat, s, w1dt, attention_w2, attention_v), last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector)
        last_output_embeddings = output_lookup[char]
        loss.append(-dy.log(dy.pick(probs, char)))
    loss = dy.esum(loss)
    return loss


def generate(in_seq, enc_fwd_lstm, enc_bwd_lstm, dec_lstm, decoder_w, decoder_b, attention_w1, output_lookup, input_lookup,  attention_w2, attention_v):
    embedded = embed_sentence(in_seq, input_lookup)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(encoded)
    w1dt = None

    last_output_embeddings = output_lookup[char2int[EOS]]
    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_output_embeddings]))

    out = ''
    count_EOS = 0
    for i in range(len(in_seq)*2):
        if count_EOS == 2: break
        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(input_mat, s, w1dt,  attention_w2, attention_v), last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector).vec_value()
        next_char = probs.index(max(probs))
        last_output_embeddings = output_lookup[next_char]
        if int2char[next_char] == EOS:
            count_EOS += 1
            continue

        out += int2char[next_char]
    return out


def get_loss(input_sentence, output_sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm, decoder_w, decoder_b, attention_w1, attention_w2, attention_v, input_lookup, output_lookup):
    dy.renew_cg()
    embedded = embed_sentence(input_sentence, input_lookup)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)
    return decode(dec_lstm, encoded, output_sentence, decoder_w, decoder_b, attention_w1, output_lookup, attention_w2, attention_v)


def save_stemming_results2(testSet,enc_fwd_lstm, enc_bwd_lstm, dec_lstm, input_lookup, output_lookup, decoder_w, decoder_b, attention_w1,  attention_w2, attention_v):
    fo = open("stemming-results-attended.txt", "w")
    fo.seek(0)
    fo.truncate()
    true = 0
    for word, stem in testSet:
        predicted_stem = generate(word, enc_fwd_lstm, enc_bwd_lstm, dec_lstm, decoder_w, decoder_b, attention_w1, output_lookup, input_lookup,  attention_w2, attention_v)
        fo.write(word + '\t' + stem + '\t' + predicted_stem + '\n')
        if stem == predicted_stem:
            true += 1
    print("Accuracy: " + str(true / float(len(testSet))))
    return true / float(len(testSet))


def train_model(train, test, fold):

    model = dy.Model()

    enc_fwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)
    enc_bwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)

    dec_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE * 2 + EMBEDDINGS_SIZE, STATE_SIZE, model)

    input_lookup = model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE))
    attention_w1 = model.add_parameters((ATTENTION_SIZE, STATE_SIZE * 2))
    attention_w2 = model.add_parameters((ATTENTION_SIZE, STATE_SIZE * LSTM_NUM_OF_LAYERS * 2))
    attention_v = model.add_parameters((1, ATTENTION_SIZE))
    decoder_w = model.add_parameters((VOCAB_SIZE, STATE_SIZE))
    decoder_b = model.add_parameters((VOCAB_SIZE))
    output_lookup = model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE))


    trainer = dy.SimpleSGDTrainer(model)
    fo = open("train-loss-encoder-decoder-attended" + str(fold), "w")
    fo.seek(0)
    fo.truncate()
    fot = open("test-loss-encoder-decoder-attended" + str(fold), "w")
    fot.seek(0)
    fot.truncate()
    max_accuracy=0
    for i in range(40):
        total_loss = 0
        for word, stem in train:
            loss = get_loss(word, stem, enc_fwd_lstm, enc_bwd_lstm, dec_lstm, decoder_w, decoder_b, attention_w1, attention_w2, attention_v, input_lookup, output_lookup)
            #loss2
            loss_value = loss.value()
            loss.backward()
            trainer.update()
            total_loss += loss_value
        print(total_loss)
        fo.write(str(total_loss) + '\n')
        total_loss = 0
        for word, stem in test:
            loss = get_loss(word, stem, enc_fwd_lstm, enc_bwd_lstm, dec_lstm, decoder_w, decoder_b, attention_w1,
                            attention_w2, attention_v, input_lookup, output_lookup)
            loss_value = loss.value()
            total_loss += loss_value
        print(total_loss)
        fot.write(str(total_loss) + '\n')
        if i % 5 == 0:
            accuracy = save_stemming_results2(test,enc_fwd_lstm, enc_bwd_lstm, dec_lstm, input_lookup, output_lookup, decoder_w, decoder_b, attention_w1,  attention_w2, attention_v)
            if accuracy>max_accuracy:
                max_accuracy=accuracy
    fo.close()
    fot.close()
    model.save("saved-model")
    return max_accuracy

def splitDict(d):
    n = len(d) // 9
    i = iter(d.items())  # alternatively, i = d.iteritems() works in Python 2

    d1 = dict(itertools.islice(i, n))  # grab first n items
    d2 = dict(i)  # grab the rest
    return d1, d2


def splitDict2(d):
    test = len(d)//10
    return d[:test], d[test:]


def remove_duplicates(d):
    table = {}
    var=0
    for word, stem in d:
        if word in table:
            var=0
            for other_stem in table[word]:
                if other_stem ==stem:
                    var=1
            if var==0:
                table[word].append(stem)
        else:
            table[word]=[stem]
    new_list = []
    for word, stem_list in table.items():
        for st in stem_list:
            new_list.append(tuple((word, st)))
    shuffle(new_list)
    return new_list


def save_test_set(testSet, fold):
    file_name ="metu-stem-test-fold-attended" +str(fold)+".txt"
    fo = open(file_name, "w")
    for word, stem in testSet:
        fo.write(word + '\t' + stem + '\n')
    fo.close()


#read_training_set_conll()
read_training_set()
for word, stem in zemberekwords.items():
    if word not in wordStems:
        wordStems[word] = stem
        allwords.append(tuple((word, stem)))

allwords = remove_duplicates(allwords)
#testSet, wordStems = splitDict(wordStems)
#alltest, allwords = splitDict2(allwords)

#Add half of the test set into the training set.
#count = int(len(alltest)/2)
#for i in range (0,len(alltest)):
#    allwords.append(alltest[i])

#make_test_set2()
# Extend the set with Zemberek words.
#read_training_set()
#for word, stem in zemberekwords.items():
#    if word not in wordStems:
#        wordStems[word] = stem
#        allwords.append(tuple((word, stem)))

#co = 0
#for word in wordStems.copy():
#    if word in testSet:
#        co+=1
#        wordStems.pop(word)
#        i=0
#        for w, s in allwords:
#            if w == word:
#                del allwords[i]
#                break
#            i+=1

#print(co)


def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

'''''
c = chunks(allwords, 100)


total_acc = 0
for f in range(100):
    print('FOLD: ' + str(f))
    train = []
    test = list(chunks(allwords, len(allwords)//100))[f]
    for other in range(100):
        if f != other:
            train+=list(chunks(allwords, len(allwords)//100))[other]
    save_test_set(test, f)
    max_accuracy = train_model(train, test, f)
    total_acc += max_accuracy


print('Average accuracy of 10 folds is: ' + str(total_acc/100.0))
'''

#print("Number of TOTAL train words: " + str(len(allwords)))
#print("Number of TOTAL test words: " + str(len(alltest)))

train_model(allwords, allwords, 0)
