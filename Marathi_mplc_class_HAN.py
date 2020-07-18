import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer,  text_to_word_sequence
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from keras import backend as K
from keras import optimizers
from keras.models import Model
import nltk
import re
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import roc_auc_score
from nltk import tokenize
import seaborn as sns
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import os
from sklearn.metrics import roc_curve, auc
import itertools
from scipy import interp
from keras.models import model_from_json
from itertools import cycle
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def model_HAN():
    embedding_layer = Embedding(max_words,100,input_length=max_senten_len, trainable=False)
    word_input = Input(shape=(max_senten_len,), dtype='float32')
    word_sequences = embedding_layer(word_input)
    word_lstm = Bidirectional(LSTM(150, return_sequences=True, kernel_regularizer=l2_reg))(word_sequences)
    word_dense = TimeDistributed(Dense(200, kernel_regularizer=l2_reg))(word_lstm)
    word_att = AttentionWithContext()(word_dense)
    wordEncoder = Model(word_input, word_att)

    sent_input = Input(shape=(max_senten_num, max_senten_len), dtype='float32')
    sent_encoder = TimeDistributed(wordEncoder)(sent_input)
    sent_lstm = Bidirectional(LSTM(150, return_sequences=True, kernel_regularizer=l2_reg))(sent_encoder)
    sent_dense = TimeDistributed(Dense(200, kernel_regularizer=l2_reg))(sent_lstm)
    sent_att = Dropout(0.5)(sent_dense)
    sent = AttentionWithContext() (sent_att)
    preds = Dense(2, activation='softmax')(sent)
    model = Model(sent_input, preds)
    return model,  wordEncoder  
class Splitter(object):
    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()
        
    def split(self, text):
        """
	input format: a paragraph of text
	output format: a list of lists of words.
	e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
	"""
        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_sentences
def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatibl|e with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithContext(Layer):
    
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

max_words=200
max_senten_len=20
max_senten_num=6
embed_size=100
VALIDATION_SPLIT = 0.2
text = []
def createList(foldername, fulldir = True, suffix=".jpg"):
    file_list_tmp = os.listdir(foldername)
    #print len(file_list_tmp)
    file_list = []
    if fulldir:
        for item in file_list_tmp:
            if item.endswith(suffix):
                file_list.append(os.path.join(foldername, item))
    else:
        for item in file_list_tmp:
            if item.endswith(suffix):
                file_list.append(item)
    return file_list

file1=createList("E:/rad_phd/mplc_tour/neg1",suffix=".txt")

file2=createList("E:/rad_phd/mplc_tour/pos1",suffix=".txt")
fname1 = []
fname2 = []
cates = []
for n in file1:
    fname1.append(n)
for n in file2:
    fname2.append(n)
for name in fname1:
    splitter = Splitter()
    fin=open(name,"r",encoding='utf8')
    text.append(fin.read())
    cates.append("neg")
print(text[0])    
print(len(text))
print(len(cates))
print(cates[0])
for name in fname2:
    splitter = Splitter()
    fin=open(name,"r",encoding='utf8')
    text.append(fin.read())
    cates.append("pos")
print(len(text))
print(len(cates))
print(cates[76])
print(text[76])

print(cates)
tags = [tag for  tag in cates]
#print(tags)

sns.countplot(cates)
plt.xlabel('Label')
plt.title('number of poems of each category')
plt.show()
	le = LabelEncoder()
	le.fit(cates)
	labels = le.transform(cates)
dummy_y = np_utils.to_categorical(labels)


#exit(0)    
paras = []
labels = []
texts = []
sent_lens = []
sent_nums = []
for idx in range(len(text)):
    text1 = text[idx]
    texts.append(text1)
    sentences = tokenize.sent_tokenize(text1)
    sent_nums.append(len(sentences))
    for sent in sentences:
        sent_lens.append(len(text_to_word_sequence(sent)))
    paras.append(sentences)

sns.distplot(sent_lens, bins=200)
plt.show()
sns.distplot(sent_nums)
plt.show()
tokenizer = Tokenizer(num_words=max_words, oov_token=True)
tokenizer.fit_on_texts(texts)
data = np.zeros((len(texts), max_senten_num, max_senten_len), dtype='int32')
print(data.shape)
for i, sentences in enumerate(paras):
    for j, sent in enumerate(sentences):
        #print(j,sent)
        if j< max_senten_num:
            wordTokens = text_to_word_sequence(sent)
       #     print(wordTokens)
            k=0
            for _, word in enumerate(wordTokens):
                try:
                    
                    if k<max_senten_len and tokenizer.word_index[word]<max_words:
                        data[i,j,k] = tokenizer.word_index[word]
                        k=k+1
                       # print(data)
                        
                except:
                   # print(word)
                    pass

print(data.shape)

word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))
#labels = pd.get_dummies(cates)
print('Shape of data tensor:', data.shape)
labels=np.asarray(dummy_y)
print('Shape of labels tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])


#x_train = data[:-nb_validation_samples]
#y_train = labels[:-nb_validation_samples]
#x_val = data[-nb_validation_samples:]
#y_val = labels[-nb_validation_samples:]

x_train,x_test , y_train, y_test = train_test_split(data,labels, test_size=0.2, random_state=42,stratify=labels)

REG_PARAM = 1e-13
l2_reg = regularizers.l2(REG_PARAM)
model, wordmodel = model_HAN()
#model = modelbuild()
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

checkpoint = ModelCheckpoint('best_model.h5', verbose=0, monitor='val_loss',save_best_only=True, mode='auto')
history = model.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=16, callbacks=[checkpoint])

print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#shuffle=False,
#,validation_data=(test_sequences_matrix, testLabels),
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='validation')
#pyplot.legend()
#pyplot.show()
                     
def plot_model_performance(train_loss, train_acc, train_val_loss, train_val_acc):
    """ Plot model loss and accuracy through epochs. """

    blue= '#34495E'
    green = '#2ECC71'
    orange = '#E23B13'

    # plot model loss
    #fig, (ax1, ax2) = plt.subplots(2, figsize=(9, 7))
    ax1=plt
    ax1.plot(range(1, len(train_loss) + 1), train_loss, blue, linewidth=5, label='training')
    ax1.plot(range(1, len(train_val_loss) + 1), train_val_loss, green, linewidth=5, label='validation')
    ax1.xlabel('# epoch')
    ax1.ylabel('loss')
   # ax1.tick_params('y')
    ax1.grid(True)
    ax1.legend(loc='upper right', shadow=False)
    ax1.title('Model loss through #epochs', color=orange, fontweight='bold')
    ax1.show()   
    # plot model accuracy
    ax2=plt
    ax2.plot(range(1, len(train_acc) + 1), train_acc, blue, linewidth=5, label='training')
    ax2.plot(range(1, len(train_val_acc) + 1), train_val_acc, green, linewidth=5, label='validation')
    ax2.xlabel('# epoch')
    ax2.ylabel('accuracy')
    #ax2.tick_params('y')
    ax2.grid(True)
    ax2.legend(loc='lower right', shadow=False)
    ax2.title('Model accuracy through #epochs', color=orange, fontweight='bold')
    ax2.show()
plot_model_performance(
    train_loss=history.history.get('loss', []),
    train_acc=history.history.get('acc', []),
    train_val_loss=history.history.get('val_loss', []),
    train_val_acc=history.history.get('val_acc', [])
)

from keras.utils import plot_model

plot_model(model, to_file='poem_stop_model.png', show_shapes=True)
plot_model(wordmodel, to_file='poem_word_model.png', show_shapes=True)

#test_sequences = tok.texts_to_sequences(x_test)
#test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

accr = model.evaluate(x_test,y_test,verbose=1)
y_pred = model.predict(x_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
print(y_pred[0])
y_p_new=[]

classes = ['neg','pos']

for i in range(len(y_pred)):
    ind = 0
    max = y_pred[i][0]
    for j in range(1,len(y_pred[0])):
        if max < y_pred[i][j] :
            max = y_pred[i][j]
            
            ind = j
    
    y_p_new.append(classes[ind])
print(y_p_new[0])
le.fit(cates)
y_p = le.transform(y_p_new)
y_p = np_utils.to_categorical(y_p)    
print(y_p)

precision = dict()
recall = dict()
ther = dict()
average_precision = dict()
print(classification_report(y_test, y_p, target_names=classes))
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),y_p.ravel())
average_precision["micro"] = average_precision_score(y_test, y_p,average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))


    
test_new = []
for i in range(len(y_test)):
    for j in range(2):
        if (y_test[i][j]==1.0):
            test_new.append(classes[j])
            
cnf_matrix = confusion_matrix(test_new, y_p_new)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=np.unique(tags),title='Confusion matrix, without normalization')





plt.figure()
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,where='post')
plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
plt.show()


           

for i in range(0,len(classes)):
    precision[i], recall[i], ther[i] = precision_recall_curve(y_test[:, i],
                                                        y_p[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_p[:, i])
#for i in range(0,len(classes)):
#    print("Precision of",classes[i],"=",precision[i])
#    print("Recall of",classes[i],"=",recall[i])
#    print("Threshold of",classes[i],"=",ther[i])
    
# A "micro-average": quantifying score on all classes jointly
#precision["micro"], recall["micro"], _ = precision_recall_curve(testLabels.ravel(),y_p.ravel())
#average_precision["micro"] = average_precision_score(testLabels, y_p,average="micro")
#print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range (0,len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_p[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_p.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(classes)):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= len(classes)
    


fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
lw = 2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]),color='deeppink' , linestyle=':', linewidth=4)
plt.plot(fpr["macro"], tpr["macro"],label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]),color='navy', linestyle=':', linewidth=4)
colors = cycle(['aqua', 'fuchsia', 'cornflowerblue','red','green','yellow','pink','teal','black'])
for i, color in zip(range(len(classes)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'''.format(classes[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

# save the model and use it
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# later...

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
test_sequences = tok.texts_to_sequences(x_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

accr = model.evaluate(test_sequences_matrix,testLabels,verbose=1)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


        

    


