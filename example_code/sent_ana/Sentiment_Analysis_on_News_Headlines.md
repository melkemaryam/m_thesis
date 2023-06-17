# Sentiment Analysis on News Headlines: Classic Supervised Learning vs Deep Learning Approach


```python
#install needed packages
#!pip install snorkel
#!pip install textblob
#import libraries and modules
from google.colab import files
import io
import pandas as pd
#Snorkel
from snorkel.labeling import LabelingFunction
import re
from snorkel.preprocess import preprocessor
from textblob import TextBlob
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
from snorkel.labeling import LFAnalysis
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.labeling import labeling_function
#NLP packages
import spacy
from nltk.corpus import stopwords
import string
import nltk
import nltk.tokenize
punc = string.punctuation
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
#Supervised learning
from tqdm.notebook import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
##Deep learning libraries and APIs
import numpy as np
import tensorflow as tfw
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.stem import WordNetLemmatizer  # lemmatization
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package wordnet to /root/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!



```python
#uplaod the data from your local directory
path = 'drive/MyDrive/data.csv'
# store the dataset as a Pandas Dataframe
df = pd.read_csv(path)
#conduct some data cleaning
#df = df.drop(['publish_date', 'Unnamed: 2'], axis=1)
df = df.rename(columns = {'headline_text': 'text'})
df['text'] = df['text'].astype(str)

df = df.iloc[:100000,:]
#check the data info
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100000 entries, 0 to 99999
    Data columns (total 2 columns):
     #   Column        Non-Null Count   Dtype 
    ---  ------        --------------   ----- 
     0   publish_date  100000 non-null  int64 
     1   text          100000 non-null  object
    dtypes: int64(1), object(1)
    memory usage: 1.5+ MB


## Snorkel: Create labels


```python
#define constants to represent the class labels :positive, negative, and abstain
POSITIVE = 1
NEGATIVE = 0
ABSTAIN = -1
#define function which looks into the input words to represent a proper label
def keyword_lookup(x, keywords, label):
    if any(word in x.text.lower() for word in keywords):
        return label
    return ABSTAIN
#define function which assigns a correct label
def make_keyword_lf(keywords, label=POSITIVE):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label))
#resource: https://www.snorkel.org/use-cases/01-spam-tutorial#3-writing-more-labeling-functions
#these two lists can be further extended
"""positive news might contain the following words' """
keyword_positive = make_keyword_lf(keywords=['boosts', 'great', 'develops', 'promising', 'ambitious', 'delighted', 'record', 'win', 'breakthrough', 'recover', 'achievement', 'peace', 'party', 'hope', 'flourish', 'respect', 'partnership', 'champion', 'positive', 'happy', 'bright', 'confident', 'encouraged', 'perfect', 'complete', 'assured' ])
"""negative news might contain the following words"""
keyword_negative = make_keyword_lf(keywords=['war','solidiers', 'turmoil', 'injur','trouble', 'aggressive', 'killed', 'coup', 'evasion', 'strike', 'troops', 'dismisses', 'attacks', 'defeat', 'damage', 'dishonest', 'dead', 'fear', 'foul', 'fails', 'hostile', 'cuts', 'accusations', 'victims',  'death', 'unrest', 'fraud', 'dispute', 'destruction', 'battle', 'unhappy', 'bad', 'alarming', 'angry', 'anxious', 'dirty', 'pain', 'poison', 'unfair', 'unhealthy'
                                              ], label=NEGATIVE)
```


```python
#set up a preprocessor function to determine polarity & subjectivity using textlob pretrained classifier
@preprocessor(memoize=True)
def textblob_sentiment(x):
    scores = TextBlob(x.text)
    x.polarity = scores.sentiment.polarity
    x.subjectivity = scores.sentiment.subjectivity
    return x
#find polarity
@labeling_function(pre=[textblob_sentiment])
def textblob_polarity(x):
    return POSITIVE if x.polarity > 0.6 else ABSTAIN
#find subjectivity
@labeling_function(pre=[textblob_sentiment])
def textblob_subjectivity(x):
    return POSITIVE if x.subjectivity >= 0.5 else ABSTAIN
```


```python
#combine all the labeling functions
lfs = [keyword_positive, keyword_negative, textblob_polarity, textblob_subjectivity ]
#apply the lfs on the dataframe
applier = PandasLFApplier(lfs=lfs)
L_snorkel = applier.apply(df=df)
#apply the label model
label_model = LabelModel(cardinality=2, verbose=True)
#fit on the data
label_model.fit(L_snorkel)
#predict and create the labels
df["label"] = label_model.predict(L=L_snorkel)
```

    100%|██████████| 100000/100000 [01:55<00:00, 863.95it/s]
    100%|██████████| 100/100 [00:00<00:00, 654.11epoch/s]



```python
#Filtering out unlabeled data points
df= df.loc[df.label.isin([0,1]), :]
#find the label counts
df['label'].value_counts()
```




    1    19259
    0    10809
    Name: label, dtype: int64



## Supervised Learning: Logistic Regression


```python
df
```





  <div id="df-ab2dbd8b-a30e-4596-b85b-28b20259ea5f">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>publish_date</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>20030219</td>
      <td>act fire witnesses must be aware of defamation</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20030219</td>
      <td>air nz staff in aust strike for pay rise</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20030219</td>
      <td>air nz strike to affect australian travellers</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20030219</td>
      <td>ambitious olsson wins triple jump</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20030219</td>
      <td>antic delighted with record breaking barca</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>99984</th>
      <td>20040630</td>
      <td>big phil and portugal close to renewing marriage</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99991</th>
      <td>20040630</td>
      <td>bureau plays down fears over coastal waters</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99994</th>
      <td>20040630</td>
      <td>call for quick plans to save softwood industry</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>20040630</td>
      <td>call in powers bill likely to fail</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99999</th>
      <td>20040630</td>
      <td>capriati confident of serena hat trick</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>30068 rows × 3 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ab2dbd8b-a30e-4596-b85b-28b20259ea5f')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ab2dbd8b-a30e-4596-b85b-28b20259ea5f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ab2dbd8b-a30e-4596-b85b-28b20259ea5f');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
#make a copy of the dataframe
data = df.copy()
#data['text']

# select raw text
raw_text = data.text.values.tolist()

'''
rt = []
for i in range(len(raw_text)):
  new = raw_text[i].split()
  rt.append(new)

'''
r = [[raw_text[i]] for i in range(6)]
r
```




    [['act fire witnesses must be aware of defamation'],
     ['air nz staff in aust strike for pay rise'],
     ['air nz strike to affect australian travellers'],
     ['ambitious olsson wins triple jump'],
     ['antic delighted with record breaking barca'],
     ['australia is locked into war timetable opp']]




```python
#define a function which handles the text preprocessing
def preparation_text_data(data):
    """
    This pipeline prepares the text data, conducting the following steps:
    1) Tokenization
    2) Lemmatization
    4) Removal of stopwords
    5) Removal of punctuation
    """
    # initialize spacy object
    nlp = spacy.load('en_core_web_sm')
    # select raw text
    raw_text = data.text.values.tolist()
    # tokenize
    tokenized_text = [[nlp(i.lower().strip())] for i in tqdm(raw_text)]
    #define the punctuations and stop words
    punc = string.punctuation
    stop_words = set(stopwords.words('english'))
    #lemmatize, remove stopwords and punctuationd
    corpus = []
    for doc in tqdm(tokenized_text):
        corpus.append([word.lemma_ for word in doc[0] if (word.lemma_ not in stop_words and word.lemma_ not in punc)])
    # add prepared data to df
    data["text"] = corpus
    return data
#apply the data preprocessing function
data =  preparation_text_data(data)
```


      0%|          | 0/30068 [00:00<?, ?it/s]



      0%|          | 0/30068 [00:00<?, ?it/s]



```python
data['text']
```




    1           [act, fire, witness, must, aware, defamation]
    3               [air, nz, staff, aust, strike, pay, rise]
    4        [air, nz, strike, affect, australian, traveller]
    5                  [ambitious, olsson, win, triple, jump]
    6                  [antic, delight, record, break, barca]
                                   ...                       
    99984       [big, phil, portugal, close, renew, marriage]
    99991                [bureau, play, fear, coastal, water]
    99994       [call, quick, plan, save, softwood, industry]
    99997                   [call, power, bill, likely, fail]
    99999           [capriati, confident, serena, hat, trick]
    Name: text, Length: 30068, dtype: object




```python
def text_representation(data):
  tfidf_vect = TfidfVectorizer()
  data['text'] = data['text'].apply(lambda text: " ".join(set(text)))
  X_tfidf = tfidf_vect.fit_transform(data['text'])
  print(X_tfidf.shape)
  print(tfidf_vect.get_feature_names_out())
  X_tfidf = pd.DataFrame(X_tfidf.toarray())
  return X_tfidf
#apply the TFIDV function
X_tfidf = text_representation(data)
```

    (30068, 13070)
    ['000' '03' '10' ... 'zulu' 'zurich' 'zvonareva']



```python
X= X_tfidf
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#fit Log Regression Model
clf= LogisticRegression()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.92      0.93      0.92      3566
               1       0.96      0.95      0.96      6357
    
        accuracy                           0.94      9923
       macro avg       0.94      0.94      0.94      9923
    weighted avg       0.94      0.94      0.94      9923
    



```python
new_data = ["The US imposes sanctions on Russia because of the Ukranian war"]
tf = TfidfVectorizer()
tfdf = tf.fit_transform(data['text'])
vect = pd.DataFrame(tf.transform(new_data).toarray())
new_data = pd.DataFrame(vect)
logistic_prediction = clf.predict(new_data)
print(logistic_prediction)
```

    [0]


## Deep Learning Approach


```python
##store headlines and labels in respective lists
text = list(data['text'])
labels = list(data['label'])
##sentences
training_text = text[0:25000]
testing_text = text[25000:]
##labels
training_labels = labels[0:25000]
testing_labels = labels[25000:]
```


```python
#preprocess
tokenizer = Tokenizer(num_words=10000, oov_token= "<OOV>")
tokenizer.fit_on_texts(training_text)
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(training_text)
training_padded = pad_sequences(training_sequences, maxlen=120, padding='post', truncating='post')
testing_sequences = tokenizer.texts_to_sequences(testing_text)
testing_padded = pad_sequences(testing_sequences, maxlen=120, padding='post', truncating='post')
# convert lists into numpy arrays to make it work with TensorFlow
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)
```


```python
model = tfw.keras.Sequential([
    tfw.keras.layers.Embedding(10000, 16, input_length=120),
    tfw.keras.layers.GlobalAveragePooling1D(),
    tfw.keras.layers.Dense(24, activation='relu'),
    tfw.keras.layers.Dense(1, activation='sigmoid')
])
##compile the model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding (Embedding)       (None, 120, 16)           160000    
                                                                     
     global_average_pooling1d (G  (None, 16)               0         
     lobalAveragePooling1D)                                          
                                                                     
     dense (Dense)               (None, 24)                408       
                                                                     
     dense_1 (Dense)             (None, 1)                 25        
                                                                     
    =================================================================
    Total params: 160,433
    Trainable params: 160,433
    Non-trainable params: 0
    _________________________________________________________________



```python
num_epochs = 10
history = model.fit(training_padded,
                    training_labels,
                    epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels),
                    verbose=2)
```

    Epoch 1/10
    782/782 - 19s - loss: 0.6510 - accuracy: 0.6398 - val_loss: 0.6307 - val_accuracy: 0.6393 - 19s/epoch - 24ms/step
    Epoch 2/10
    782/782 - 4s - loss: 0.4667 - accuracy: 0.7802 - val_loss: 0.3114 - val_accuracy: 0.8869 - 4s/epoch - 5ms/step
    Epoch 3/10
    782/782 - 3s - loss: 0.2269 - accuracy: 0.9269 - val_loss: 0.2100 - val_accuracy: 0.9256 - 3s/epoch - 4ms/step
    Epoch 4/10
    782/782 - 4s - loss: 0.1619 - accuracy: 0.9475 - val_loss: 0.1758 - val_accuracy: 0.9355 - 4s/epoch - 5ms/step
    Epoch 5/10
    782/782 - 4s - loss: 0.1303 - accuracy: 0.9573 - val_loss: 0.1578 - val_accuracy: 0.9442 - 4s/epoch - 5ms/step
    Epoch 6/10
    782/782 - 3s - loss: 0.1123 - accuracy: 0.9631 - val_loss: 0.1459 - val_accuracy: 0.9515 - 3s/epoch - 4ms/step
    Epoch 7/10
    782/782 - 3s - loss: 0.0976 - accuracy: 0.9678 - val_loss: 0.1419 - val_accuracy: 0.9530 - 3s/epoch - 4ms/step
    Epoch 8/10
    782/782 - 5s - loss: 0.0868 - accuracy: 0.9708 - val_loss: 0.1399 - val_accuracy: 0.9526 - 5s/epoch - 7ms/step
    Epoch 9/10
    782/782 - 4s - loss: 0.0791 - accuracy: 0.9742 - val_loss: 0.1318 - val_accuracy: 0.9566 - 4s/epoch - 6ms/step
    Epoch 10/10
    782/782 - 3s - loss: 0.0721 - accuracy: 0.9770 - val_loss: 0.1348 - val_accuracy: 0.9546 - 3s/epoch - 4ms/step



```python
new_headline = ["The US imposes sanctions on Russia because of the Ukranian war"]
##prepare the sequences of the sentences in question
sequences = tokenizer.texts_to_sequences(new_headline)
padded_seqs = pad_sequences(sequences, maxlen=120, padding='post', truncating='post')
print(model.predict(padded_seqs))
```

    1/1 [==============================] - 0s 169ms/step
    [[0.00083828]]



```python

```
