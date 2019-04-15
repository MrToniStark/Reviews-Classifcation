#Reading Input
import pandas as pd
df1 = pd.read_csv('Round1_Problem1-of-2_Dataset_amazon_cells_labelled.txt', delimiter = '\t',header = None)
df2 = pd.read_csv('Round1_Problem1-of-2_Dataset_imdb_labelled.txt', delimiter = '\t',header = None)
df1.columns = ['Text', 'Sentiments']
df2.columns = ['Text', 'Sentiments']
df = pd.concat([df1, df2], ignore_index=True)
# Cleaning the texts
import re
import nltk
import emoji
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1748):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', df['Text'][i])
    text = re.sub('[^a-zA-Z]', ' ', df['Text'][i])
    text = emoji.demojize(df['Text'][i])
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpus.append(text)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

#Test 01: Using NB Classifier
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
preds = classifier.predict(X_test)

# Analysing the result
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, preds)
print(cm)
print('\n')
print(classification_report(y_test, preds))
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0])
print('Accuracy = '+str(accuracy))

#Accuracy = 0.786

#Test 02: Using Neural Network
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 750, init = 'uniform', activation = 'relu', input_dim = 1500))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 750, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
preds = classifier.predict(X_test)
preds = (preds > 0.5)

# Analysing the result
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, preds)
print(cm)
print('\n')
print(classification_report(y_test, preds))
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0])
print('Accuracy = '+str(accuracy))

#Accuracy = 0.786