from math import prod
import os.path
import nltk, re
from nltk import FreqDist, NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.corpus.reader import CategorizedPlaintextCorpusReader
import random
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer

def remove_emoji(text):
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002700-\U000027BF"  # Dingbats
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text) 

def remove_punc(text):
	punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
	for ele in text:
		if ele in punc:
			text = text.replace(ele, "")
	return text

bumble=CategorizedPlaintextCorpusReader('nltk_data/corpora/bumble',  \
                                 '.*', \
                                 cat_pattern=r'(.*)[/]')

docs = []

lemmatizer = WordNetLemmatizer()
punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

def accronym(word):
	abbreviations=[
		('app','application'),
		('bot','robot'),
		('bots','robots'),
		('vs','versus'),
		('cuz', 'because'),
		('ads', 'advertisements'),
		('ad', 'advertisement'),
		('bio','biography')
	]
	for acc in abbreviations:
		x,y=acc
		if x==word:
			return y
	return word

for category in bumble.categories():
	for fileid in bumble.fileids(category):
		if len(bumble.words(fileid)) > 9:
			docs.append((bumble.words(fileid), category))

#create documents as a list of pairs
documents=[]
for doc in docs:
	x,y=doc
	documents.append((list(x),y))

#Remove punctuations, emojies, lemmatization, replace accronyms
for i in range(len(documents)):
	x,y=documents[i]
	for j in range(len(x)):
		word=x[j]
		if word in punc:
			word=''
		word=remove_emoji(word)
		word=word.lower()
		word=lemmatizer.lemmatize(word)
		word=accronym(word)
		x[j]=word
	documents[i]=(x,y)

#Remove Stopwords
stopwords_english = stopwords.words('english')
cleaned_docs=[]

for i in range(len(documents)):
	x,y=documents[i]
	z=[]
	for word in x:
		if word not in stopwords_english and word!='':
			z.append(word)
	documents[i]=(z,y)

#create a list of all words
all_words_clean=[]
for document in documents:
	docx,docy=document
	for word in docx:
		all_words_clean.append(word)

#Most common 2000 words
all_words_frequency = FreqDist(all_words_clean)
most_common_words = all_words_frequency.most_common(2000)
word_features = [item[0] for item in most_common_words]

#create feature set
def document_features(document):
	document_words = set(document)
	features = {}
	for word in word_features:
		features['contains(%s)' % word] = (word in document_words)
	return features

feature_set = [(document_features(doc), category) for (doc, category) in documents]

#randomly select the train and test sets 60% train, 40% test
random.shuffle(feature_set)
train_set = feature_set[400:]
test_set = feature_set[:400]

################ Naive Bayes Classifier ################
NBclassifier = NaiveBayesClassifier.train(train_set)

#build the confusion matrix
app_func=[]
app_func=[0 for i in range(3)]
cust_serv=[]
cust_serv=[0 for i in range(3)]
prod_sat=[]
prod_sat=[0 for i in range(3)]

y_true=[]
y_pred=[]

for document in test_set:
	x,y=document
	if y=="app_func":
		i=0
		y_true.append(0)
	elif y=="cust_serv":
		i=1
		y_true.append(1)
	elif y=="prod_sat":
		i=2
		y_true.append(2)
	this_class=NBclassifier.classify(x)
	if this_class=="app_func":
		app_func[i]=app_func[i]+1
		y_pred.append(0)
	elif this_class=="cust_serv":
		cust_serv[i]=cust_serv[i]+1
		y_pred.append(1)
	elif this_class=="prod_sat":
		prod_sat[i]=prod_sat[i]+1
		y_pred.append(2)

target_names=['app_func', 'cust_serv', 'prod_sat']
print()
print('Imbalanced NBC')
print()
print(classification_report(y_true,y_pred,target_names=target_names))


################ Support Vector Classifier ################
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC

#SVC model training and accuracy
model_svc = SklearnClassifier(SVC(kernel = 'linear'))
model_svc.train(train_set)

y_true=[]
y_pred=[]
for document in test_set:
	x,y=document
	if y=="app_func":
		i=0
		y_true.append(0)
	elif y=="cust_serv":
		i=1
		y_true.append(1)
	elif y=="prod_sat":
		i=2
		y_true.append(2)
	this_class=model_svc.classify(x)
	if this_class=="app_func":
		app_func[i]=app_func[i]+1
		y_pred.append(0)
	elif this_class=="cust_serv":
		cust_serv[i]=cust_serv[i]+1
		y_pred.append(1)
	elif this_class=="prod_sat":
		prod_sat[i]=prod_sat[i]+1
		y_pred.append(2)

print()
print('Imbalanced SVC')
print()
print(classification_report(y_true,y_pred,target_names=target_names))

################ Balancing Data ################
balanced_feature_set=feature_set

app_func=0
cust_serv=0
prod_sat=0
for member in balanced_feature_set:
	x,y=member
	if y=="app_func":
		app_func+=1
	elif y=="cust_serv":
		cust_serv+=1
	elif y=="prod_sat":
		prod_sat+=1

while app_func < prod_sat and cust_serv<prod_sat:
	for member in balanced_feature_set:
		x,y=member
		if y=="app_func":
			if app_func < prod_sat:
				balanced_feature_set.append((x,y))
				app_func+=1
		elif y=="cust_serv" and cust_serv < prod_sat:
			#if cust_serv < prod_sat:
			balanced_feature_set.append((x,y))
			cust_serv+=1

#Shuffle and randomly select 400 from the balanced feature set to test, rest train
random.shuffle(balanced_feature_set)
balanced_train_set = balanced_feature_set[400:]
balanced_test_set = balanced_feature_set[:400]

################ Naive Bayes Classifier, Balanced ################
NBclassifier = NaiveBayesClassifier.train(balanced_train_set)

app_func=[]
app_func=[0 for i in range(3)]
cust_serv=[]
cust_serv=[0 for i in range(3)]
prod_sat=[]
prod_sat=[0 for i in range(3)]

y_true=[]
y_pred=[]

for document in balanced_test_set:
	x,y=document
	if y=="app_func":
		i=0
		y_true.append(0)
	elif y=="cust_serv":
		i=1
		y_true.append(1)
	elif y=="prod_sat":
		i=2
		y_true.append(2)
	this_class=NBclassifier.classify(x)
	if this_class=="app_func":
		app_func[i]=app_func[i]+1
		y_pred.append(0)
	elif this_class=="cust_serv":
		cust_serv[i]=cust_serv[i]+1
		y_pred.append(1)
	elif this_class=="prod_sat":
		prod_sat[i]=prod_sat[i]+1
		y_pred.append(2)

print()
print('Balanced NBC')
print()
print(classification_report(y_true,y_pred,target_names=target_names))

################ Support Vector Classifier, Balanced ################
model_svcBAL = SklearnClassifier(SVC(kernel = 'linear'))
model_svcBAL.train(balanced_train_set)

y_true=[]
y_pred=[]
for document in balanced_test_set:
	x,y=document
	if y=="app_func":
		i=0
		y_true.append(0)
	elif y=="cust_serv":
		i=1
		y_true.append(1)
	elif y=="prod_sat":
		i=2
		y_true.append(2)
	this_class=model_svc.classify(x)
	if this_class=="app_func":
		app_func[i]=app_func[i]+1
		y_pred.append(0)
	elif this_class=="cust_serv":
		cust_serv[i]=cust_serv[i]+1
		y_pred.append(1)
	elif this_class=="prod_sat":
		prod_sat[i]=prod_sat[i]+1
		y_pred.append(2)

print()
print('Balanced SVC')
print()
print(classification_report(y_true,y_pred,target_names=target_names))