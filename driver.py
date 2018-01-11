import csv
import sys
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer as cv
from sklearn.feature_extraction.text import TfidfVectorizer as tv
from sklearn.linear_model import SGDClassifier as sgd
from sklearn import model_selection

train_path = "aclImdb/train/" # use terminal to ls files under this directory
test_path = "imdb_te.csv" # test data for grade evaluation


def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
 	#'''Implement this module to extract
	#and combine text files under train_path directory into 
    #imdb_tr.csv. Each text file in train_path should be stored 
    #as a row in imdb_tr.csv. And imdb_tr.csv should have two 
    #columns, "text" and label'''

    # Prepare file to write to.
	output = open(name, 'w+')
	csv_writer = csv.writer(output, delimiter='|')

	csv_writer.writerow(['id', 'text', 'label'])

	# Load stopwords to remove.
	stop_text = open('stopwords.en.txt')
	stop_words = set()
	for word in stop_text:
		stop_words.add(word.replace('\n', ''))


	# Get positive and negative files.
	pos_files = os.listdir(inpath+'/pos')
	neg_files = os.listdir(inpath+'/neg')

	# Unpack positive files.
	for n in range(0, len(pos_files), 2):
	#for n in range(0, 10, 2):
		pos_review = open(inpath+'/pos/'+pos_files[n])
		for rev in pos_review:
			# Remove undesired words.
			revised_rev = []
			for word in rev.split(' '):
				# remove undesired HTML.
				cleansed_w = word.replace('<br />', '').replace('/>', '').replace('<br', '')
				# Remove punctuation.
				cleansed_w = cleansed_w.replace('.', '').replace(',', '').replace(';', '').replace(':', '').replace('?', '').replace('!', '')
				# Remove quotes, etc.
				cleansed_w = cleansed_w.replace('\'', '').replace('""', '').replace('"', '').replace('|', '')
				word = cleansed_w.lower()
				if word not in stop_words:
					revised_rev.append(word+' ')
			csv_writer.writerow([n, ''.join(revised_rev), 1])
			#csv_writer.writerow([n, rev, 1])
		neg_review = open(inpath+'/neg/'+neg_files[n])
		for rev in neg_review:
			# Remove undesired words.
			revised_rev = []
			for word in rev.split(' '):
				# remove undesired HTML.
				cleansed_w = word.replace('<br />', '').replace('/>', '').replace('<br', '')
				# Remove punctuation.
				cleansed_w = cleansed_w.replace('.', '').replace(',', '').replace(';', '').replace(':', '').replace('?', '').replace('!', '')
				# Remove quotes, etc.
				cleansed_w = cleansed_w.replace('\'', '').replace('""', '').replace('"', '').replace('|', '')
				word = cleansed_w.lower()
				if word not in stop_words:
					revised_rev.append(word+' ')
			n += 1
			csv_writer.writerow([n, ''.join(revised_rev), 0])
			#csv_writer.writerow([n, rev, 0])






	#pass
  
if __name__ == "__main__":
	# Preprocess the data.
	imdb_data_preprocess(train_path)

 	#'''train a SGD classifier using unigram representation,
    #predict sentiments on imdb_te.csv, and write output to
    #unigram.output.txt'''
  	train_data = pd.read_csv('imdb_tr.csv', sep='|', error_bad_lines=False, quoting=3).as_matrix()
  	test_data = pd.read_csv(test_path, sep=',', error_bad_lines=False, quoting=2).as_matrix()

  	# Load stopwords to remove.
	stop_text = open('stopwords.en.txt')
	stop_words = set()
	for word in stop_text:
		stop_words.add(word.replace('\n', ''))

  	test_text = list()
  	n = 0
  	for rev in test_data:
  		revised_rev = []
  		for word in rev[1].split(' '):
			# remove undesired HTML.
			cleansed_w = word.replace('<br />', '').replace('/>', '').replace('<br', '')
			# Remove punctuation.
			cleansed_w = cleansed_w.replace('.', '').replace(',', '').replace(';', '').replace(':', '').replace('?', '').replace('!', '')
			# Remove quotes, etc.
			cleansed_w = cleansed_w.replace('\'', '').replace('""', '').replace('"', '').replace('|', '')
			word = cleansed_w.lower()
			if word not in stop_words:
				revised_rev.append(word+' ')
		test_text.append(''.join(revised_rev))

  	#print len(test_data)
  	#print len(test_text)
  	train_text = list()
  	labels = list()
  	n = 0
  	for data in train_data:
  		train_text.append(data[1])
  		labels.append(data[2])

  	vectorizer = cv(encoding='utf-8', strip_accents='unicode', ngram_range=(1,1), decode_error='replace')
  	vector_data = vectorizer.fit_transform(train_text)
  	#vector_test_data = vectorizer.transform(test_text)



  	model_selector = model_selection
	X_train, X_test, y_train, y_test = model_selector.train_test_split(vector_data, labels, stratify=labels, test_size=0.2)

  	#vector_test = vectorizer.fit_transform(test_text)

  	classifier = sgd(loss='hinge', penalty='l1')
  	#classifier.fit(vector_data, labels)
  	classifier.fit(X_train, y_train)


  	#train_scores = classifier.score(vector_data, labels)
  	train_scores = classifier.score(X_train, y_train)
  	print 'Unigram Results'
	print 'Train Scores'
	print train_scores
	print("Accuracy: %0.2f (+/- %0.2f)" % (train_scores.mean(), train_scores.std() * 2))

	#test_scores = classifier.score(vector_test, test_labels)
	test_scores = classifier.score(X_test, y_test)
	print 'Test Scores'
	print test_scores
	print("Accuracy: %0.2f (+/- %0.2f)" % (test_scores.mean(), test_scores.std() * 2))
	print ''

	#f = open('unigram.output.txt', 'w+')
	#print vector_test_data.shape
	#for test_set in vector_test_data:
	#	prediction = classifier.predict(test_set)
	#	f.write(str(prediction[0])+'\n')


    #'''train a SGD classifier using bigram representation,
    #predict sentiments on imdb_te.csv, and write output to
    #unigram.output.txt'''


	vectorizer = cv(encoding='utf-8', strip_accents='unicode', ngram_range=(1,2), decode_error='replace')
  	vector_data = vectorizer.fit_transform(train_text)
  	#vector_test_data = vectorizer.transform(test_text)

  	model_selector = model_selection
	X_train, X_test, y_train, y_test = model_selector.train_test_split(vector_data, labels, stratify=labels, test_size=0.2)

  	#vector_test = vectorizer.fit_transform(test_text)

  	classifier = sgd(loss='hinge', penalty='l1')
  	#classifier.fit(vector_data, labels)
  	classifier.fit(X_train, y_train)


  	#train_scores = classifier.score(vector_data, labels)
  	train_scores = classifier.score(X_train, y_train)
  	print 'Bigram Results'
	print 'Train Scores'
	print train_scores
	print("Accuracy: %0.2f (+/- %0.2f)" % (train_scores.mean(), train_scores.std() * 2))
	#test_scores = classifier.score(vector_test, test_labels)
	test_scores = classifier.score(X_test, y_test)
	print 'Test Scores'
	print test_scores
	print("Accuracy: %0.2f (+/- %0.2f)" % (test_scores.mean(), test_scores.std() * 2))
	print ''

	#f = open('bigram.output.txt', 'w+')
	#for test_set in vector_test_data:
	#	prediction = classifier.predict(test_set)
	#	f.write(str(prediction[0])+'\n')
     
     #'''train a SGD classifier using unigram representation
     #with tf-idf, predict sentiments on imdb_te.csv, and write 
     #output to unigram.output.txt'''

	vectorizer = tv(encoding='utf-8', strip_accents='unicode', ngram_range=(1,1), decode_error='replace')
  	vector_data = vectorizer.fit_transform(train_text)
  	#vector_test_data = vectorizer.transform(test_text)

  	model_selector = model_selection
	X_train, X_test, y_train, y_test = model_selector.train_test_split(vector_data, labels, stratify=labels, test_size=0.2)

  	#vector_test = vectorizer.fit_transform(test_text)

  	classifier = sgd(loss='hinge', penalty='l1')
  	#classifier.fit(vector_data, labels)
  	classifier.fit(X_train, y_train)


  	#train_scores = classifier.score(vector_data, labels)
  	train_scores = classifier.score(X_train, y_train)
  	print 'TF-IDF Unigram Results'
	print 'Train Scores'
	print train_scores
	print("Accuracy: %0.2f (+/- %0.2f)" % (train_scores.mean(), train_scores.std() * 2))

	#test_scores = classifier.score(vector_test, test_labels)
	test_scores = classifier.score(X_test, y_test)
	print 'Test Scores'
	print test_scores
	print("Accuracy: %0.2f (+/- %0.2f)" % (test_scores.mean(), test_scores.std() * 2))
	print ''


	#f = open('unigramtfidf.output.txt', 'w+')
	#for test_set in vector_test_data:
	#	prediction = classifier.predict(test_set)
	#	f.write(str(prediction[0])+'\n')
  	
     #'''train a SGD classifier using bigram representation
     #with tf-idf, predict sentiments on imdb_te.csv, and write 
     #output to unigram.output.txt'''
     #pass


	vectorizer = tv(encoding='utf-8', strip_accents='unicode', ngram_range=(1,2), decode_error='replace')
  	vector_data = vectorizer.fit_transform(train_text)
  	#vector_test_data = vectorizer.transform(test_text)

  	model_selector = model_selection
	X_train, X_test, y_train, y_test = model_selector.train_test_split(vector_data, labels, stratify=labels, test_size=0.2)

  	#vector_test = vectorizer.fit_transform(test_text)

  	classifier = sgd(loss='hinge', penalty='l1')
  	#classifier.fit(vector_data, labels)
  	classifier.fit(X_train, y_train)


  	#train_scores = classifier.score(vector_data, labels)
  	train_scores = classifier.score(X_train, y_train)
  	print 'TF-IDF Bigram Results'
	print 'Train Scores'
	print train_scores
	print("Accuracy: %0.2f (+/- %0.2f)" % (train_scores.mean(), train_scores.std() * 2))

	#test_scores = classifier.score(vector_test, test_labels)
	test_scores = classifier.score(X_test, y_test)
	print 'Test Scores'
	print test_scores
	print("Accuracy: %0.2f (+/- %0.2f)" % (test_scores.mean(), test_scores.std() * 2))


	#f = open('bigramtfidf.output.txt', 'w+')
	#for test_set in vector_test_data:
	#	prediction = classifier.predict(test_set)
	#	f.write(str(prediction[0])+'\n')