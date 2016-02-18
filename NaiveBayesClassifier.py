import string
import re
import math
import time

startTrain = time.time()
#Open files
trainSet = open('training.txt')
trainText = trainSet.read()
testSet = open('testing.txt')
testText = testSet.read()

#Define positive and negative vocab
positiveVocab = {}
negativeVocab = {}
classList = []

def getReviews(x):
	reviews = re.split(r'\n+', x)
	return reviews

def standardize(x):
	#Remove punctuations
	x = x.translate(string.maketrans("",""), string.punctuation)

	#Make all lower case
	x = x.lower()

	#Split text into words
	return re.split("\W+", x)


def wordCounter(w, train):
	#For training:
	if train == 1:
		if w[0] == '1':
			#Remove the number and tab
			w[2:]
			for word in w:
				positiveVocab[word] = positiveVocab.get(word, 0.0) + 1.0

		elif w[0] == '0':
			#Remove the number and tab
			w[2:]
			for word in w:
				negativeVocab[word] = negativeVocab.get(word, 0.0) + 1.0
	else:
		trueClass = w[0]
		w[2:]
		vocab = {}
		for word in w:
			vocab[word] = vocab.get(word,0.0) + 1.0
		return vocab, trueClass

def removeCommonWords():
	commonWords = ['the','be','to','of','and','a','in','that','have','I','it','for','not','on','with','he','as','you','do','at','so','up','out','if','about','who','get','which','go','me','when','like','time','no','him','know','take','people','into','year','your','some','could','them','see','other','than','then','now','look','come','its','over','think','also','back','after','use','two','how','our','first','way','even','because','any','these','give','day','us']
	for i in range(0,len(commonWords)):
		if commonWords[i] in positiveVocab:
			positiveVocab.pop(commonWords[i], 0)
		if commonWords[i] in negativeVocab:
			negativeVocab.pop(commonWords[i], 0)

def removeRarelyUsedWords():
	for i in positiveVocab.keys():
		if positiveVocab[i] < 8:
			del positiveVocab[i]

	for j in negativeVocab.keys():
		if negativeVocab[i] < 8:
			del negativeVocab[i]
	

def classify(r):

	log_prob_pos = 0.0
	log_prob_neg = 0.0

	wordsInPos = sum(positiveVocab.values())
	wordsInNeg = sum(negativeVocab.values())

	for word, count in r.items():
		if not word in positiveVocab or not word in negativeVocab or len(word) <= 5:
			continue

		#p_word_pos = positiveVocab[word]/(wordsInPos+wordsInNeg)
		#p_word_neg = negativeVocab[word]/(wordsInPos+wordsInNeg)

		p_w_given_pos = positiveVocab.get(word, 0.0) 
		p_w_given_neg = negativeVocab.get(word, 0.0) 


		#So that words that never appears in the catagory is omitted
		if p_w_given_pos > 0.0:
			log_prob_pos += (p_w_given_pos)
		if p_w_given_neg > 0.0:
			log_prob_neg += (p_w_given_neg)

	scorePos = (log_prob_pos)
	scoreNeg = (log_prob_neg)

	#print scorePos
	#print scoreNeg
	#print '\n'


	if scorePos > scoreNeg:
		return 1

	elif scoreNeg > scorePos:
		return 0 

	else:
		return 0

def main():

	trainCorrect = 0.0
	totalTrainTested = 0.0

	testCorrect = 0.0
	totalTestTested = 0.0

	#Get the reviews as seperate entries in the list trainingReviews
	trainingReviews = getReviews(trainText)

	#Training loop for establishing a vocab for negative and positive words
	for i in range(0, len(trainingReviews)):
		wordCounter(standardize(trainingReviews[i]), 1)
		i += 1

	#Remove common words and stop words
	removeCommonWords()

	#Remove words not commonly used
	#removeRarelyUsedWords()

	#Get the reviews as seperate entries in the list testingReviews
	testingReviews = getReviews(testText)

	endTrain = time.time()

	startTest = time.time()

	#Classify loop for the training set
	for i in range(0, len(trainingReviews)):
		standardizeTraining = standardize(trainingReviews[i])
		trainingCount, trueClass = wordCounter(standardizeTraining, 0)
		trueClass = str(trueClass)
		classification = classify(trainingCount)
		classification = str(classification)
		if classification == (trueClass):
			trainCorrect += 1.0
		totalTrainTested += 1.0
		i += 1

	#Classify loop for the testing set
	for i in range(0, len(testingReviews)):
		standardizeTesting = standardize(testingReviews[i])
		testingCount, trueClass = wordCounter(standardizeTesting, 0)
		trueClass = str(trueClass)
		classification = classify(testingCount)
		classification = str(classification)
		if classification == trueClass:
			testCorrect += 1.0
		totalTestTested += 1.0
		i += 1
		print classification
	endTest = time.time()

	#Print functions to get right output
	print math.ceil((endTrain-startTrain)), ' (training)'
	print math.ceil((endTest-startTest)), ' (labeling)'
	print (trainCorrect/totalTrainTested), ' (training)'
	print (testCorrect/totalTestTested), ' (testing)'



main()

