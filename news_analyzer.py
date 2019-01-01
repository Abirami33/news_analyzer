import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#vader sentiment analyser package

''' PARAGRAPH TAKEN :
['Rising sea level spells disaster for delta regions',
'Sea levels are rising at different rates along the Indian coast,
with projections till the end of the current century varying between 3.5 and 34.6 inches
and which may pose a danger to major deltas, including the Cauvery delta, in east India 
and vast stretches of the western coastline, including Mumbai.', 
'Sharing studies by the Hyderabadbased Indian National Centre for Ocean Information Services,
the government on Friday told the Lok Sabha that sea level rise between 1990 and 2100 can make west coast
stretches like Khambat and Kutch in Gujarat, Mumbai and parts of Konkan and south Kerala most vulnerable.', 
'The threats posed by sea level rise due to global warming have direct implications for India’s food security
as hundreds of millions of people are dependent on the river water systems that could be adversely impacted
by the possible inundation and rapid changes in the ecosystem.',
'As it is water demand is rising and a Unesco report earlier this year warned that central and south India
will face high levels of deterioration of water supply by 2050. '] '''


doc1 = "Rising sea level spells disaster for delta regions"
doc2 = "Sea levels are rising at different rates along the Indian coast, with projections till the end of the current century varying between 3.5 and 34.6 inches and which may pose a danger to major deltas, including the Cauvery delta, in east India and vast stretches of the western coastline, including Mumbai."
doc3 = "Sharing studies by the Hyderabadbased Indian National Centre for Ocean Information Services, the government on Friday told the Lok Sabha that sea level rise between 1990 and 2100 can make west coast stretches like Khambat and Kutch in Gujarat, Mumbai and parts of Konkan and south Kerala most vulnerable."
doc4 = "The threats posed by sea level rise due to global warming have direct implications for India’s food security as hundreds of millions of people are dependent on the river water systems that could be adversely impacted by the possible inundation and rapid changes in the ecosystem."
doc5 = "As it is water demand is rising and a Unesco report earlier this year warned that central and south India will face high levels of deterioration of water supply by 2050. "

#print(type(doc1))
 # compile documents
doc_complete = [doc1, doc2, doc3, doc4, doc5]

print("\n")
print("SAMPLE PARAGRAPH TAKEN:")
print("\n")
print(doc_complete)

from nltk.corpus import stopwords   
#stopwords are like 'the' which is removed
#stemming is to find the root word like fish is the root word for fishes and fishing
#lemmatization is to extract 'play' for playing plays played
from nltk.stem.wordnet import WordNetLemmatizer
import string

stop = set(stopwords.words('english'))
#print(stop)
exclude = set(string.punctuation)
#print(exclude)
lemma = WordNetLemmatizer()

def clean(doc):
	#document converted to lower case and then split as words
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
	#include the words which are not in stop words
	
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
	#From the stop words remove punctuation
	
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
	#from the punctuation free stop words do lemmatization
	
    return normalized

#Cleaning document by removing stopwords and by lemmatizing
doc_clean = [clean(doc).split() for doc in doc_complete]   

# Importing Gensim to create matrix
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus,
#where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(k) for k in doc_clean]

'''
Convert document (a list of words) into the bag-of-words format = list of (token_id, token_count) 2-tuples.
Each word is assumed to be a tokenized and normalized string (either unicode or utf8-encoded). 
No further preprocessing is done on the words in document; 
apply tokenization, stemming etc. before calling this method.'''

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Training LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=2, id2word = dictionary, passes=50)
#doc_term_matrix is passed
#num_topics (int, optional) – The number of topics to be selected, if -1 - all topics will be in result 
#Mapping from word IDs to words. It is used to determine the vocabulary size, as well as for debugging and topic printing.

print("\n")
print("POSSIBLE TOPICS INFERRED FROM GIVEN PARAGRAPH:")
print("\n")
result=ldamodel.print_topics(num_topics=1, num_words=8)
print(result)
#Each line is a topic with individual topic terms and weights.


print("\n")
comments=doc_complete
analyzer = SentimentIntensityAnalyzer()
print("\n")
print("Analysis of news paragraph taken:")
print("\n")

#scaling it to the polarity scores
for corpus in comments:
    score = analyzer.polarity_scores(corpus)
    print("{:-<70} {}".format(corpus, str(score)))
    #formatting the corpus on left and it's scores on right
print("\n")

#collecting as a summary
summary = {"positive":0,"neutral":0,"negative":0}
for x in comments: 
    ss = analyzer.polarity_scores(x)
    if ss["compound"] == 0.0: 
        summary["neutral"] +=1
    elif ss["compound"] > 0.0:
        summary["positive"] +=1
    else:
        summary["negative"] +=1
print(summary)

#print(type(summary))--> dict

#splitting as keys and values
keys= list(summary)  
temps = summary.values()
vals = list(temps)  

#plotting as a chart
figureObject, axesObject = plt.subplots()
axesObject.pie(vals,
        labels=keys,
		autopct='%1.2f',
        startangle=90)

# Aspect ratio - equal means pie is a circle
axesObject.axis('equal')
plt.show()
