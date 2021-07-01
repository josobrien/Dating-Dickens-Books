import nltk
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

#format plots
plt.style.use("seaborn")
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('figure', titlesize=18)  # fontsize of the figure title

def importFile(filePath): #function to read in and format each text
    with open(filePath, "r") as file:
        fileText = file.read().replace("\n\n", " ") 
    fileText=fileText.replace("\'", "'")
    fileText=fileText.replace("\n", " ")
    fileText=fileText.replace("   ", " ")
    fileText=fileText.replace("  ", " ")
    return fileText

#import the extract files and create some other versions in new variables

#training set

bleak = importFile("C:/Users/josob/Documents/Uni/3rd Year/MAS360/P4/BLEAKH.TXT")
bleaklim=bleak[:4116] #some extracts are shorter than others, sometimes this limited version will be needed as a countermeasure
bleakTok=nltk.word_tokenize(bleak) #separate into words
bleakComps=np.array(nltk.pos_tag(nltk.word_tokenize(bleak))) #array of these words and their word types (noun, adjective etc)
bleakNopunc=bleak.translate(str.maketrans('', '', string.punctuation)) #without punctuation


copper = importFile("C:/Users/josob/Documents/Uni/3rd Year/MAS360/P4/COPPER.TXT")
copperlim=copper[:4116]
copperTok=nltk.word_tokenize(copper)
copperComps=np.array(nltk.pos_tag(nltk.word_tokenize(copper)))
copperNopunc=copper.translate(str.maketrans('', '', string.punctuation))


ldorit = importFile("C:/Users/josob/Documents/Uni/3rd Year/MAS360/P4/LDORIT.TXT")
ldoritlim=ldorit[:4116]
ldoritTok=nltk.word_tokenize(ldorit)
ldoritComps=np.array(nltk.pos_tag(nltk.word_tokenize(ldorit)))
ldoritNopunc=ldorit.translate(str.maketrans('', '', string.punctuation))


mdrood = importFile("C:/Users/josob/Documents/Uni/3rd Year/MAS360/P4/MDROOD.TXT")
mdroodlim=mdrood[:4116]
mdroodTok=nltk.word_tokenize(mdrood)
mdroodComps=np.array(nltk.pos_tag(nltk.word_tokenize(mdrood)))
mdroodNopunc=mdrood.translate(str.maketrans('', '', string.punctuation))


oldcshop = importFile("C:/Users/josob/Documents/Uni/3rd Year/MAS360/P4/OLDCSHOP.TXT")
oldcshoplim=oldcshop[:4116]
oldcshopTok=nltk.word_tokenize(oldcshop)
oldcshopComps=np.array(nltk.pos_tag(nltk.word_tokenize(oldcshop)))
oldcshopNopunc=oldcshop.translate(str.maketrans('', '', string.punctuation))


thepp = importFile("C:/Users/josob/Documents/Uni/3rd Year/MAS360/P4/THEPP.TXT")
thepplim=thepp[:4116]
theppTok=nltk.word_tokenize(thepp)
theppComps=np.array(nltk.pos_tag(nltk.word_tokenize(thepp)))
theppNopunc=thepp.translate(str.maketrans('', '', string.punctuation))


#testing set

great = importFile("C:/Users/josob/Documents/Uni/3rd Year/MAS360/P4/GREATEX.TXT")
greatlim=great[:4116]
greatTok=nltk.word_tokenize(great)
greatComps=np.array(nltk.pos_tag(nltk.word_tokenize(great)))
greatNopunc=great.translate(str.maketrans('', '', string.punctuation))


chimes = importFile("C:/Users/josob/Documents/Uni/3rd Year/MAS360/P4/CHIMES.TXT")
chimeslim=chimes[:4116]
chimesTok=nltk.word_tokenize(chimes)
chimesComps=np.array(nltk.pos_tag(nltk.word_tokenize(chimes)))
chimesNopunc=chimes.translate(str.maketrans('', '', string.punctuation))


cities = importFile("C:/Users/josob/Documents/Uni/3rd Year/MAS360/P4/CITIES.TXT")
citieslim=cities[:4116]
citiesTok=nltk.word_tokenize(cities)
citiesComps=np.array(nltk.pos_tag(nltk.word_tokenize(cities)))
citiesNopunc=cities.translate(str.maketrans('', '', string.punctuation))


haunted = importFile("C:/Users/josob/Documents/Uni/3rd Year/MAS360/P4/HAUNTED.TXT")
hauntedlim=haunted[:4116]
hauntedTok=nltk.word_tokenize(haunted)
hauntedComps=np.array(nltk.pos_tag(nltk.word_tokenize(haunted)))
hauntedNopunc=haunted.translate(str.maketrans('', '', string.punctuation))


#unknowns

ourmf = importFile("C:/Users/josob/Documents/Uni/3rd Year/MAS360/P4/OURMF.TXT")
ourmflim=ourmf[:4116]
ourmfTok=nltk.word_tokenize(ourmf)
ourmfComps=np.array(nltk.pos_tag(nltk.word_tokenize(ourmf)))
ourmfNopunc=ourmf.translate(str.maketrans('', '', string.punctuation))


hardt=importFile("C:/Users/josob/Documents/Uni/3rd Year/MAS360/P4/HARDT.TXT")
hardtlim=hardt[:4116]
hardtTok=nltk.word_tokenize(hardt)
hardtComps=np.array(nltk.pos_tag(nltk.word_tokenize(hardt)))
hardtNopunc=hardt.translate(str.maketrans('', '', string.punctuation))


    
allTok=[theppTok,oldcshopTok,chimesTok,hauntedTok,copperTok,bleakTok,hardtTok,ldoritTok,citiesTok,greatTok,ourmfTok,mdroodTok]
allToklim=[nltk.word_tokenize(thepplim),nltk.word_tokenize(oldcshoplim),nltk.word_tokenize(chimeslim),nltk.word_tokenize(hauntedlim),nltk.word_tokenize(copperlim),nltk.word_tokenize(bleaklim),nltk.word_tokenize(hardtlim),nltk.word_tokenize(ldoritlim),nltk.word_tokenize(citieslim),nltk.word_tokenize(greatlim),nltk.word_tokenize(ourmflim),nltk.word_tokenize(mdroodlim)]
nopuncTok=[nltk.word_tokenize(theppNopunc),nltk.word_tokenize(oldcshopNopunc),nltk.word_tokenize(chimesNopunc),nltk.word_tokenize(hauntedNopunc),nltk.word_tokenize(copperNopunc),nltk.word_tokenize(bleakNopunc),nltk.word_tokenize(hardtNopunc),nltk.word_tokenize(ldoritNopunc),nltk.word_tokenize(citiesNopunc),nltk.word_tokenize(greatNopunc),nltk.word_tokenize(ourmfNopunc),nltk.word_tokenize(mdroodNopunc)]
allLens=[len(thepp),len(oldcshop),len(chimes),len(haunted),len(copper),len(bleak),len(hardt),len(ldorit),len(cities),len(great),len(ourmf),len(mdrood)]
allNltkTok=[theppComps,oldcshopComps,chimesComps,hauntedComps,copperComps,bleakComps,hardtComps,ldoritComps,citiesComps,greatComps,ourmfComps,mdroodComps]
ordYears=[1836,1840,1844,1848,1849,1852,1854,1855,1859,1860,1864,1870]


#find average word length for each book
allLenAvgs=[]
for i in range(len(nopuncTok)):
    wordLen=0
    for j in range(len(nopuncTok[i])):
        wordLen+=len(nopuncTok[i][j])
    allLenAvgs.append(wordLen/len(nopuncTok[i]))
plt.plot(ordYears,allLenAvgs)
plt.title("Average Word Length Over Time")
plt.xlabel("Year of Publishing")
plt.ylabel("Number of Characters Per Word")
#plt.savefig("C:/Users/josob/Documents/Uni/3rd Year/MAS360/P4/wordLenPlot.png")
plt.show()


#find number of unique words and punctuation for each book
#find number of unique words/punctuation for each book (limited to 4116 characters) - solved
uniqueTotals=[]
for i in range(len(allToklim)):
    uniqueTotals.append(len(set(allToklim[i])))
plt.plot(ordYears,uniqueTotals)
plt.title("Number of Unique Words and Punctuation Over Time")
plt.xlabel("Year of Publishing")
plt.ylabel("Number of Unique Words and Punctuation")
#plt.savefig("C:/Users/josob/Documents/Uni/3rd Year/MAS360/P4/uniqueTotalsPlot.png")
plt.show()


#find number of different punctuation types for each book (limited to 4116 characters)
# limit lengths of extracts - solved
puncTypes=[]
for i in range(len(allToklim)):
    puncList=[",",".",";","[","]","(",")","{","}",":","~","?","!","&","/","-","_","=","+","*"]
    puncCount=0
    for j in range(len(allToklim[i])):
        for k in allToklim[i][j]:
            if k in puncList:
                puncList.remove(k)
                puncCount+=1
    puncTypes.append(puncCount)
plt.plot(ordYears,puncTypes)
plt.title("Number of Unique Punctuation Marks Over Time")
plt.xlabel("Year of Publishing")
plt.ylabel("Number of Unique Punctuation Marks")
#plt.savefig("C:/Users/josob/Documents/Uni/3rd Year/MAS360/P4/puncTypesPlot.png")
plt.show()


#find number of sentences for each book
#find number of characters per sentence for each book - solved
sentenceTotals=[]
for i in range(len(allTok)):
    sentenceEndList=[".","?","!"]
    sentenceCount=0
    for j in allTok[i]:
        if j in sentenceEndList:
            sentenceCount+=1
    sentenceTotals.append(allLens[i]/sentenceCount)
plt.plot(ordYears,sentenceTotals)
plt.title("Average Sentence Length Over Time")
plt.xlabel("Year of Publishing")
plt.ylabel("Number of Characters Per Sentence")
#plt.savefig("C:/Users/josob/Documents/Uni/3rd Year/MAS360/P4/sentenceTotalsPlot.png")
plt.show()


#find total number of punctuation marks for each book
#find punctuation per character for each book - removed ' to minimise impact of speech - solved
puncTotals=[]
puncList=[",",".",";","[","]","(",")","{","}",":","~","?","!","&","/","-","_","=","+","*"]
for i in range(len(allTok)):
    puncCount=0
    for j in range(len(allTok[i])):
        for k in allTok[i][j]:
            if k in puncList:
                puncCount+=1
    puncTotals.append(puncCount/allLens[i])
plt.plot(ordYears,puncTotals)
plt.title("Punctuation Marks Per Character Over Time")
plt.xlabel("Year of Publishing")
plt.ylabel("Punctuation Marks Per Character")
#plt.savefig("C:/Users/josob/Documents/Uni/3rd Year/MAS360/P4/puncTotalsPlot.png")
plt.show()


#find number of each (nltk) type of token per character for each book
nltkTypes=["CC" ,"CD" ,"DT" ,"EX" ,"FW" ,"IN" ,"JJ" ,"JJR" ,"JJS" ,"LS" ,"MD" ,"NN" ,"NNS" ,"NNP" ,"NNPS" ,"PDT" ,"POS" ,"PRP" ,"PRP$" ,"RB" ,"RBR" ,"RBS" ,"RP" ,"SYM" ,"TO" ,"UH" ,"VB" ,"VBD" ,"VBG" ,"VBN" ,"VBP" ,"VBZ" ,"WDT" ,"WP" ,"WP$" ,"WRB"]
nltkTypeTotals=[]
for i in range(len(allNltkTok)):
    countList=[]
    for j in nltkTypes:
        typeCount=np.count_nonzero(allNltkTok[i]==j)
        countList.append(typeCount/allLens[i])
    nltkTypeTotals.append(countList)



columns=["wordLen","uniqueWords","puncTotalNum","puncTypesNum","sentenceNum"]#+nltkTypes
scaler=StandardScaler()

df=pd.DataFrame(np.concatenate([np.reshape(allLenAvgs, [12,1]), np.reshape(uniqueTotals, [12,1]), np.reshape(puncTotals, [12,1]), np.reshape(puncTypes, [12,1]), np.reshape(sentenceTotals, [12,1]) ],1))# ,  nltkTypeTotals],1))
scaler.fit(df)
features=pd.DataFrame(scaler.transform(df),columns=columns,index=["thepp","oldcshop","chimes","haunted","copper","bleak","hardt","ldorit","cities","great","ourmf","mdrood"])
testBooks=features.iloc[[2,3,8,9]]
testBooksY5=np.array([2,3,5,5])
testBooksY10=np.array([1,2,3,3])
testBooksY20=np.array([1,1,2,2])
newBooks=features.iloc[[6,10]]
features=features.drop(["chimes","hardt","great","ourmf","haunted","cities"])
y5=np.array([1,1,3,4,4,7])
y10=np.array([1,1,2,2,3,4])
y20=np.array([1,1,1,2,2,2])


#LDA
lda=LDA()
lda.fit(features,y20)

lda.predict(features)
lda.score(features,y20)
lda.score(testBooks,testBooksY20)
lda.predict(testBooks)

lda.predict(newBooks)


""" WITH NLTK
On features,
y5, 67%, not great
y10, 83%, not bad
y20, 67%, somehow worse?

#On testBooks BEFORE 4 testBooks 
#y5, [1,1], 0%, but impossible for it to get right since trained on y5=[1,1,3,4,4,7], but correct is [2,5]
#y10, [1,1] 50% should be [1,3]
#y20, [1,1] 50% should be [1,2], seems to think they are both early books, they are 1844 and 1860 though


After 4 testBooks,
y5, [1,1,4,1] 0% should be [2,3,5,5] but impossible to get 2 and 5 so unfair
y10, [1,1,2,1] 25% should be [1,2,3,3] 
y20, [1,1,2,1], 75% should be [1,1,2,2] 

WITHOUT NLTK
On features,
y5, 50%
y10, 67%
y20, 83%
4 books,
y5, 0%, [1, 4, 1, 4] should be [2,3,5,5] but impossible to get 2 and 5 so unfair
y10, 50%, [1, 2, 1, 2] should be [1,2,3,3]
y20, 25%, [2, 2, 1, 2] should be [1,1,2,2]

ONLY WORDLEN AND SENTENCELEN
4 books,
y5, 0%, [1, 1, 1, 4] should be [2,3,5,5] but impossible to get 2 and 5 so unfair
y10, 75%, [1, 2, 1, 3] should be [1,2,3,3]
y20, 75%, [1, 1, 1, 2] should be [1,1,2,2]

ONLY WORDLEN, UNIQUE WORDS AND SENTENCELEN 
4 books,
y5, didn't test
y10, 25%, [1, 1, 1, 4] should be [1,2,3,3]
y20, 75%, [1, 1, 1, 2] should be [1,1,2,2]

ONLY WORDLEN AND UNIQUE WORDS
4 books,
y10, 25%, [1, 1, 1, 4]
y20, 50%, [2, 2, 2, 2]
"""

score=lda.score(features,y20)
y_pred=lda.fit(features,y20).predict(features)
cm=metrics.confusion_matrix(y20,y_pred)
sns.heatmap(cm,annot=True,fmt="d",linewidths=.5,square=True)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title='Accuracy Score: {0}'.format(score)
plt.title(all_sample_title,size=15)
plt.show()


#QDA
qda=QDA()
qda.fit(features,y20)

qda.predict(features)
qda.score(features, y20)

qda.predict(testBooks)
qda.score(testBooks,testBooksY20)

""" ACTUAL RESULT
ALL
y20, 0%, [1,1] should be [2,2]

WITHOUT NLTK
y20, 0%, [1,1]

ONLY WORDLEN, UNIQUE WORDS AND SENTENCELEN 
y20, 0%, [1,1]

ONLY WORDLEN AND UNIQUE WORDS
y20, 0%, [1,1]
"""

""" WITH NLTK
needs more than one book in each class, doesn't work with y5 and y10.

On features,
y20, 100%, good sign, the risk now is overfitting not underfitting


4 books,
y20, [1,1,2,2], 100% correct

WITHOUT NLTK
4 books,
y20, 50%, [2, 2, 2, 2] should be [1,1,2,2]

ONLY WORDLEN AND SENTENCELEN
4 books,
y20, 75%, [1, 2, 2, 2] should be [1,1,2,2]

ONLY WORDLEN, UNIQUE WORDS AND SENTENCELEN 
4 books,
y20, 50%, [1, 1, 1, 1] should be [1,1,2,2]

ONLY WORDLEN AND UNIQUE WORDS
4 books,
y20, 50%, [1, 2, 1, 2]
"""


#cm=metrics.confusion_matrix(y20,y_pred)
#sns.heatmap(cm,annot=True,fmt="d",linewidths=.5,square=True)
#plt.ylabel('Actual label')
#plt.xlabel('Predicted label')
#all_sample_title='Accuracy Score: {0}'.format(score)
#plt.title(all_sample_title,size=15)


#Decision tree
treeclf=DecisionTreeClassifier(max_depth=2)
treeclf.fit(features,y10)

metrics.accuracy_score(y20,treeclf.predict(features))
treeclf.predict(features)

print(metrics.accuracy_score(testBooksY10,treeclf.predict(testBooks)))
treeclf.predict(testBooks)

np.around(treeclf.feature_importances_*100, decimals=1)

#loop and get average score
x=1000
treeclfScore=0
for i in range(x):
    treeclf=DecisionTreeClassifier(max_depth=2)#2 for 10, 8 for 20
    treeclf.fit(features,y10)
    #print(treeclf.predict(testBooks))
    treeclfScore+=metrics.accuracy_score(testBooksY10,treeclf.predict(testBooks))
treeclfScore/x

#loop and get range scores
x=500
scores=[]
for j in range(1,11):
    treeclfScore=0
    treeclf=DecisionTreeClassifier(max_depth=j)
    for i in range(x):
        treeclf.fit(features,y20)
        #print(treeclf.predict(testBooks))
        treeclfScore+=metrics.accuracy_score(testBooksY20,treeclf.predict(testBooks))
    scores.append([j,treeclfScore*100/x])
scores=np.array(scores)
plt.plot(scores[:,0],scores[:,1])
plt.xlabel("Max Depth")
plt.ylabel("Accuracy %")
plt.title("Decision Tree Accuracy by Max Depth (20 year)")
#plt.savefig("C:/Users/josob/Documents/Uni/3rd Year/MAS360/P4/treeclf20.png")
plt.show()

""" ACTUAL RESULT
ALL
y10, average 33%
y20, average 51%

WITHOUT NLTK
y10, average 21%
y20, average 50%

ONLY WORDLEN, UNIQUE WORDS AND SENTENCELEN 
y10, average 0%
y20, average 53%

ONLY WORDLEN AND UNIQUE WORDS
y10, average 50%
y20, average 0%
"""

"""WITH NLTK
4 books,
y5, average 3% lol
y10, average 32% 
y20, average 63%

WITHOUT NLTK 
4 books,
y5, average 0%
y10, average 34%
y20, average 69%

ONLY WORDLEN AND SENTENCELEN 
4 books,
y5, average ?
y10, average ?
y20, average ?

ONLY WORDLEN, UNIQUE WORDS AND SENTENCELEN  
4 books,
y5, average 0%
y10, average 36%
y20, average 70%

ONLY WORDLEN AND UNIQUE WORDS 
4 books,
y5, average 0%
y10, average 75% 
y20, average 75%
"""


#Decision tree regression (continuous), only useful in attempting to give year of closest book in training set
yCont=[1836,1840,1849,1852,1855,1870]

treereg=DecisionTreeRegressor(max_depth=20)
treereg.fit(features,yCont)

treereg.predict(features)

treereg.predict(testBooks)

if test==3:
    print("\nmse =", ((treereg.predict(testBooks)[0] -1844)**2 + (treereg.predict(testBooks)[1] -1854)**2 + (treereg.predict(testBooks)[2] -1860)**2 + (treereg.predict(testBooks)[3] -1864)**2)/4)
elif test==4:
    ypred=treereg.predict(testBooks)
    print("\nmse =", ((ypred[0] -1844)**2 + (ypred[1] -1848)**2 + (ypred[2] -1859)**2 + (ypred[3] -1860)**2)/4)
else:
    print("\nmse =", ((treereg.predict(testBooks)[0] -1854)**2 + (treereg.predict(testBooks)[1] -1864)**2)/2)

np.around(treereg.feature_importances_*100, decimals=1)

#average mse
x=1000
mse=0
for i in range(x):
    treereg.fit(features,yCont)
    ypred=treereg.predict(testBooks)
    mse+=((ypred[0] -1844)**2 + (ypred[1] -1848)**2 + (ypred[2] -1859)**2 + (ypred[3] -1860)**2)/4
mse/x


#Random forest
rfclf=RandomForestClassifier(n_estimators=15)
rfclf.fit(features,y10)

metrics.accuracy_score(y20,rfclf.predict(features))
rfclf.predict(features)

print(metrics.accuracy_score(testBooksY10,rfclf.predict(testBooks)))
rfclf.predict(testBooks)

np.around(rfclf.feature_importances_*100, decimals=1)

#for rfclf average accuracy for single n_estimators value
#"""
x=100
tot=0
avgoutput=[0,0]
for i in range(x):
    rfclf=RandomForestClassifier(n_estimators=80)#15or20 for 10, 80 for 20
    rfclf.fit(features,y20)
    #tot+=metrics.accuracy_score(testBooksY20,rfclf.predict(testBooks))
    avgoutput=[a+b for a,b, in zip(avgoutput,rfclf.predict(testBooks))]
#tot/x
avgoutput=[a/x for a in avgoutput]
#"""

""" ACTUAL RESULT
ALL
y10, 28%
y20, 60%

WITHOUT NLTK
y10, 36%
y20, 92%

ONLY WORDLEN, UNIQUE WORDS AND SENTENCELEN 
y10, 33%
y20, 99%

ONLY WORDLEN AND UNIQUE WORDS
y10, 50%
y20, 55%
"""

"""WITH NLTK, 
4 books,
y5, average 0%
y10, average 26%, 15 estimators
y20, average 73%, 100 estimators

WITHOUT NLTK
4 books, 
y5, average 0%
y10, average 39%
y20, average 65%

ONLY WORDLEN AND SENTENCELEN
4 books,
y5, average ?
y10, average ?
y20, average ?

ONLY WORDLEN, UNIQUE WORDS AND SENTENCELEN 
4 books,
y5, average 0%
y10, average 29%
y20, average 75%

ONLY WORDLEN AND UNIQUE WORDS
4 books,
y5, average 0%
y10, average 75%
y20, average 75%
"""
 
#Random forest continuous variable output (year)
yCont=[1836,1840,1849,1852,1855,1870]

rfreg=RandomForestRegressor(n_estimators=80)
rfreg.fit(features,yCont)

rfreg.predict(features)

rfreg.predict(testBooks)

if test==3:
    print("\nmse =", ((rfreg.predict(testBooks)[0] -1844)**2 + (rfreg.predict(testBooks)[1] -1854)**2 + (rfreg.predict(testBooks)[2] -1860)**2 + (rfreg.predict(testBooks)[3] -1864)**2)/4)
if test==4:
    print("\nmse =", ((rfreg.predict(testBooks)[0] -1844)**2 + (rfreg.predict(testBooks)[1] -1848)**2 + (rfreg.predict(testBooks)[2] -1859)**2 + (rfreg.predict(testBooks)[3] -1860)**2)/4)
else:
    print("\nmse =", ((rfreg.predict(testBooks)[0] -1854)**2 + (rfreg.predict(testBooks)[1] -1864)**2)/2)

np.around(rfreg.feature_importances_*100, decimals=1)


#for rfreg average mse for single n_estimators value, when test=4 only
#"""
x=100
tot=0
for j in range(x):
    rfreg=RandomForestRegressor(n_estimators=80)
    rfreg.fit(features,yCont)
    tot+=((rfreg.predict(testBooks)[0] -1844)**2 + (rfreg.predict(testBooks)[1] -1848)**2 + (rfreg.predict(testBooks)[2] -1859)**2 + (rfreg.predict(testBooks)[3] -1860)**2)/4
tot/x
#"""

#for rfreg average mse for single n_estimators value, when test=0 only
#"""
x=100
tot=0
for j in range(x):
    rfreg=RandomForestRegressor(n_estimators=80)
    rfreg.fit(features,yCont)
    tot+=((rfreg.predict(testBooks)[0] -1854)**2 + (rfreg.predict(testBooks)[1] -1864)**2)/2
tot/x
#"""

#for rfreg average year for single n_estimators value, when test=0 only
#"""
x=100
tot=[0,0]
for j in range(x):
    rfreg=RandomForestRegressor(n_estimators=80)
    rfreg.fit(features,yCont)
    ypred=rfreg.predict(testBooks)
    tot=[a+b for a,b, in zip(tot,ypred)]
[a/x for a in tot]
#"""

""" ACTUAL RESULT
ALL
average mse: 56

WITHOUT NLTK
average mse: 22

ONLY WORDLEN, UNIQUE WORDS AND SENTENCELEN 
average mse: 9.2

ONLY WORDLEN AND UNIQUE WORDS
average mse: 6.3
"""

"""WITH NLTK, estimators = 80
4 books, 
average mse: 38

WITHOUT NLTK
4 books,
average mse: 29

ONLY WORDLEN AND SENTENCELEN
4 books,
average mse: 26

ONLY WORDLEN, UNIQUE WORDS AND SENTENCELEN
4 books,
average mse: 22

ONLY WORDLEN AND UNIQUE WORDS
4 books,
average mse: 19
"""


#for rfreg average (mse) for range of n_estimators values 
"""
totals=[]
mse=[]
x=100
mult=1
for i in range(80,81):
    tot=[0,0]
    for j in range(x):
        rfreg=RandomForestRegressor(n_estimators=mult*i)
        rfreg.fit(features,yCont)
        tot=[a + b for a, b in zip(tot, rfreg.predict(testBooks))]
    totals.append([np.around(a/x,decimals=1) for a in tot])
    mse.append([mult*i,((tot[0]/x -1854)**2 + (tot[1]/x -1864)**2)/2])
mse=np.array(mse)
plt.plot(mse[:,0],mse[:,1])
#"""

#for rfreg average mse for single n_estimators value, when test=4 only
"""
x=100
tot=0
for j in range(x):
    rfreg=RandomForestRegressor(n_estimators=80)
    rfreg.fit(features,yCont)
    tot+=((rfreg.predict(testBooks)[0] -1844)**2 + (rfreg.predict(testBooks)[1] -1848)**2 + (rfreg.predict(testBooks)[2] -1859)**2 + (rfreg.predict(testBooks)[3] -1860)**2)/4
tot/x
"""

#for rfreg good predictor feature importances at single n_estimators
"""
importances=[0,0,0,0,0]
x=1000
count=0
rfreg=RandomForestRegressor(n_estimators=80)
for i in range(x):
    
    rfreg.fit(features,yCont)
    if ((rfreg.predict(testBooks)[0] -1854)**2 + (rfreg.predict(testBooks)[1] -1864)**2)/2 < 10:
        importances=[a + b for a, b in zip(importances, rfreg.feature_importances_)]
        count+=1
np.around([a*100/count for a in importances],decimals=1)
#"""

#for rfreg good predictor feature importances at single n_estimators, test=4 only
#"""
importances=[0,0,0,0,0]
x=1000
count=0
rfreg=RandomForestRegressor(n_estimators=80)
for i in range(x):
    rfreg.fit(features,yCont)
    ypred=rfreg.predict(testBooks)
    if ((ypred[0] -1844)**2 + (ypred[1] -1848)**2 + (ypred[2] -1859)**2 + (ypred[3] -1860)**2)/4 < 19.5:
        importances=[a + b for a, b in zip(importances, rfreg.feature_importances_)]
        count+=1
goodimportances=[a*100/count for a in importances]
np.around(goodimportances,decimals=1)
#"""  [23.3, 13.2, 21.1, 16. , 26.3]

#for rfreg average feature importances at single n_estimators, test=4 only
#"""
importances=[0,0,0,0,0]
x=1000
rfreg=RandomForestRegressor(n_estimators=80)
for i in range(x):
    rfreg.fit(features,yCont)
    ypred=rfreg.predict(testBooks)
    importances=[a + b for a, b in zip(importances, rfreg.feature_importances_)]
avgimportances=[a*100/x for a in importances]
np.around(avgimportances,decimals=1)
#""" [22.4, 11.6, 21.8, 17.5, 26.7]

np.around([a - b for a, b in zip(goodimportances, avgimportances)],decimals=2)
#[1.21,  0.85, -0.78, -1.34,  0.06]

features=features.drop(columns=["puncTotalNum","puncTypesNum"])
testBooks=testBooks.drop(columns=["puncTotalNum","puncTypesNum"])


#[37.9, 20.8, 41.3]
#[37.7, 19. , 43.3]
#[ 0.16,  1.82, -1.98]
features=features.drop(columns=["puncTotalNum","puncTypesNum","sentenceNum"])
testBooks=testBooks.drop(columns=["puncTotalNum","puncTypesNum","sentenceNum"])

#1000 time average:   [24.5, 19.2,  6.1, 18.3, 31.8]
#example with 23 mse: [26.9, 16.5,  4.2, 15.2, 37.3]
#middle 3 are less used in the better one. What if I only gave it the first and last variables to work with?
features=features.drop(columns=["uniqueWords","puncTotalNum","puncTypesNum"])
testBooks=testBooks.drop(columns=["uniqueWords","puncTotalNum","puncTypesNum"])


#corrected features 1000 time average: [19.9,  9.1, 19.2, 17.6, 34.1]
#example with 10 mse:                  [19.6,  5.9, 12.6, 20.7, 41.2]
#example with 6 mse:                   [24.1,  8.9, 19.7, 11.1, 36.2]
#avg from 29 examples <10 mse:         [22.3,  8.9, 16.2, 14.8, 37.9]
#remove least used columns from features:

rfreg=RandomForestRegressor(n_estimators=80)
rfreg.fit(features,yCont)

rfreg.predict(features)

rfreg.predict(testBooks)

if test==3:
    print("\nmse =", ((rfreg.predict(testBooks)[0] -1844)**2 + (rfreg.predict(testBooks)[1] -1854)**2 + (rfreg.predict(testBooks)[2] -1860)**2 + (rfreg.predict(testBooks)[3] -1864)**2)/4)
if test==4:
    print("\nmse =", ((rfreg.predict(testBooks)[0] -1844)**2 + (rfreg.predict(testBooks)[1] -1848)**2 + (rfreg.predict(testBooks)[2] -1859)**2 + (rfreg.predict(testBooks)[3] -1860)**2)/4)
else:
    print("\nmse =", ((rfreg.predict(testBooks)[0] -1854)**2 + (rfreg.predict(testBooks)[1] -1864)**2)/2)


np.around(rfreg.feature_importances_*100, decimals=1)

#for rfclf average accuracy for range of n_estimators values
#"""
totals=[]
x=50
mult=5
for i in range(1,20):
    tot=0
    rfclf=RandomForestClassifier(n_estimators=mult*i)
    for j in range(x):
        rfclf.fit(features,y20)
        tot+=metrics.accuracy_score(testBooksY20,rfclf.predict(testBooks))
    totals.append([mult*i,tot/x])
totals=np.array(totals)
plt.plot(totals[:,0],totals[:,1])
plt.xlabel("Number of Estimators")
plt.ylabel("Accuracy %")
plt.title("Random Forest Classifier Accuracy by Number of Estimators (20 year)")
#plt.savefig("C:/Users/josob/Documents/Uni/3rd Year/MAS360/P4/rfclf20.png")
plt.show()
#"""

#for rfclf average feature importances at single n_estimators
"""
importances=[0,0,0,0,0]
x=100
for i in range(x):
    rfclf=RandomForestClassifier(n_estimators=80)
    rfclf.fit(features,y20)
    importances=[a + b for a, b in zip(importances, rfclf.feature_importances_)]
np.around(importances,decimals=1)
#"""


#PCA
pca=PCA()
p=pca.fit_transform(features)

"""
for i in range(len(y5)):
    plt.scatter(x=p[i,0],y=p[i,1],label=ordYears[i])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PC1 vs PC2 ")
plt.legend(title="Year")
#plt.savefig("C:/Users/josob/Documents/Uni/3rd Year/")
plt.show()
"""

cumexp=np.concatenate([[0],pca.explained_variance_ratio_])
plt.plot(np.cumsum(cumexp))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Prop.")
plt.title("Scree Plot: Books PCA")
#plt.savefig("C:/Users/josob/Documents/Uni/3rd Year/")
plt.show()


#word length - maybe plot the distribution, look at skewness
#number of unique words
#sentence length - maybe same as punc (no. of [. ! ?])
#number of types of words and puncs
#number of types of punctuation
#natural language toolkit types of words

#number of each type of punctuation
    #https://jerichowriters.com/punctuation-writers/
#number of specific words - perhaps "and", "really", "very" (weak words?)...
    #https://mybookcave.com/authorpost/17-weak-words-to-avoid-in-your-writing/
    #https://thewritepractice.com/better-writer-now/
    #https://annerallen.com/2017/06/filter-words-and-phrases-to-avoid-in-writing/


#Can I use more data? - find other books, full books
#My plan is to use measures such as (above), is there any other way of doing it? Otherwise what measures do you suggest?
#Can I summarise the data with a single continuous variable (time) via machine learning


#order:                chapter/book   chapter title y/n   hyphen       5-year  10-year  20-year(before/after 1850)
#thepp 1836            chapter        yes                 no           1       1        1
#oldcshop 1840         chapter        no                  no           1       1        1
#copper 1849           chapter        yes                 no           3       2        1
#bleak 1852            chapter        yes                 no           4       2        2
#(hardt unknown (1854)        book           yes                 yes)   4       2        2
#ldorit 1855           book           yes                 no           4       3        2
#(ourmf unknown (1864)        book           yes                 no)    6       3        2
#mdrood 1870           chapter        yes                 yes          7       4        2

#y5=[1,1,3,4,4,4,6,7]
#y10=[1,1,2,2,2,3,3,4]
#y20=[1,1,1,2,2,2,2,2]

#chimes 1844           chapter                                         2       1        1
#great  1860           chapter                                         5       3        2

#y5=[1,1,2,3,4,4,4,5,6,7]
#y10=[1,1,1,2,2,2,3,3,3,4]
#y20=[1,1,1,1,2,2,2,2,2,2]

#haunted 1848                                                          3       2        1
#cities  1859                                                          5       3        2

#y5=[1,1,2,3,3,4,4,4,5,5,6,7]
#y10=[1,1,1,2,2,2,2,3,3,3,3,4]
#y20=[1,1,1,1,1,2,2,2,2,2,2,2]
