import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression


#Create your df here:
DF = pd.read_csv("profiles.csv", encoding='utf-8-sig')

#Information about the DataFrame.
DF.head()
DF.info()

#General exploration of Dataset:
##Education
DF['education'].value_counts()
DF['education'].describe()
EDUCATION_TYPE_COUNT = DF['education'].value_counts()
sns.set(style="darkgrid")
G = sns.stripplot(y=EDUCATION_TYPE_COUNT.index, x=EDUCATION_TYPE_COUNT.values, alpha=0.9, linewidth=1)
plt.title('Frequency Distribution of Education')
plt.ylabel('Education', fontsize=12)
plt.xlabel('Number of Occurrences (Log Scale)', fontsize=12)
G.set(xscale="log")
plt.show()

##Job
DF['job'].value_counts()
DF['job'].describe()
JOB_TYPE_COUNT = DF['job'].value_counts()
sns.set(style="darkgrid")
sns.stripplot(y=JOB_TYPE_COUNT.index, x=JOB_TYPE_COUNT.values, alpha=0.9, linewidth=1)
plt.title('Frequency Distribution of Job')
plt.ylabel('Type of Job', fontsize=12)
plt.xlabel('Number of Occurrences', fontsize=12)
plt.show()

##Diet
DF['diet'].value_counts()
DF['diet'].describe()
DIET_TYPE_COUNT = DF['diet'].value_counts()
sns.set(style="darkgrid")
G = sns.stripplot(y=DIET_TYPE_COUNT.index, x=DIET_TYPE_COUNT.values, alpha=0.9, linewidth=1)
plt.title('Frequency Distribution of Diet')
plt.ylabel('Type of Diet', fontsize=12)
plt.xlabel('Number of Occurrences (Log Scale)', fontsize=12)
G.set(xscale="log")
plt.show()

##Body Type
DF['body_type'].value_counts()
DF['body_type'].describe()
BODY_TYPE_COUNT = DF['body_type'].value_counts()
sns.set(style="darkgrid")
G = sns.stripplot(y=BODY_TYPE_COUNT.index, x=BODY_TYPE_COUNT.values, alpha=0.9, linewidth=1)
plt.title('Frequency Distribution of Body Types')
plt.ylabel('Body Type', fontsize=12)
plt.xlabel('Number of Occurrences (Log Scale)', fontsize=12)
G.set(xscale="log")
plt.show()

#Augmented Data: Mapping
EDUCATION_MAPPING = {
        "graduated from college/university": 0,
        "graduated from masters program": 1,
        "working on college/university": 2,
        "working on masters program": 3,
        "graduated from two-year college": 4,
        "graduated from high school": 5,
        "graduated from ph.d program": 6,
        "graduated from law school": 7,
        "working on two-year college": 8,
        "dropped out of college/university": 9,
        "working on ph.d program": 10,
        "college/university": 11,
        "graduated from space camp": 12,
        "dropped out of space camp": 13,
        "graduated from med school": 14,
        "working on space camp": 15,
        "working on law school": 16,
        "two-year college": 17,
        "working on med school": 18,
        "dropped out of two-year college": 19,
        "dropped out of masters program": 20,
        "masters program": 21,
        "dropped out of ph.d program": 22,
        "dropped out of high school": 23,
        "high school": 24,
        "working on high school": 25,
        "space camp": 26,
        "ph.d program": 27,
        "law school": 28,
        "dropped out of law school": 29,
        "dropped out of med school": 30,
        "med school": 31
        }

JOB_MAPPING = {
        'other': 0,
        'student': 1,
        'science / tech / engineering': 2,
        'computer / hardware / software': 3,
        'artistic / musical / writer': 4,
        'sales / marketing / biz dev': 5,
        'medicine / health': 6,
        'education / academia': 7,
        'executive / management': 8,
        'banking / financial / real estate': 9,
        'entertainment / media': 10,
        'law / legal services': 11,
        'hospitality / travel': 12,
        'construction / craftsmanship': 13,
        'clerical / administrative': 14,
        'political / government': 15,
        'rather not say': 16,
        'transportation': 17,
        'unemployed': 18,
        'retired': 19,
        'military': 20
        }

DIET_MAPPING = {
        'mostly anything': 0,
        'anything': 1,
        'strictly anything': 2,
        'mostly vegetarian': 3,
        'mostly other': 4,
        'strictly vegetarian': 5,
        'vegetarian': 6,
        'strictly other': 7,
        'mostly vegan': 8,
        'other': 9,
        'strictly vegan': 10,
        'vegan': 11,
        'mostly kosher': 12,
        'mostly halal': 13,
        'strictly kosher': 14,
        'strictly halal': 15,
        'halal': 16,
        'kosher': 17
        }

BODY_TYPE_MAPPING = {
        'average': 0,
        'fit': 1,
        'athletic': 2,
        'thin': 3,
        'curvy': 4,
        'a little extra': 5,
        'skinny': 6,
        'full figured': 7,
        'overweight': 8,
        'jacked': 9,
        'used up': 10,
        'rather not say': 11
        }

#New DataFrame from augmented data.
DF1 = pd.DataFrame()
DF1['job1'] = DF.job.map(JOB_MAPPING)
DF1['education1'] = DF.education.map(EDUCATION_MAPPING)
DF1['diet1'] = DF.diet.map(DIET_MAPPING)
DF1['body_type1'] = DF.body_type.map(BODY_TYPE_MAPPING)
DF1 = DF1.dropna()

#Frequency from augmented data on job
EDUCATION_FREQ = DF1.groupby(['education1', 'job1']).size().reset_index(name='job_frequency')
DIET_FREQ = DF1.groupby(['diet1', 'job1']).size().reset_index(name='job_frequency')
BODY_FREQ = DF1.groupby(['body_type1', 'job1']).size().reset_index(name='job_frequency')
EDUCATION_COUNTS = DF1.education1.value_counts()
DIET_COUNTS = DF1.diet1.value_counts()
BODY_TYPE_COUNTS = DF1.body_type1.value_counts()

#Percentage from augmented data
PERCENTAGEQ = []
for i in range(len(EDUCATION_FREQ)):
    percentage = round(100 * (EDUCATION_FREQ['job_frequency'][i] / EDUCATION_COUNTS[EDUCATION_FREQ['education1'][i]]), 2)
    PERCENTAGEQ.append(percentage)
EDUCATION_FREQ['job_percentage'] = PERCENTAGEQ

PERCENTAGEQ = []
for i in range(len(DIET_FREQ)):
    percentage = round(100 * (DIET_FREQ['job_frequency'][i] / DIET_COUNTS[DIET_FREQ['diet1'][i]]), 2)
    PERCENTAGEQ.append(percentage)
DIET_FREQ['job_percentage'] = PERCENTAGEQ

PERCENTAGEQ = []
for i in range(len(BODY_FREQ)):
    percentage = round(100 * (BODY_FREQ['job_frequency'][i] / BODY_TYPE_COUNTS[BODY_FREQ['body_type1'][i]]), 2)
    PERCENTAGEQ.append(percentage)
BODY_FREQ['job_percentage'] = PERCENTAGEQ


#Augmented scatterplot
FIG = plt.gcf()
FIG.set_size_inches(10, 6)
sns.scatterplot(x=DIET_FREQ['diet1'], y=DIET_FREQ['job1'], hue=DIET_FREQ['job_percentage'], s=DIET_FREQ['job_percentage']**2, alpha=0.9, linewidth=1, palette="Set1", legend=False)
plt.xlabel("Diet", fontsize=12)
plt.ylabel("Job", fontsize=12)
plt.title('Comparing Diet and Job')
plt.show()

FIG = plt.gcf()
FIG.set_size_inches(10, 6)
sns.scatterplot(x=EDUCATION_FREQ['education1'], y=EDUCATION_FREQ['job1'], hue=EDUCATION_FREQ['job_percentage'], s=EDUCATION_FREQ['job_percentage']**2, alpha=0.9, linewidth=1, palette="Set1", legend=False)
plt.xlabel("Education", fontsize=12)
plt.ylabel("Job", fontsize=12)
plt.title('Comparing Education and Job')
plt.show()

FIG = plt.gcf()
FIG.set_size_inches(10, 6)
sns.scatterplot(x=BODY_FREQ['body_type1'], y=BODY_FREQ['job1'], hue=BODY_FREQ['job_percentage'], s=BODY_FREQ['job_percentage']**2, alpha=0.9, linewidth=1, palette="Set1", legend=False)
plt.xlabel("Body Type", fontsize=12)
plt.ylabel("Job", fontsize=12)
plt.title('Comparing Body Type and Job')
plt.show()

#KNN
#https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/
X_train, X_test, y_train, y_test = train_test_split(DF1[DF1.columns[-3:]], DF1['job1'], random_state=42)

ACCURACIES = []
for k in range(1, 50):
    Classifier = KNeighborsClassifier(n_neighbors=k)
    Classifier.fit(X_train, y_train)
    ACCURACIES.append(Classifier.score(X_test, y_test))

FIG = plt.gcf()
FIG.set_size_inches(10, 6)
plt.plot(range(1, 50), ACCURACIES)
plt.xlabel('K')
plt.ylabel('Validation Accuracy')
plt.title('Job Classifier Accuracy')
plt.show()

print(ACCURACIES)

#Naive Bayes Classification
Classifier = MultinomialNB()
Classifier.fit(X_train, y_train)
score = Classifier.score(X_test, y_test)
print(score)

#Emotional words for regression?
#https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
essay_cols = ["essay0", "essay1", "essay2", "essay3", "essay4", "essay5", "essay6", "essay7", "essay8", "essay9"]

# Copied from capstone instruction to format the text data, remove NaN, lowercase and non-text values.
all_essays = DF[essay_cols].fillna('')
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
all_essays = all_essays.replace(regex=r'[\W]', value=' ').replace(regex=r'[\s_]+', value=' ')

#Word lists that I have created based on my really poor model of reality what higher/lower income people would say
list_lower_half = ['school', 'college', 'student', 'careless', 'sex', 'sexy', 'stupid', 'butt', 'ass', 'asshole', 'youtube', 'youtuber', 'boring', 'bored', 'lame', 'bad', 'video game', 'video games', 'vidya gaems', 'dream', 'dreaming', 'dreamer', 'chill', 'shit', 'annoying', 'mom', 'mother', 'basement', 'unemployed', 'bullshit', 'drinking', 'party', 'stress', 'stressful', 'hate', 'desperate', 'difficult', 'troubled', 'fun', 'sarcastic', 'fucking', 'fucka', 'fack', 'fucker', 'fucks', 'procrastination']
list_upper_half = ['work', 'working', 'workaholic', 'career', 'confident', 'smart', 'learn', 'learning', 'improve', 'improving', 'bank', 'banking', 'intelligence', 'intelligent', 'manager', 'competitive', 'competition', 'corp', 'corporation', 'house', 'adventure', 'adventurer', 'solve', 'solving', 'geek', 'nerd', 'optimistic', 'restaurant', 'restaurants', 'travel', 'travelling', 'employed', 'teach', 'teaching', 'honest', 'children', 'help', 'exploring', 'parent', 'son', 'daughter']

vectorizer = CountVectorizer()

vectorizer.fit(list_lower_half)
word_array = vectorizer.transform(all_essays).toarray()
counts_lower = pd.DataFrame(word_array, columns=vectorizer.get_feature_names()).sum(axis=1)
print(counts_lower)

vectorizer.fit(list_upper_half)
word_array = vectorizer.transform(all_essays).toarray()
counts_higher = pd.DataFrame(word_array, columns=vectorizer.get_feature_names()).sum(axis=1)
print(counts_higher)
#https://stackoverflow.com/questions/36572221/how-to-find-ngram-frequency-of-a-column-in-a-pandas-dataframe
#Income vs Type of Word
DF2 = pd.DataFrame()
DF2['income2'] = DF['income']
DF2['income2'] = DF2[DF2.income2 != -1]
DF2['lower_income_words'] = counts_lower
DF2['higher_income_words'] = counts_higher
DF2 = DF2.dropna()

G = sns.scatterplot(x=DF2['income2'], y=DF2['lower_income_words'], alpha=0.9, linewidth=1)
plt.xlabel('Income (Log Scale)', fontsize=12)
plt.ylabel('Lower Income Words', fontsize=12)
G.set(xscale="log")
plt.title('Income V Lower Income Words')
plt.show()

G = sns.scatterplot(x=DF2['income2'], y=DF2['higher_income_words'], alpha=0.9, linewidth=1)
plt.xlabel('Income (Log Scale)', fontsize=12)
plt.ylabel('Higher Income Words', fontsize=12)
G.set(xscale="log")
plt.title('Income V Higher Income Words')
plt.show()

X = DF2[['lower_income_words', 'higher_income_words']]
y = DF2['income2']
print(X)
print(y)

lm = LinearRegression()
model = lm.fit(X, y)
print(lm.score(X, y))
