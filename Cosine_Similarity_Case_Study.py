# %% markdown
# ## Cosine Similarity Calculations
# Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them. Similarity measures have a multiude of uses in machine learning projects; they come in handy when matching strings, measuring distance, and extracting features. This similarity measurement is particularly concerned with orientation, rather than magnitude.
# In this case study, you'll use the cosine similarity to compare both a numeric data within a plane and a text dataset for string matching.
# %% markdown
# Load the Python modules, including cosine_similarity, from sklearn.metrics.pairwise
# %% codecell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
%matplotlib inline
plt.style.use('ggplot')
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
# %% markdown
# **<font color='teal'> Load the distance dataset into a dataframe. </font>**
# %% codecell
cd_data = 'data/'
df = pd.read_csv(cd_data+'distance_dataset.csv')
# %% markdown
# ### Cosine Similarity with clusters and numeric matrices
#
# All points in our dataset can be thought of as feature vectors. We illustrate it here as we display the __Cosine Similarity__ between each feature vector in the YZ plane and the [5, 5] vector we chose as reference. The sklearn.metrics.pairwise module provides an efficient way to compute the __cosine_similarity__ for large arrays from which we can compute the similarity.
# %% markdown
#  **<font color='teal'> First, create a 2D and a 3D matrix from the dataframe. The 2D matrix should contain the 'Y' and 'Z' columns and the 3D matrix should contain the 'X','Y', and 'Z' columns.</font>**
# %% codecell
yz2d = df[['Y', 'Z']]
xyz3d = df[['X', 'Y', 'Z']]

# %% markdown
# Calculate the cosine similarity for those matrices with reference planes of 5,5 and 5,5,5. Then subtract those measures from 1 in new features.
# %% codecell
simCosine3D = 1. - cosine_similarity(xyz3d, [[5,5,5]], 'cosine')
simCosine = 1. - cosine_similarity(yz2d, [[5,5]], 'cosine')
# %% markdown
# Using the 2D matrix and the reference plane of (5,5) we can use a scatter plot to view the way the similarity is calculated using the Cosine angle.
# %% codecell
figCosine = plt.figure(figsize=[10,8])

plt.scatter(df.Y, df.Z, c=simCosine[:,0], s=20)
plt.plot([0,5],[0,5], '--', color='dimgray')
plt.plot([0,3],[0,7.2], '--', color='dimgray')
plt.text(0.7,2.6,r'$\theta$ = 22.4 deg.', rotation=47, size=14)
plt.ylim([0,10])
plt.xlim([0,10])
plt.xlabel('Y', size=14)
plt.ylabel('Z', size=14)
plt.title('Cosine Similarity')
cb = plt.colorbar()
cb.set_label('Similarity with (5,5)', size=14)

figCosine.savefig('figures/similarity-cosine.png')
# %% markdown
# Now, plot the 3D matrix with the similarity and the reference plane, (5,5,5).
# %% codecell
from mpl_toolkits.mplot3d import Axes3D
figCosine3D = plt.figure(figsize=(10, 8))
ax = figCosine3D.add_subplot(111, projection='3d')

p = ax.scatter(np.array(xyz3d)[:,0], np.array(xyz3d)[:,1], np.array(xyz3d)[:,2], c=simCosine3D[:,0])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
cb = figCosine3D.colorbar(p)
cb.set_label('Similarity with (5,5,5)', size=14)

figCosine3D.tight_layout()
figCosine3D.savefig('figures/cosine-3D.png', dpi=300, transparent=True)
# %% markdown
# ----
# %% markdown
# ### Cosine Similarity with text data
# This is a quick example of how you can use Cosine Similarity to compare different text values or names for record matching or other natural language proecessing needs.
# First, we use count vectorizer to create a vector for each unique word in our Document 0 and Document 1.
# %% codecell
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
Document1 = "Starbucks Coffee"
Document2 = "Essence of Coffee"

corpus = [Document1,Document2]

X_train_counts = count_vect.fit_transform(corpus)

pd.DataFrame(X_train_counts.toarray(),columns=count_vect.get_feature_names(),index=['Document 0','Document 1'])
# %% markdown
# Now, we use a common frequency tool called TF-IDF to convert the vectors to unique measures.
# %% codecell
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
trsfm=vectorizer.fit_transform(corpus)
pd.DataFrame(trsfm.toarray(),columns=vectorizer.get_feature_names(),index=['Document 0','Document 1'])
# %% markdown
# Here, we finally apply the __Cosine Similarity__ measure to calculate how similar Document 0 is compared to any other document in the corpus. Therefore, the first value of 1 is showing that the Document 0 is 100% similar to Document 0 and 0.26055576 is the similarity measure between Document 0 and Document 1.
# %% codecell
cosine_similarity(trsfm[0:1], trsfm)
# %% markdown
# Replace the current values for `Document 0` and `Document 1` with your own sentence or paragraph and apply the same steps as we did in the above example.
# %% markdown
#  **<font color='teal'> Combine the documents into a corpus.</font>**
# %% codecell
count_vect = CountVectorizer()
Document1 = """
Thou shalt have no other gods before me
Thou shalt not make unto thee any graven image
Thou shalt not take the name of the Lord thy God in vain
Remember the sabbath day, to keep it holy
Honour thy father and thy mother
Thou shalt not kill
Thou shalt not commit adultery
Thou shalt not steal
Thou shalt not bear false witness against thy neighbour
Thou shalt not covet
"""
Document2 = """
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
"""

corpus = [Document1,Document2]


# %% markdown
#  **<font color='teal'> Apply the count vectorizer to the corpus to transform it into vectors.</font>**
# %% codecell
X_train_counts = count_vect.fit_transform(corpus)

# %% markdown
#  **<font color='teal'> Convert the vector counts to a dataframe with Pandas.</font>**
# %% codecell
pd.DataFrame(X_train_counts.toarray(),columns=count_vect.get_feature_names(),index=['Document 0','Document 1'])

# %% markdown
#  **<font color='teal'> Apply TF-IDF to convert the vectors to unique frequency measures.</font>**
# %% codecell
vectorizer = TfidfVectorizer()
trsfm=vectorizer.fit_transform(corpus)
pd.DataFrame(trsfm.toarray(),columns=vectorizer.get_feature_names(),index=['Document 0','Document 1'])
# %% markdown
# %% markdown
#  **<font color='teal'> Use the cosine similarity function to get measures of similarity for the sentences or paragraphs in your original document.</font>**
# %% codecell
cosine_similarity(trsfm[0:1], trsfm)
# %% markdown
# As it turns out, the 10 comandments are not much like the Zen of Python.
