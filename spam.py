import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def read(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameDirectory(path, classification):
    rows = []
    index = []
    for filename, message in read(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})

data = data.append(dataFrameDirectory(path, 'spam'))
data = data.append(dataFrameDirectory(path, 'ham'))

vector = CountVectorizer()
count = vector.fit_transform(data['message'].values)

classifier = MultinomialNB()
target = data['class'].values
classifier.fit(count, target)

example = ['Free Viagra!!!']
example1 = vector.transform(example)
predict = classifier.predict(example1)
print predict

example = ["Send me documents today."]
example1 = vector.transform(example)
predict = classifier.predict(example1)
print predict

example = ['Shopping discount']
example1 = vector.transform(example)
predict = classifier.predict(example1)
print predict