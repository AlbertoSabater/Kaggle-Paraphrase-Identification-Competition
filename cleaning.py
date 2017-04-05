import numpy as np
import pandas as pd
import re
import chardet

np.random.seed(1337)  # for reproducibility


# Remove quotes on the left and right side
def removeExternalQuotes(s):
    result = re.match(r'^\"(.*)\"$', s)
    if result:
        return s[1:-1]
    else:
        return s


data = pd.read_csv('data/train.csv')

q1 = data[['qid1', 'question1']].drop_duplicates()
q2 = data[['qid2', 'question2']].drop_duplicates()

q1.columns = ['qid', 'question']
q2.columns = ['qid', 'question']

q = pd.concat([q1, q2], ignore_index=True).drop_duplicates()

q.question = q.question.apply(lambda x: str(x))
q.question = q.question.apply(lambda x: x.lower())
q.question = q.question.apply(lambda x: x.strip())
q.question = q.question.apply(lambda x: removeExternalQuotes(x))




#
#def isAscii(s):
#    encoding = chardet.detect(s)
#    if encoding['encoding'] != 'ascii':
#        print s
##    try:
##        s.decode('utf8')
##    except:
##        print s
##        return
#        
#
#aux = q.question.apply(lambda x: isAscii(x))

