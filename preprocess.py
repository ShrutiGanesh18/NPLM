
import collections
import os
import zipfile
from urllib2 import urlopen, HTTPError, URLError
from nltk.stem import WordNetLemmatizer
import pickle

def read_file(par_dir,pathname):
	try:
		with zipfile.ZipFile(pathname) as zf:
			file_name = zf.namelist()[0]

		words = []
		count = 0
		with open(par_dir+file_name) as f:
			for line in f:
				for word in line.split():
					words.append(word)
		print len(words)
		return words

	except Exception as e:
		print "Something bad happened! Ooops!" + str(type(e))

def create_dictionary(words,num):
	i = 0
	count = [['UNK', -1]]
	count.extend(collections.Counter(words).most_common(num))
	dictionary = dict()
	c_d = dict()
	for word, _ in count:
		dictionary[word] = len(dictionary)
	c_d['UNK'] = 0
	for n,word in enumerate(words):
		if word not in dictionary:
			words[n]='UNK'
			#print "*"
			c_d['UNK'] = c_d['UNK'] + 1
		else:
			if word not in c_d:
				#print "**"
				c_d[word] = 1
			else:
				#print "***"
				c_d[word] = c_d[word] + 1
	with open("dict_dump",'wb') as fp:
		pickle.dump(dictionary,fp)
	with open("word_dump","wb") as fp:
		pickle.dump(words,fp)
	with open("c_dict_dump","wb") as fp:
		pickle.dump(c_d,fp)

	print i

def parse_file(pathname):
	count = 10000
	words = read_file("data/",pathname)
	dictionary = create_dictionary(words,count)


def set_up_and_preprocess():
    url = "http://mattmahoney.net/dc/text8.zip"
    pathname = "data/text8.zip"

    try:
        if not os.path.exists(pathname):
            with open(pathname,"wb") as f:
                f.write(urlopen(url).read())

        parse_file(pathname)
    except HTTPError as e:
        print "A HTTP error: Code" + str(e.code())
    except URLError as e:
        print "A URL error has occured: Reason: "+ str(e.reason)
    except Exception as e:
        print type(e)



if __name__=="__main__":
    set_up_and_preprocess()
