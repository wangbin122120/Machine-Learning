import W_bayes
import feedparser

ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
vocabList, pSF, pNY = W_bayes.localWords(ny, sf)
vocabList, pSF, pNY = W_bayes.localWords(ny, sf)
