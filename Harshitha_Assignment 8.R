#Assignment 8 - LDA 
library(ndjson)
reviews<-ndjson::stream_in("C:/Users/Bharat Rao/Desktop/Musical_Instruments_5.json")
View(reviews)
reviews

library(quanteda)
library(stm)
library(tm)
library(NLP)
library(openNLP)
library(ggplot2)
library(ggdendro)
library(cluster)
library(fpc) 
library(LDAvis)
library(servr)
library(stringr)
library(lda)

require(quanteda)
#using Corpus from quanteda - US President inaugural speeches
summary(reviews)
warnings()
#Generate DFM
corpus<- toLower(reviews$reviewText, keepAcronyms = FALSE) 
cleancorpus <- tokenize(corpus, 
                        removeNumbers=TRUE,  
                        removePunct = TRUE,
                        removeSeparators=TRUE,
                        removeTwitter=FALSE,
                        verbose=TRUE)

dfm<- dfm(cleancorpus,
          toLower = TRUE, 
          ignoredFeatures =stopwords("english"), 
          verbose=TRUE, 
          stem=TRUE)
# Reviewing top features
topfeatures(dfm, 50)     # displays 50 features

#####################
# Hierc. Clustering 
#####################
dfm.tm<-convert(dfm, to="tm")     #move dfm to another package to get sparsity data
dfm.tm
dtmss <- removeSparseTerms(dfm.tm, 0.85)  #0.20 indicates number of "terms" in output to be considered
dtmss
d.dfm <- dist(t(dtmss), method="euclidian") #euclidan is distance
fit <- hclust(d=d.dfm, method="average")
hcd <- as.dendrogram(fit)

require(cluster)
k<-5       #number of clusters
plot(hcd, ylab = "Distance", horiz = FALSE, 
     main = "Five Cluster Dendrogram",        ##how the words in clusters make sense when put together (semantically related)
     edgePar = list(col = 2:3, lwd = 2:2))
rect.hclust(fit, k=k, border=1:5) # draw dendogram with red borders around the 5 clusters

ggdendrogram(fit, rotate = TRUE, size = 4, theme_dendro = FALSE,  color = "blue") +
  xlab("Features") + 
  ggtitle("Cluster Dendrogram")

require(fpc)   
d <- dist(t(dtmss), method="euclidian")   
kfit <- kmeans(d, 5)   
clusplot(as.matrix(d), kfit$cluster, color=T, shade=T, labels=2, lines=0)


#######################################
### Advanced method for Topic Modeling
#######################################


library(dplyr)
require(magrittr)
library(tm)
library(ggplot2)
library(stringr)
library(NLP)
library(openNLP)

#load .csv file with news articles
#url<-"https://raw.githubusercontent.com/jcbonilla/BusinessAnalytics/master/Week-08_and_09/NewsText.csv"
#precorpus<- read.csv(url, 
 #                    header=TRUE, stringsAsFactors=FALSE)

#passing Full Text to variable news_2015
prodrev<-reviews$reviewText


#Cleaning corpus
stop_words <- stopwords("SMART")
## additional junk words showing up in the data
stop_words <- c(stop_words, "said", "the", "also", "say", "just", "like","for", 
                "us", "can", "may", "now", "year", "according", "mr")
stop_words <- tolower(stop_words)


prodrev <- gsub("'", "", prodrev) # remove apostrophes
prodrev <- gsub("[[:punct:]]", " ", prodrev)  # replace punctuation with space
prodrev <- gsub("[[:cntrl:]]", " ", prodrev)  # replace control characters with space
prodrev <- gsub("^[[:space:]]+", "", prodrev) # remove whitespace at beginning of documents
prodrev <- gsub("[[:space:]]+$", "", prodrev) # remove whitespace at end of documents
prodrev <- gsub("[^a-zA-Z -]", " ", prodrev) # allows only letters
prodrev <- tolower(prodrev)  # force to lowercase

## get rid of blank docs
prodrev <- prodrev[prodrev != ""]

# tokenize on space and output as a list:
doc.list <- strsplit(prodrev, "[[:space:]]+")

# compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)


# remove terms that are stop words or occur fewer than 5 times:
del <- names(term.table) %in% stop_words | term.table < 5
term.table <- term.table[!del]
term.table <- term.table[names(term.table) != ""]
vocab <- names(term.table)

# now put the documents into the format required by the lda package:
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)

#############
# Compute some statistics related to the data set:
D <- length(documents)  # number of documents (1)
W <- length(vocab)  # number of terms in the vocab (1741)
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document [312, 288, 170, 436, 291, ...]
N <- sum(doc.length)  # total number of tokens in the data (56196)
term.frequency <- as.integer(term.table) 

# MCMC and model tuning parameters:
K <- 10
G <- 3000   #iterations
alpha <- 0.02   
eta <- 0.02

# Fit the model:
library(lda)
set.seed(357)
t1 <- Sys.time()
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()
## display runtime
t2 - t1  

theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))

reviews_for_LDA <- list(phi = phi,
                     theta = theta,
                     doc.length = doc.length,
                     vocab = vocab,
                     term.frequency = term.frequency)

library(LDAvis)
library(servr)

# create the JSON object to feed the visualization:
json <- createJSON(phi = reviews_for_LDA$phi, 
                   theta = reviews_for_LDA$theta, 
                   doc.length = reviews_for_LDA$doc.length, 
                   vocab = reviews_for_LDA$vocab, 
                   term.frequency = reviews_for_LDA$term.frequency)

serVis(json, out.dir = 'visual', open.browser = TRUE)
getwd(visual)

