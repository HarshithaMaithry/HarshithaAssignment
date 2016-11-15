# Harshitha_Business-Analytics-Assignment

The following Assignment was designed to build an interactive LDA environment to interpret the topics for Product Reviews. The product that I have chosen is Amazon Musical Instruments reviews. The following dataset has 10,261 reviews. 

The reviews document was in the form of a .json file. This file was read in RStudio and analyzed by using the below code:

library(ndjson)

reviews<-ndjson::stream_in("C:/Users/Bharat Rao/Desktop/Musical_Instruments_5.json")

View(reviews)

reviews

LDA Analysis: 

LDAvis allows us to create an interactive interface and understand the topic modelling concepts.

# Fit the model:
library(lda)

set.seed(357)

t1 <- Sys.time()

fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab,

                                   num.iterations = G, alpha = alpha, 
                                   
                                   eta = eta, initial = NULL, burnin = 0,
                                   
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()

## Display runtime

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

Visualizing the Fitted Model using LDAvis:

# Create the JSON object to feed the visualization:

json <- createJSON(phi = reviews_for_LDA$phi, 

                   theta = reviews_for_LDA$theta, 
                   
                   doc.length = reviews_for_LDA$doc.length, 
                   
                   vocab = reviews_for_LDA$vocab, 
                   
                   term.frequency = reviews_for_LDA$term.frequency)

serVis(json, out.dir = 'visual', open.browser = TRUE)

The topic modelling for Musical Instruments data has given 10 most relevant topics. When we consider the topic 1, the LDAvis shows the top 30 terms associated with topic 1. The lambda values gives the ranking of terms based on their relevance to each of these topics.

In the example, if we consider topic 1, the most relevant terms are selling, interested, refund, arrive, information, quality,  for a lambda of 0. 

Similarly, the terms for different topics can be sorted using this method for different values of lambda. 
The link for the LDAvis interactive file is given below.

https://htmlpreview.github.io/?https://github.com/HarshithaMaithry/HarshithaAssignment/blob/master/index.html

The "visual" folder comprises of 5 formats of files. A .json, .html, .v3, ldavis and lda files. 
The files have been uploaded onto the github repository.

