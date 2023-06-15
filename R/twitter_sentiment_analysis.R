##Big Data Project 87% Final Mark

# install.packages('pacman')
#Import libraries
library(sparklyr)
library(dplyr)
library(rtweet)
library(ROAuth)
library(rjson)
library(ggplot2)
library(jsonlite)
library(tidyselect)
library(tm)
library(pacman)
library(wordcloud)
library(RTextTools)
library(e1071)
library(caret)
library(syuzhet)
library(stats)
library(datasets)
library(prediction)
library(ROCR)

## Store api keys
api_key <- 'aGkJhTzIJVfVzH26EAT09yvem'
api_secret_key <- '8ewMy5Nq7Q8TJxM53yeWc8AmrW0CL4vCmqnykKhg1FYTMLvfcG'
access_token <- '2430717803-Pv4Py8fBCaT6KXGhgyxNWSxRxrIAEPYwCTONI5h'
access_token_secret <- 'WIZe9t3LWkKz7wpJUesbi0KH9ZemTVx6bXwCUS56od1cY'

#Token rtweet
token <- create_token(
app = "blmresearch",
consumer_key = api_key,
consumer_secret = api_secret_key,
access_token = access_token,
access_secret = access_token_secret
)
get_token()

##Search for up to 10,000 (non-retweeted+retweeted) tweets containing the blm hashtag
df<-search_tweets(
"Spotify,spotify",
n=1000,
timeout= 86400,
lang="en"
)
tweets <- df

#Data exploration
corpus <-iconv(tweets$text, to="utf-8-mac")
corpus<-Corpus(VectorSource(corpus))
corpus <-tm_map(corpus,tolower)
corpus<-tm_map(corpus,removePunctuation)
corpus<-tm_map(corpus,removeNumbers)
cleanset<-tm_map(corpus,removeWords,stopwords('english'))
inspect(cleanset[1:5])
removeURL<-function(x) gsub('http[[:alnum:]]*','', x)
cleanset<-tm_map(cleanset,content_transformer(removeURL))
cleanset<-tm_map(cleanset,removeWords,c('Spotify','spotify'))
cleanset<-tm_map(cleanset,stripWhitespace)
inspect(cleanset[1:5])
tdm<-TermDocumentMatrix(cleanset)
tdm<-as.matrix(tdm)
tdm[1:10,5:10]

#Preprocessing graphs
##barplot
w<-rowSums(tdm)
w<-subset(w,w>=25)
barplot(w,las=2, col=rainbow(50) )

###wordcloud
w<-sort(rosums(tdm),decreasing=TRUE)
set.seed(222)
wordcloud(word=names(w),
freq=w,
max.words=100,
random.order=F,
min.freq=5,
colors=brewer.pal(8,'Dark2'),
scale=c(5,0.3),
rot.per=0.7)
library(wordcloud2)

w<-data.frame(names(w),w)
colnames(w)<-c('word','freq')
wordcloud2(w,
size=0.7,
shape='triangle',
rotateRatio=0.5,
minSize=1)
wordfeqsum<-w[20:30,]
wordfeqsum
w[30:40,] %>%
arrange(freq)

#Sentiment score
stweets <-iconv(tweets$text, to="utf-8-mac")
s<-get_nrc_sentiment(stweets)
name_list<-names(s)
tweets$rating <- as.factor(ifelse(s$positive >= s$negative,"Positive","Negative"))
table(tweets$rating)
barplot(colSums(s),
col=rainbow(10),
ylab='Count',
main="BLM sentiment Score",
names.arg=c(name_list),
beside=TRUE)


#Naive Bayes Model
## Create random samples
set.seed(123)
train_index <- sample(900, 700)
train <- tweets[train_index, ]
test <- tweets[-train_index, ]

##check the proportion of class variable
prop.table(table(train$rating))
train_corpus <- VCorpus(VectorSource(train$text))
test_corpus <- VCorpus(VectorSource(test$text))

##create a document-term sparse matrix directly for train and test
train_dtm <- DocumentTermMatrix(train_corpus, control = list(
tolower = TRUE,
removeNumbers = TRUE,
stopwords = TRUE,
removePunctuation = TRUE,
stemming = TRUE
))

test_dtm <- DocumentTermMatrix(test_corpus, control = list(
tolower = TRUE,
removeNumbers = TRUE,
stopwords = TRUE,
removePunctuation = TRUE,
stemming = TRUE
))
train_dtm

##create function to convert counts to a factor
convert_counts <- function(x) {
x <- ifelse(x > 0, "Yes", "No")
}
##apply() convert_counts() to columns of train/test data
train_dtm_binary <- apply(train_dtm, MARGIN = 2, convert_counts)
test_dtm_binary <- apply(test_dtm, MARGIN = 2, convert_counts)
xTrain<-as.matrix(train_dtm_binary)
yTrain<-train$rating
xTest<-as.matrix(test_dtm_binary)
yTest<-test$rating
model <- naiveBayes(xTrain, yTrain)

##look at confusion matrix
table(predict(model, xTest), yTest)
probs <- predict(model, xTest, type="raw")
qplot(x=probs[, "Positive"], geom="histogram")

##Plot ROC curve
pred <- prediction(probs[, "Positive"], yTest)
perf_nb <- performance(pred, measure='tpr', x.measure='fpr')
plot(perf_nb)

##Print ROC
auc_ROCR <- performance(pred, measure = "auc")
auc_ROCR <- auc_ROCR@y.values[[1]]

##Plot calibration | note overconfidence
calib<-data.frame(predicted=probs[, "Positive"], actual=yTest) %>%
group_by(predicted=round(predicted*10)/10) %>%
summarize(num=n(), actual=mean(actual == "Positive"))
ggplot(data=calib, aes(x=predicted, y=actual, size=num)) +
geom_point() +
geom_abline(a=1, b=0, linetype=2) +
scale_x_continuous(labels=scales::percent, lim=c(0,1)) +
scale_y_continuous(labels=scales::percent, lim=c(0,1))

# install.packages('sparklyr')
# Kmeans clustering
library(factoextra)
library(gridExtra)
library(sparklyr)

#Spark connetion data
# spark_install()
sc <- spark_connect(master = "local")
sspark<-s
sspark$label <- as.numeric(ifelse(s$positive >= s$negative,1,0))
drops <- c("negative","positive")
sspark<-sspark[ , !(names(sspark) %in% drops)]


##Create Euclidean distance
distance <- get_dist(sspark)
fviz_dist(distance, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))

##generate blobs with various clases
k2 <- kmeans(sspark, centers = 2, nstart = 25)
k3 <- kmeans(sspark, centers = 3, nstart = 25)

k4 <- kmeans(sspark, centers = 5, nstart = 25)
k5 <- kmeans(sspark, centers = 10, nstart = 25)
## plots to compare
p1 <- fviz_cluster(k2, geom = "point", data = sspark) + ggtitle("k = 2")
p2 <- fviz_cluster(k3, geom = "point", data = sspark) + ggtitle("k = 3")
p3 <- fviz_cluster(k4, geom = "point", data = sspark) + ggtitle("k = 4")
p4 <- fviz_cluster(k5, geom = "point", data = sspark) + ggtitle("k = 5")
grid.arrange(p1, p2, p3, p4, nrow = 2)
fviz_nbclust(sspark, kmeans, method = "silhouette")



tweets_tbl <- sdf_copy_to(sc, sspark, name = "tweets_tbl", overwrite = TRUE)
partitions <- tweets_tbl %>%
sdf_random_split(training = 0.7, test = 0.3, seed = 1111)
tweets_train <- partitions$training
tweets_test <- partitions$test
nb_model <- tweets_train %>%
ml_naive_bayes(Species ~ .)
pred <- ml_predict(nb_model, tweets_test)
ml<-ml_multiclass_classification_evaluator(pred)
ml
# spark_disconnect(sc)
