import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import collections

############## GLOBAL CONSTANTS AND VARIABLES ################

# Mapping of genres from data to ID tag
GENRE_MAP = {"Animation": 0, "Comedy": 1, "Family": 2, "Adventure": 3, "Fantasy": 4, "Romance": 5, "Drama": 6, 
"Action": 7, "Science Fiction": 8, "Mystery": 9, "War": 10, "Foreign": 11, "Documentary": 12, "Crime": 13, 
"Thriller": 14, "Horror": 15, "History": 16, "Music": 17, "Western": 18, "TV Movie": 19}
GENRE_COUNT = len(GENRE_MAP)

# Pre-processing constants
PRE_FLOATS = [1, 5, 6, 7]
PRE_REVENUE_INDEX = 7
PRE_RATING_INDEX = 6
PRE_GENRE_INDEX = 2
PRE_MONTH_INDEX = 4
PRE_TITLE_INDEX = 3
PRE_COLL_INDEX = 0

# Post-processing constants
POST_BUDGET_INDEX = 0
POST_RUNTIME_INDEX = 1
POST_GENRE_START = 2
POST_MONTH_START = POST_GENRE_START + GENRE_COUNT   
POST_TITLE_INDEX = POST_MONTH_START + 12              
POST_COLL_INDEX = POST_TITLE_INDEX + 1


   
############## LOAD DATA ###############

# Load .txt file into lists for input and output
# ----------------------------------------------
# Returns input vectors (input_vecs) and output vectors 
# separately for revenue (revenue_vecs) and rating (rating_vecs)
def loadFile(x):
    reader = csv.reader(open(x))
    input_vecs, revenue_vecs, rating_vecs = [], [], []
    firstRow = True
    for row in reader:
        if firstRow: firstRow = False
        else:
            new_input, new_output_rev, new_output_rat = [], [], []
            for i in range(len(row)):
                if i in PRE_FLOATS: row[i] = float(row[i])
                if i == PRE_REVENUE_INDEX: 
                    rev = np.log(row[i])
                    if rev == float('-inf'): rev = 0
                    new_output_rev.append(rev)
                elif i == PRE_RATING_INDEX: new_output_rat.append(row[i])
                else: new_input.append(row[i])
            input_vecs.append(new_input)
            revenue_vecs.append(new_output_rev)
            rating_vecs.append(new_output_rat)

    # shuffle data
    combined = list(zip(input_vecs, revenue_vecs, rating_vecs))
    random.shuffle(combined)
    new_input, new_revenue, new_rating = [], [], []
    for x, y, z in combined:
        new_input.append(x)
        new_revenue.append(y)
        new_rating.append(z)

    return new_input, new_revenue, new_rating

# Selects desired variables for prediction
# ----------------------------------------
# Returns updated features per vector (fpervec), modified data as list (vecs)
def selectPredictors(vecs):
    updated_vecs = []
    for i in range(len(vecs)):
        new_vec = []
        counter = 0
        for f in vecs[i]:
            counter += 1
            if isinstance(f, int) or isinstance(f, float): new_vec.append(f)
        # add genre binary features
        genres = parseGenre(vecs[i][PRE_GENRE_INDEX])
        new_vec += genres
        
        # add month binary features
        month = int(vecs[i][PRE_MONTH_INDEX])
        months_vec = [0]*12
        months_vec[month-1] = 1
        new_vec += months_vec
        
        # add title feature
        title_vec = [len(vecs[i][PRE_TITLE_INDEX])]
        new_vec += title_vec
        
        # add feature for collection
        collection = 0
        if len(vecs[i][PRE_COLL_INDEX]) != 0: collection = 1
        coll_vec = [collection]
        new_vec += coll_vec
        
        updated_vecs.append(new_vec)
      
        
    np_vecs = np.array(updated_vecs)
    budget_col = np.log(np_vecs[:,0])
    np_vecs[:,0] = budget_col
    updated_vecs = np_vecs.tolist()

    for vec in updated_vecs:
        if vec[0] == float('-inf'): vec[0] = 0

    return updated_vecs

############## DATA / FEATURE MODIFICIATION ###############

# Normalize budget feature in data samples
# ----------------------------------------
# Returns updated data as list (vecs)
def normalizeBudget(vecs):
    total = 0
    count = 0
    for vec in vecs:
        if vec[0] != 0:
            total += vec[0]
            count += 1
    avg = total/float(count)
    for vec in vecs: 
        if vec[0] == 0: vec[0] = avg
    return vecs

# Add new feature modeling Budget^2
# ---------------------------------
# Returns updated data as list (vecs)
def addBudgetSquaredF(vecs):
    for vec in vecs:
        squared_budget = vec[0]**2
        vec.append(squared_budget)
    return vecs

# Convert budget string to sparse vector
# --------------------------------------
# Returns vector containing genre info
def parseGenre(inputStr):
    if len(inputStr) == 2: return [0]*GENRE_COUNT
    result = [0]*GENRE_COUNT
    inputStr = inputStr[2:-2]
    genreListUnparsed = inputStr.split("}, {")
    for unparsedGenre in genreListUnparsed:
        genreIndex = unparsedGenre.find("'name': ")
        genreStr = unparsedGenre[genreIndex + 8:]
        genreIndex = GENRE_MAP[genreStr[1:-1]]
        result[genreIndex] = 1
    return result
    

############## MODELING / PREDICTION ###############

# Runs linear regression, ridge regression, and lasso given input samples and output samples
# ------------------------------------------------------------------------------------------
# Prints training, validation, and test errors for desired models
def runRegression(x, rating, revenue):

    # create regression models
    linear = linear_model.LinearRegression()
    ridge = linear_model.Ridge()
    lasso = linear_model.Lasso()
    
    while(True):
        # ask user if predicting rating or revenue
        output = raw_input("Type 'rating' to predict rating or 'revenue' to predict revenue: ")
    
        while(True):
            if output != 'rating' and output != 'revenue': 
                output = raw_input("Please try again: ")
            if output == 'rating' or output == 'revenue': break
        if output == 'rating': y = rating
        if output == 'revenue': y = revenue
      
        # create training set
        train_size = int(.8*len(x))
        x_train = x[:train_size]
        y_train = y[:train_size]

        # take user queries for prediction
        regressionQuery(linear, x_train, y_train, output)
    
        # perform 10-fold cross-validations 
        print "Predicting %s:" % output
        num_folds = 10
        splitXTrain = np.array_split(x_train, num_folds)
        splitYTrain = np.array_split(y_train, num_folds)
        x_vals = np.array(splitXTrain)
        y_vals = np.array(splitYTrain)

        validationError(linear, "Linear", num_folds, splitXTrain, splitYTrain, x_train, y_train, x_vals, y_vals)
        validationError(ridge, "Ridge", num_folds, splitXTrain, splitYTrain, x_train, y_train, x_vals, y_vals)
        validationError(lasso, "Lasso", num_folds, splitXTrain, splitYTrain, x_train, y_train, x_vals, y_vals)

        # calculate test errors
        test_size = int(.2*len(x))
        x_test = x[test_size:]
        y_test = y[test_size:]

        testError(linear, "Linear", train_size, x, x_test, y, y_test, output)
        testError(ridge, "Ridge", train_size, x, x_test, y, y_test, output)
        testError(lasso, "Lasso", train_size, x, x_test, y, y_test, output)
        
        repeat = raw_input("Enter 'y' to do another prediction or 'n' to quit: ")
        if repeat == 'n': break
                    
def regressionQuery(model, x_train, y_train, output):
    model.fit(x_train, y_train)
    
    print("Design your own film: choose the qualities of the film and see predictions for rating and revenue!")
    
    # collection variable
    coll = float(raw_input("Enter 1 if movie will be part of collection, or 0 if not: "))
    while(True):
        if coll != 1 and coll != 0:
            coll = float(raw_input("Please try again: "))
        if coll == 1 or coll == 0: break
          
    # budget variable
    budget = raw_input("Enter budget of movie: ")
    while(True):
        if not budget.isdigit():
            budget = raw_input("Please try again: ")
        if budget.isdigit(): 
            budget = np.log(float(budget))
            break
          
    # genre variable
    print '''Available genres: Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, 
    Foreign, History, Horror, Music, Mystery, Romance, Science Fiction, Thriller, TV Movie, War, Western'''
    genre = raw_input("Enter genre(s) of movie, uppercased and separated by a comma and a space: ").split(", ")
    while(True):
        correct = True
        for g in genre:
            if g not in GENRE_MAP.keys(): 
                correct = False
        if correct: break
        genre = raw_input("Please try again: ").split(", ")
    genres = [0]*GENRE_COUNT
    for g in genre: genres[GENRE_MAP[g]] = 1
        
    # title variable
    title = raw_input("Enter movie title: ")
    while(True):
        if type(title) != str: 
            title = raw_input("Please try again: ")
        if type(title): 
            title = len(title)
            break
          
    # runtime variable
    runtime = raw_input("Enter runtime of the movie (in minutes): ")
    while(True):
        if not runtime.isdigit():
            runtime = raw_input("Please try again: ")
        if runtime.isdigit(): break
          
    # release month variable
    month = int(raw_input("Enter month to release the movie (as number): "))
    while(True):
        if month not in range(1,13):
            month = int(raw_input("Please try again: "))
        if month in range(1,13): break
    months = [0]*12
    monthIndex = month - 1
    months[monthIndex] = 1
    
    
    query = []
    query += [budget]
    query += [runtime]
    query += genres
    query += months
    query += [title]
    query += [coll]
    
    query = [float(i) for i in query]
    
    if output == 'rating':
        model.fit(x_train, y_train)
        rating_pred = model.predict([query])[0][0]
        if rating_pred > 10: rating_pred = 10.0
        print "Prediction for rating (out of 10): ", rating_pred
      
    if output == 'revenue':
        model.fit(x_train, y_train)
        revenue_pred = np.exp(model.predict([query])[0][0])
        revenue_pred = "$" + "{:,}".format(revenue_pred)
        print "Prediction for revenue: ", revenue_pred
        

# Prints validation error
# -----------------------
# Takes desired model, number of folds, and training set to calculate
# k-fold cross-validation error
def validationError(model, name, num_folds, splitx, splity, x_train, y_train, x_vals, y_vals):
    errors = [0]*10

    for foldNum in range(num_folds):
        xTrainFold = splitx[:foldNum] + splitx[foldNum + 1:]
        x_train[foldNum] = np.concatenate(xTrainFold)

        yTrainFold = splity[:foldNum] + splity[foldNum + 1:]
        y_train[foldNum] = np.concatenate(yTrainFold)

        model.fit(x_train[foldNum], y_train[foldNum])
        y_pred_test = model.predict(x_vals[foldNum])
        errors[foldNum] = mean_squared_error(y_vals[foldNum], y_pred_test)
        #thresholdRating(y_pred_test_linear, y_vals[foldNum], 1)

    print "%s Validation error: %f" %(name, np.mean(errors))

# Prints test error
# -----------------
# Takes desired model and test set to calculate test error
def testError(model, name, train_size, x, x_test, y, y_test, output):
    model.fit(x[:train_size], y[:train_size])
    y_pred_train = model.predict(x[:train_size])
    y_pred_test = model.predict(x_test)

    print "%s Test error: %f" %(name, mean_squared_error(y_test, y_pred_test))
    print "%s Train error: %f" %(name, mean_squared_error(y[:train_size], y_pred_train))
    if output == "rating": 
        thresholdRating(y_pred_test, y_test, 1.0)
        thresholdRating(y_pred_test, y_test, .5)
    else: thresholdRevenue(y_pred_test, y_test, 0.1)


# Calculate % of rating predictions within threshold
# --------------------------------------------------
# Prints percent of predictions within threshold
def thresholdRating(prediction, test, threshold):
    numSatisfy = 0
    for i, pred in enumerate(prediction):
        truth = test[i]
        if abs(truth - pred) <= threshold:
            numSatisfy += 1
    print "Percent Correct Within +/- %f: %f" %(threshold, numSatisfy/float(len(prediction)))

# Calculate % of revenue predictions within threshold
# ---------------------------------------------------
# Prints percent of predictions within threshold
def thresholdRevenue(prediction, test, threshold):
    numSatisfy = 0
    for i, pred in enumerate(prediction):
        truth = np.exp(test[i])
        pred = np.exp(pred)
        if abs((truth - pred)/float(pred)) <= threshold:
            numSatisfy += 1
    print "Percent Correct Within +/- %f Percent: %f" %(threshold*100, numSatisfy/float(len(prediction)))
    
############# CREATE BAYESIAN NETWORK ################


#Predetermined constants:
NUM_BUDGET_BUCKETS = 10
NUM_RUNTIME_BUCKETS = 10
NUM_RATING_BUCKETS = 10
RATING_BUCKET_WIDTH = 1
NUM_REVENUE_BUCKETS = 10
TITLE_LEN_BUCKET_WIDTH = 5

#Constants calculated based upon number of buckets and range of values in dataset
NUM_TITLE_LEN_BUCKETS = -1
REVENUE_BUCKET_WIDTH = -1
BUDGET_BUCKET_WIDTH = -1
RUNTIME_BUCKET_WIDTH = -1

# Calculate probabilities of features in network
# ----------------------------------------------
# Calculates conditional probabilities to derive probabilities of revenue and rating
# given some choice of film features and prints statistics 
def bayesianNetwork(inputs, revenues, ratings):
  
    getBudgetBucket, getRuntimeBucket, getTitleLenBucket, getRatingBucket, getRevenueBucket = getBucketFns(inputs, revenues)
  
    # probability of a movie having each genre given month of release
    # probability of genre j given release month i (where month 0 is january, 1 is feb) is genreProbabilities[i][j]
    genreProbabilities = getGenreProbabilities(inputs)
    
    #probability of a movie being released in a given month
    releaseMonthProbabilities = getMonthProbabilities(inputs)
    
    # budget conditioned on genre
    budgetProbabilities = getBudgetProbabilities(inputs, getBudgetBucket)
    # Probability of budget in bucket j given genre i is budgetProbabilities[i][j]
    
    # Probability of Runtime given genre
    # probability of runtime in bucket j given genre i is runtimeProbabilities[i][j]
    runtimeProbabilities = getRuntimeProbabilities(inputs, getRuntimeBucket)
    
    # probability of title length
    # probability of title length in bucket i is titleLenProbabilities[i]
    titleLenProbabilities = getTitleLenProbabilities(inputs, getTitleLenBucket)
    
    # probability of in collection (given as scalar value)
    inCollectionProbabilities = getCollectionProbabilities(inputs)
    
    # Probability of rating given genre, budget, and runtime. Also returns revenue given genre, budget, and runtime
            ## # note that these are dictionaries with parents and rating (or revenue) bucket as key, probability as value
    ratingProbabilities, revenueProbabilities = \
            getRatingRevenueProbabilities(ratings, revenues, inputs, getBudgetBucket, getRuntimeBucket, getRatingBucket, getRevenueBucket, getTitleLenBucket)
      
    displayNonMarginals(releaseMonthProbabilities, titleLenProbabilities, inCollectionProbabilities)
    
    marginals = getMarginals(genreProbabilities, releaseMonthProbabilities, budgetProbabilities, runtimeProbabilities, titleLenProbabilities, inCollectionProbabilities, ratingProbabilities, revenueProbabilities)
    ratingGivenGenre, revenueGivenGenre, budgetGivenCollection, budgetGivenGenre = marginals
    
    
    
    
#### How to get value range from buckets
    # each bucket function has associated bucket width, calculated or determined by constant
    # minimum value of a bucket is (index of bucket) * (bucket width)
    # maximum value of a bucket is (index of bucket + 1) * (bucket width) - 1
        #index + 1 gets you start of the next bucket, so subtract 1 to remain within current bucket
      
### Also note that the budget and revenue are logs, so to get the actual range get min and max of bucket
    #  as above, then np.exp(val) to get actual number from log
def getBucketFns(input, revenues):
  
    max_budget = 0
    max_runtime = 0
    max_revenue = 0
    max_titleLen = 0
    for i, vec in enumerate(input):
        budget = vec[POST_BUDGET_INDEX]
        max_budget = max(max_budget, budget)
        
        runtime = vec[POST_RUNTIME_INDEX]
        max_runtime = max(max_runtime, runtime)
        
        revenue = revenues[i][0]
        max_revenue = max(max_revenue, revenue)
        
        title_len = vec[POST_TITLE_INDEX]
        max_titleLen = max(max_titleLen, title_len)
    
    # make sure that max will always fall into bucket, avoid index error
    MAX_LOG_BUDGET = int(max_budget) + 1
    MAX_RUNTIME = int(max_runtime) + 1
    MAX_LOG_REVENUE = int(max_revenue) + 1
    
    def getBudgetBucket(budget):
        #bucketWidth = MAX_LOG_BUDGET//NUM_BUDGET_BUCKETS
        global BUDGET_BUCKET_WIDTH
        BUDGET_BUCKET_WIDTH = max_budget//NUM_BUDGET_BUCKETS + 1
        return int(budget/BUDGET_BUCKET_WIDTH)
      
        
    def getRuntimeBucket(runtime):
        global RUNTIME_BUCKET_WIDTH
        RUNTIME_BUCKET_WIDTH = (max_runtime)//NUM_RUNTIME_BUCKETS + 1
        return int(runtime/RUNTIME_BUCKET_WIDTH)
      
    def getRatingBucket(rating):
        return int(rating/RATING_BUCKET_WIDTH)
      
    def getRevenueBucket(revenue):
        #bucketWidth = MAX_LOG_REVENUE//NUM_REVENUE_BUCKETS
        global REVENUE_BUCKET_WIDTH
        REVENUE_BUCKET_WIDTH = (max_revenue)//NUM_REVENUE_BUCKETS + 1
        
        return int(revenue/REVENUE_BUCKET_WIDTH)
      
    def getTitleLenBucket(title_len):
        return int(title_len/TITLE_LEN_BUCKET_WIDTH)
      
    global NUM_TITLE_LEN_BUCKETS 
    NUM_TITLE_LEN_BUCKETS = int(max_titleLen/TITLE_LEN_BUCKET_WIDTH) + 1
      
    return getBudgetBucket, getRuntimeBucket, getTitleLenBucket, getRatingBucket, getRevenueBucket


def getGenreProbabilities(input):
    
    genreCounts = [[0]*GENRE_COUNT for i in range(12)]
    for vec in input:
        genres = vec[POST_GENRE_START:POST_GENRE_START + GENRE_COUNT]
        months = vec[POST_MONTH_START:POST_MONTH_START + 12]
            
        monthIndex = months.index(1)
        for i, isGenre in enumerate(genres):
            genreCounts[monthIndex][i] += isGenre
            
    totalGenre = sum(sum(genreCounts, []))
    genreProbabilities = [[0]*GENRE_COUNT for i in range(12)]
    for i in range(12):
        for j in range(GENRE_COUNT):
            genreProbabilities[i][j] = genreCounts[i][j]/float(totalGenre)
        
    return genreProbabilities
  
def getMonthProbabilities(inputs):
    monthCounts = [0]*12
    for vec in inputs:
        months = vec[POST_MONTH_START:POST_MONTH_START + 12]
        monthIndex = months.index(1)
        monthCounts[monthIndex] += 1
    
    totalMonths = sum(monthCounts)
    monthProbabilities = [0]*12
    for i in range(12):
        monthProbabilities[i] = monthCounts[i]/float(totalMonths)
        
    return monthProbabilities
            
    
# probability conditioned on genre and on collection (boolean if in a collection)
def getBudgetProbabilities(input, getBudgetBucket):
    
    budgetCounts = [[[0]*NUM_BUDGET_BUCKETS for i in range(GENRE_COUNT)] for j in range(2)]

    for vec in input:
        genres = vec[POST_GENRE_START:POST_GENRE_START + GENRE_COUNT]
        budget = vec[POST_BUDGET_INDEX]
        inCollection = int(vec[POST_COLL_INDEX])
        
        #budgetBucket = int(budget/bucketWidth)
        budgetBucket = getBudgetBucket(budget)
        
        for genreIndex, isGenre in enumerate(genres):
            budgetCounts[inCollection][genreIndex][budgetBucket] += isGenre
            
    totalBudgetCounts = sum(sum(sum(budgetCounts, []),[]))
    budgetProbabilities = [[[0]*NUM_BUDGET_BUCKETS for i in range(GENRE_COUNT)] for j in range(2)]
    for inCollection in range(2):
        for genreIndex in range(GENRE_COUNT):
            for budgetBucket in range(NUM_BUDGET_BUCKETS):
                budgetProbabilities[inCollection][genreIndex][budgetBucket] = \
                            budgetCounts[inCollection][genreIndex][budgetBucket]/float(totalBudgetCounts)
            
    return budgetProbabilities

# probability of runtime conditioned on genre
def getRuntimeProbabilities(input, getRuntimeBucket):
    
    runtimeCounts = [[0]*NUM_RUNTIME_BUCKETS for i in range(GENRE_COUNT)]

    for vec in input:
        genres = vec[POST_GENRE_START:POST_GENRE_START + GENRE_COUNT]
        runtime = vec[POST_RUNTIME_INDEX]
        
        runtimeBucket = getRuntimeBucket(runtime)
        for i, genre in enumerate(genres):
            runtimeCounts[i][runtimeBucket] += genre
            
    totalRuntimeCounts = sum(sum(runtimeCounts, []))
    runtimeProbabilities = [[0]*NUM_RUNTIME_BUCKETS for i in range(GENRE_COUNT)]
    for i in range(GENRE_COUNT):
        for j in range(NUM_RUNTIME_BUCKETS):
            runtimeProbabilities[i][j] = runtimeCounts[i][j]/float(totalRuntimeCounts)
            
    return runtimeProbabilities
  
# probability of length of title
def getTitleLenProbabilities(input, getTitleLenBucket):
    titleLenCounts = [0]*NUM_TITLE_LEN_BUCKETS
    for vec in input:
        titleLen = vec[POST_TITLE_INDEX]
        titleLenBucket = getTitleLenBucket(titleLen)
        titleLenCounts[titleLenBucket] += 1
    
    totalTitleLen = sum(titleLenCounts)
    titleLenProbabilities = [0]*NUM_TITLE_LEN_BUCKETS
    for i in range(NUM_TITLE_LEN_BUCKETS):
        titleLenProbabilities[i] = titleLenCounts[i]/float(totalTitleLen)
        
    return titleLenProbabilities
  
#probability that a film is in a collection
def getCollectionProbabilities(inputs):
    numInCollection = 0
    for vec in inputs:
        inCollection = int(vec[POST_COLL_INDEX])
        numInCollection += inCollection
    
    InCollectionProbability = numInCollection/float(len(inputs)) 
        
    return InCollectionProbability
    
    
# probability of rating conditioned on genre, budget, runtime, title length, in collection
# probability of revenue conditioned on genre, budget, runtime, title length, in collection
def getRatingRevenueProbabilities(ratings, revenues, inputs, getBudgetBucket, getRuntimeBucket, getRatingBucket, getRevenueBucket, getTitleLenBucket):

    
    ratingCounts = collections.defaultdict(int)
    #dictionary mapping (parents, ratingbucket) to the number of ratings in that bucket
        #parents is a tuple of (genres, budget, runtime, title length, in collection) where genres is casted to a tuple
      
    revenueCounts = collections.defaultdict(int)
    #dictionary mapping (parents, ratingbucket) to the number of ratings in that bucket
        #parents is a tuple of (genres, budget, runtime, title length, in collection) where genres is casted to a tuple

    for i, vec in enumerate(inputs):
        genres = vec[POST_GENRE_START:POST_GENRE_START + GENRE_COUNT]
        months = vec[POST_MONTH_START:POST_MONTH_START + 12]
        budget = vec[POST_BUDGET_INDEX]
        runtime = vec[POST_RUNTIME_INDEX]
        titleLen = vec[POST_TITLE_INDEX]
        inCollection = vec[POST_COLL_INDEX]
        
        # rating and revenue are a list with 1 element, so access 0th to take raw value
        rating = ratings[i][0]
        revenue = revenues[i][0]
        
        
        monthIndex = months.index(1)
        budgetBucket = getBudgetBucket(budget)
        runtimeBucket = getRuntimeBucket(runtime)
        ratingBucket = getRatingBucket(rating)
        revenueBucket = getRevenueBucket(revenue)
        titleBucket = getTitleLenBucket(titleLen)

        #using each genre individually instead of the whole set. Have multiple keys for each        
        for genreIndex in range(GENRE_COUNT):
            if genres[genreIndex]:
                parents = (genreIndex, monthIndex, budgetBucket, runtimeBucket, titleBucket, inCollection)
                ratingCounts[(parents, ratingBucket)] += 1
                revenueCounts[(parents, revenueBucket)] += 1
        
    totalRatingCounts = sum(ratingCounts.values())
    ratingProbabilities = collections.defaultdict(float)
    for key, value in ratingCounts.iteritems():
        ratingProbabilities[key] = value/float(totalRatingCounts)
  
    totalRevenueCounts = sum(revenueCounts.values())
    revenueProbabilities = collections.defaultdict(float)
    for key, value in revenueCounts.iteritems():
        revenueProbabilities[key] = value/float(totalRevenueCounts)
            
    return ratingProbabilities, revenueProbabilities
  

def displayNonMarginals(releaseMonthProbabilities, titleLenProbabilities, inCollectionProbabilities):
    print("\n\n")
    print("Displaying probabilities for release in each month")
    for monthIndex in range(12):
        probability = releaseMonthProbabilities[monthIndex]
        print("Probability that month is ", monthIndex + 1, " is ", probability)
        
        
    print("\n\n")
    print("Displaying probabilities for length of the title")
    for titleBucket in range(NUM_TITLE_LEN_BUCKETS):
        minBucketVal = (TITLE_LEN_BUCKET_WIDTH * titleBucket)
        maxBucketVal = (TITLE_LEN_BUCKET_WIDTH * (titleBucket+1))
        probability = titleLenProbabilities[titleBucket]
        print("Probability that title length is between",  minBucketVal, " and ", maxBucketVal, " characters is ", probability)
        
    print("\n\n")
    probCollection = inCollectionProbabilities
    print("Probability that a movie is part of a collection of films is", probCollection*100, "%")
        


def getMarginals(genreProbabilities, releaseMonthProbabilities, budgetProbabilities, runtimeProbabilities, titleLenProbabilities, \
                    inCollectionProbabilities, ratingProbabilities, revenueProbabilities):
  
    print "List of known genres: ", GENRE_MAP
    givenGenre = int(raw_input("Please select genre ID from the above list to see probability of rating and revenue given your choice of genre: "))
    
    while(True):
        if givenGenre not in range(GENRE_COUNT): 
            output = raw_input("Please try again: ")
        if givenGenre in range(GENRE_COUNT): break
    
    # marginal of ratings given genre
    ratingGivenGenre = collections.defaultdict(float)
    for key, probability in ratingProbabilities.iteritems():
        parents, ratingBucket = key
        genreIndex, monthIndex, budgetBucket, runtimeBucket, titleBucket, inCollection = parents
        if genreIndex == givenGenre:
            ratingGivenGenre[ratingBucket] += probability
            
    ratingGivenGenre = normalizeDictProbabilities(ratingGivenGenre)
        
    for ratingBucket in range(NUM_RATING_BUCKETS):
        minBucketVal = RATING_BUCKET_WIDTH * ratingBucket
        maxBucketVal = RATING_BUCKET_WIDTH * (ratingBucket+1)
        print("Probability that rating is from ", minBucketVal, " up to ", maxBucketVal, " is ", ratingGivenGenre[ratingBucket])
        
        
    #marginal of revenue given genre
    revenueGivenGenre = collections.defaultdict(float)
    for key, probability in revenueProbabilities.iteritems():
        parents, revenueBucket = key
        genreIndex, monthIndex, budgetBucket, runtimeBucket, titleBucket, inCollection = parents
        if genreIndex == givenGenre:
            revenueGivenGenre[revenueBucket] += probability
        
    revenueGivenGenre = normalizeDictProbabilities(revenueGivenGenre)
    
    print("\n\n")
    for revenueBucket in range(NUM_REVENUE_BUCKETS):
        minBucketVal = np.exp(REVENUE_BUCKET_WIDTH * revenueBucket)
        maxBucketVal = np.exp(REVENUE_BUCKET_WIDTH * (revenueBucket+1))
        revenueVal = revenueGivenGenre[revenueBucket]
        revenueStr = "$" + "{:,}".format(revenueVal)
        print("Probability that revenue is from ", minBucketVal, " up to ", maxBucketVal, " is " + revenueStr)
       
    
    budgetGivenCollection = displayBudgetGivenCollection(budgetProbabilities, inCollection = 0)
    budgetGivenCollection = displayBudgetGivenCollection(budgetProbabilities, inCollection = 1)
    
    #marginal of budget given genre
    budgetGivenGenre = collections.defaultdict(float)
    for inCollection in range(2):
        for genreIndex in range(GENRE_COUNT):
            if genreIndex == givenGenre:
                for budgetBucket in range(NUM_BUDGET_BUCKETS):
                    probability = budgetProbabilities[inCollection][genreIndex][budgetBucket]
                    budgetGivenGenre[budgetBucket] += probability
                    
                
    budgetGivenGenre = normalizeDictProbabilities(budgetGivenGenre)
    for budgetBucket in range(NUM_BUDGET_BUCKETS):
        minBucketVal = np.exp(BUDGET_BUCKET_WIDTH * budgetBucket)
        maxBucketVal = np.exp(BUDGET_BUCKET_WIDTH * (budgetBucket+1))
        budgetVal = budgetGivenGenre[budgetBucket]
        budgetStr = "$" + "{:,}".format(budgetVal)
        print("Probability that budget is from ", minBucketVal, " up to ", maxBucketVal, " is " + revenueStr)
    
    #marginal of runtime given genre
    print("\n\n")
    print("Displaying the probability of runtime given selected genre")
    runtimeGivenGenre = collections.defaultdict(float)
    for runtimeBucket in range(NUM_RUNTIME_BUCKETS):
        probability = runtimeProbabilities[givenGenre][runtimeBucket]
        runtimeGivenGenre[runtimeBucket] += probability               

    runtimeGivenGenre = normalizeDictProbabilities(runtimeGivenGenre)
    for runtimeBucket in range(NUM_RUNTIME_BUCKETS):
        minBucketVal = (RUNTIME_BUCKET_WIDTH * runtimeBucket)
        maxBucketVal = (RUNTIME_BUCKET_WIDTH * (runtimeBucket+1))
        probability = runtimeGivenGenre[runtimeBucket]
        print("Probability that runtime is from ", minBucketVal, " up to ", maxBucketVal, " is ", probability)
                
    marginals = (ratingGivenGenre, revenueGivenGenre, budgetGivenCollection, budgetGivenGenre)
    return marginals
    
        
                
def normalizeDictProbabilities(probabilities):
    probabilitySum = sum(probabilities.values())
    if probabilitySum == 0:
        return collections.defaultdict(float)
    
    for key, probability in probabilities.iteritems():
        probabilities[key] = probability/float(probabilitySum)
        
    return probabilities
  
def displayBudgetGivenCollection(budgetProbabilities, inCollection):
    budgetGivenCollection = collections.defaultdict(float)
    for genreIndex in range(GENRE_COUNT):
        for budgetBucket in range(NUM_BUDGET_BUCKETS):
            probability = budgetProbabilities[inCollection][genreIndex][budgetBucket]
            budgetGivenCollection[budgetBucket] += probability

    budgetGivenCollection = normalizeDictProbabilities(budgetGivenCollection)
    print("\n\n")
    for budgetBucket in range(NUM_BUDGET_BUCKETS):
        minBucketVal = REVENUE_BUCKET_WIDTH * budgetBucket
        maxBucketVal = REVENUE_BUCKET_WIDTH * (budgetBucket+1)
        introStr = ""
        if inCollection:
            introStr = "Given that the film is in a collection:"
        else:
            introStr = "Given that the film is not in a collection:"
          
        print(introStr)
        budgetVal = budgetGivenCollection[budgetBucket]
        budgetStr = "$" + "{:,}".format(budgetVal)
        print("Probability that budget is from ", minBucketVal, " up to ", maxBucketVal, " is " + budgetStr)
        
    return budgetGivenCollection


################################################################

# Do work
# -------
def main():
    file = 'newdata3.csv'
      
    input_vecs, revenue_vecs, rating_vecs = loadFile(file)
    current_vecs = selectPredictors(input_vecs)

    bayesianNetwork(current_vecs, revenue_vecs, rating_vecs)
    runRegression(current_vecs, rating_vecs, revenue_vecs)

if __name__ == '__main__':
    main()