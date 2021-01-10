# ab-sentiment-analysis
Aspect Based Sentiment Analysis on Anonymous User Reviews

We have a file containing anonymized user reviews. For each review, we are also provided the `aspect` term or the word of interest. Our goal for each review is to predict the polarity (`POSITIVE`, `NEGATIVE` or `NEUTRAL`) of the user's opinion for the given aspect term.

For example, if the <user review, aspect term> is 

    "The soup at this restaurant is awful!", "soup"
    
our task is to predict `NEGATIVE`. It is possible that a single review might have contrasting reviews about different aspects.

For example,

    "The soup at this restaurant is awful, however the service is excellent!", "service"
    
the output for the above user review when aspect term is `service` should be `POSITIVE`.

# Data

Two csv files containing user reviews, aspect term, aspect category and review polarity can be found [here](https://drive.google.com/drive/folders/1Q-z6XCQWnEHNecFNSAmEVt5D4wkr1tDF?usp=sharing). We train on `traindata.csv` and perform validation on `devdata.csv`. The final test set is hidden.

# Solution

Running the python script as below calls the classifier, trains the model and displays results on dev set.

    python testing.py

### Text Preprocessing

  1. Basic preprocessing - converting to lowercase, punctuation and stop-word removal, lemmatization and spell correction.
  2. Some target words are phrases so combine all words in the phrase into a new word. Apply transformation on original sentences.
  3. Use polarity lexicons to assign higher weightage to words with repetitive letters like : 'looooove' or 'gooooood' (indicators of emotion?)

### Feature Generation

  1. Dependency graphs to parse sentences and find modifiers (ADJ/ADV/amod/advmod/attr) of target word. Add pre-trained word embeddings for modifier as feature
  (300 long vector, Google word embeddings)
  2. Split Aspect Category columns into categories and sub-categories and one-hot encode
  3. Look for negations present within 3 preceding words of target modifiers and assign a polarity score using Vader polarity lexicon
  4. Binary variable that is 1 when there is a CAPITAL CASE in the sentence and 0 otherwise.
  5. tf-idf representation of the sentence

### Training

Tried deep vs non-deep models. The non-deep model performed slightly better than the Deep Neural Network model on average.

Ensemble approach - combining predictions from different (weak & strong) classifiers. VotingClassifier over  RandomForest Classifier, PassiveAggressiveClassifier and a Linear SVC.

# Results

Mean accuracy of `82.82%` on dev dataset

