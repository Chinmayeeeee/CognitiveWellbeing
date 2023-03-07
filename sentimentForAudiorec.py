def test(test_data):
    import pandas as pd
    import joblib
    import re
    import matplotlib.pyplot as plt
    import nltk
    import neattext.functions as nfx
    from neattext.functions import clean_text
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    from nltk import word_tokenize
    import string
    # nltk.download()

    lemmatizer = WordNetLemmatizer()



    def sentiment_analyze(sentiment_prediction):
        neu = 0
        # .count()-> counts the number of occurences of 0 in sentiment_pred
        neg = sentiment_prediction.count(0)
        # .count()-> counts the number of occurences of 1 in sentiment_pred
        pos = sentiment_prediction.count(1)
        # print('Neg:', neg, ', Pos:', pos)
        if neg > pos:
            return ('Feedback: Negative')
            # SpeakText('Feedback is Negative')
        elif pos > neg:
            return('Feedback: Positive')
            # SpeakText('Feedback is Positive')
        else:
            neu += 1
            return('Feedback: Neutral')
            # SpeakText('Feedback is Neutral')
        #     graph in next line
        # feedbackGraph(pos, neu, neg)

    def feedbackGraph(pos, neu, neg):
        fig, ax1 = plt.subplots()
        feedback = ['positive', 'neutral', 'negative']
        ax1.bar(feedback, [pos, neu, neg])
        fig.autofmt_xdate()
        plt.savefig('feedback_graph.png')
        plt.show()


    print('================Loading the Model================')
    model = joblib.load('svm.pkl')
    print('\n*****Ready*****')
    # test_data = open('read', encoding='utf-8').read()
    print("test data: ",test_data)

    # tokenization
    word_list = nltk.word_tokenize(test_data)
    print("tokenized words: ", word_list)

    # negation handling
    def Negation(sentence):
        '''
        Input: Tokenized sentence (List of words)
        Output: Tokenized sentence with negation handled (List of words)
        '''
        temp = int(0)
        for i in range(len(sentence)):
            if sentence[i - 1] in ['not', "n't"]:
                antonyms = []
                for syn in wordnet.synsets(sentence[i]):
                    syns = wordnet.synsets(sentence[i])
                    w1 = syns[0].name()
                    temp = 0
                    for l in syn.lemmas():
                        if l.antonyms():
                            antonyms.append(l.antonyms()[0].name())
                    max_dissimilarity = 0
                    for ant in antonyms:
                        syns = wordnet.synsets(ant)
                        w2 = syns[0].name()
                        syns = wordnet.synsets(sentence[i])
                        w1 = syns[0].name()
                        word1 = wordnet.synset(w1)
                        word2 = wordnet.synset(w2)
                        if isinstance(word1.wup_similarity(word2), float) or isinstance(word1.wup_similarity(word2),
                                                                                        int):
                            temp = 1 - word1.wup_similarity(word2)
                        if temp > max_dissimilarity:
                            max_dissimilarity = temp
                            antonym_max = ant
                            sentence[i] = antonym_max
                            sentence[i - 1] = ''
        while '' in sentence:
            sentence.remove('')
        return sentence

    negated_word_list = Negation(word_list)
    print(negated_word_list)

    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in negated_word_list])
    print("lemmatizedoutput: ", lemmatized_output)

    no_stopword = clean_text(lemmatized_output, puncts=True, stopwords=True)
    print("words without stopwords and punctuations: ", no_stopword)

    # nostopwordslist in list form
    predlist = []
    predlist.append(no_stopword)
    print("final prediction input: ",predlist)

    # prediction
    prediction = model.predict(predlist)
    prediction = prediction.tolist()
    print("prediction: ", prediction)

    output = sentiment_analyze(prediction)
    print("Output: ", output)
    return output


test("hello")