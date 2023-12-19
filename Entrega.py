import pandas as pd
import time
from langdetect import detect
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, roc_auc_score
from sklearn.metrics import f1_score, ConfusionMatrixDisplay
import sys

with open('resultats.txt', 'w') as file:
    sys.stdout = file
    
    def count_words(X_train, y_train, remove_words=True):
        words = {}

        for train, target in zip(X_train.to_dict('records'), y_train):
            for word in train['tweetText'].split():
                if word not in words:
                    words[word] = [0, 0]
                words[word][target] += 1

        if remove_words:
            stopW_en = stopwords.words('english')
            for k in stopW_en:
                words.pop(k, None)

        return words


    def resample_data(d, mida_dict):
        sorted_items = sorted(d.items(), key=lambda item: item[1], reverse=True)
        selected_items_count = int(mida_dict * len(sorted_items))
        
        # Seleccionar solo la cantidad requerida de elementos del diccionario ordenado
        df = dict(sorted_items[:selected_items_count])
        return df


    def probs_dict(count_1, count_0, words):
        probs = {}
        for k, v in words.items():
            probs[k] = [v[0] / count_0, v[1] / count_1]

        return probs


    def model_predict(perc0, perc1, X_test, probabilities):
        y_preds = [] 
        for test in X_test.to_dict('records'):
            Prob0, Prob1 = perc0, perc1
            for word in test['tweetText'].split():
                if word in probabilities:
                    Prob0 *= probabilities[word][0]
                    Prob1 *= probabilities[word][1]
            if Prob0 > Prob1:
                y_preds.append(0)
            else:
                y_preds.append(1)
                    
        return y_preds

    def model_predict_laplace(perc0, perc1, X_test, y_train, words, probabilities, alpha):
        y_preds = []

        count_1 = sum(1 for item in y_train if item == 1)
        count_0 = sum(1 for item in y_train if item == 0)

        for test in X_test.to_dict('records'):
            Prob0, Prob1 = perc0, perc1
            for word in test['tweetText'].split():
                if word in probabilities:
                    Prob0 *= (words[word][0] + alpha) / (count_0 + (alpha * 2))
                    Prob1 *= (words[word][1] + alpha) / (count_1 + (alpha * 2))
                else:
                    Prob0 *= (0 + alpha) / (count_0 + (alpha * 2))
                    Prob1 *= (0 + alpha) / (count_1 + (alpha * 2))
            if Prob0 > Prob1:
                y_preds.append(0)
            else:
                y_preds.append(1)
                    
        return y_preds


    def xarxa_bayesiana(X_train, y_train, X_test, mida_dict=1.0, smoothing="None", alpha=1):

        words = count_words(X_train, y_train, True)
        if mida_dict < 1:
            words = resample_data(words, mida_dict)

        param = 0
        
        if smoothing == "laplace":
            param = alpha
            
        count_1 = sum(1 for item in y_train if item == 1) + param
        count_0 = sum(1 for item in y_train if item == 0) + param

        probabilities = probs_dict(count_1, count_0, words)

        prob1 = count_1 / len(y_train) + param * 2
        prob0 = count_0 / len(y_train) + param * 2

        if smoothing == "None":
            y_preds = model_predict(prob0, prob1, X_test, probabilities)
        elif smoothing == "laplace":
            y_preds = model_predict_laplace(prob0, prob1, X_test, X_train, words, probabilities, alpha)

        return y_preds


    def avaluacio_xarxa_k_fold(kfold, X, y, smoothing="None"):
        y_preds = []
        y_tests = []
        i = 0
        inici_kfold = time.time()
        for train_index, test_index in kfold.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            i += 1
            y_pred = xarxa_bayesiana(X_train, y_train, X_test, mida_dict=0.1, smoothing=smoothing)
            acc = accuracy_score(y_pred, y_test)
            f1 = f1_score(y_pred, y_test)
            print("Para el fold ", i, "; accuracy: ", acc, " f1_score: ", f1)
            y_preds.extend(y_pred)
            y_tests.extend(y_test)
        final_kfold = time.time()
        print('\n Mitjana dels resultats \n')
        print(classification_report(y_preds, y_tests))
        disp = ConfusionMatrixDisplay.from_predictions(y_preds, y_tests, normalize='true')
        disp.plot(cmap='Blues', values_format='.4g')
        plt.title('Matriz de Confusión')
        plt.show()
        return final_kfold - inici_kfold


    def apartat1(X_train, X_test, y_train, y_test, smoothing="None"):
        print('1. Ampliem conjunt de train i també augmentem la mida del diccionari \n')
        
        augments = [0.1, 0.3, 0.5, 0.7, 0.8]
        for aug in augments:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-aug, random_state=42)
            print('Test amb ', aug*100, '% train i ', int(100-aug*100), '% test i ', int(aug*100), '% de diccionari \n')
            inici= time.time()
            y_pred = xarxa_bayesiana(X_train, y_train, X_test, mida_dict=aug, smoothing=smoothing, alpha=1)
            final= time.time()
            print(classification_report(y_pred, y_test))
            print("Temps: ", final - inici, '\n')
            
            
    def apartat2(X_train, X_test, y_train, y_test, smoothing="None"):
        print('2. Augmentem la mida del diccionari \n')
        augments = [0.1, 0.5, 0.75, 1]
        for aug in augments:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            print('Test amb 70% train i 30% test i ', int(aug*100), '%de diccionari \n')
            inici= time.time()
            y_pred = xarxa_bayesiana(X_train, y_train, X_test, mida_dict=aug, smoothing="None", alpha=1)
            final= time.time()
            print(classification_report(y_pred, y_test))
            print("Temps: ", final - inici, '\n')
            
        
    def apartat3(X_train, X_test, y_train, y_test, smoothing="None"):
        print('3. Utilitzar sempre la mateixa mida de diccionari, però modificant el conjunt de train \n')
        augments = [0.1, 0.3, 0.5, 0.7, 0.8]
        for aug in augments:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-aug, random_state=42)
            print('Test amb ', int(aug*100), '% train i ', int(100-aug*100), '% test i 50% de diccionari \n')
            inici= time.time()
            y_pred = xarxa_bayesiana(X_train, y_train, X_test, mida_dict=1, smoothing="None", alpha=1)
            final= time.time()
            print(classification_report(y_pred, y_test))
            print("Temps: ", final - inici, '\n')
            

    def exercici1(X_train, X_test, y_train, y_test):
        print('Exercici 1 \n')
        print('Cross validation \n')
        
        ################################### CROSSVALIDATION K FOLD #######################################
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        temps = avaluacio_xarxa_k_fold(kfold, X_train, y_train, smoothing="None")
        
        print("Temps per al cross validation: ", temps, '\n')  
        ###########################################################################################

        print('Test amb 70% train i 30% test \n')
        inici= time.time()
        y_pred = xarxa_bayesiana(X_train, y_train, X_test, smoothing="None", alpha=1)
        final= time.time()
        print(classification_report(y_pred, y_test))
        print("Temps: ", final - inici, '\n')
        disp = ConfusionMatrixDisplay.from_predictions(y_pred, y_test, normalize='true')
        disp.plot(cmap='Blues', values_format='.4g')
        plt.title('Matriz de Confusión')
        plt.show()
        
        
    def exercici2(X_train, X_test, y_train, y_test):
        print('\n Exercici 2 \n')
        
        # APARTAT 2.1
        apartat1(X_train, X_test, y_train, y_test)
        
        # APARTAT 2.2
        apartat2(X_train, X_test, y_train, y_test)
        
        # APARTAT 2.3
        apartat3(X_train, X_test, y_train, y_test)
            
            
    def exercici3(X_train, X_test, y_train, y_test):
        print('Xarxa Bayesiana amb Laplace smoothing \n')
        print('\n Exercici 3 \n')
        print('Cross validation \n')
        
        ################################### CROSSVALIDATION K FOLD #######################################
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        temps = avaluacio_xarxa_k_fold(kfold, X_train, y_train, smoothing="laplace")
        
        print("Temps per al cross validation: ", temps, '\n')
        
        ###########################################################################################
        
        # APARTAT 3.1
        apartat1(X_train, X_test, y_train, y_test, smoothing="laplace")
        
        # APARTAT 3.2
        apartat2(X_train, X_test, y_train, y_test, smoothing="laplace")
        
        # APARTAT 3.3
        apartat3(X_train, X_test, y_train, y_test, smoothing="laplace")
        
        
    if __name__ == '__main__':
        
        df = pd.read_csv('FinalStemmedSentimentAnalysisDataset.csv', delimiter=';')
        df.drop(['tweetDate', 'tweetId'], axis=1, inplace=True)
        df = df[df['tweetText'].notna()]
        X = df.drop('sentimentLabel', axis=1)
        y = df['sentimentLabel']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        print('Xarxa Bayesiana sense Laplace smoothing \n')
        
        ################################### EXERCICI 1 #################################################
        
        exercici1(X_train, X_test, y_train, y_test)
        
        ################################################################################################


        ################################### EXERCICI 2 #################################################
        
        exercici2(X_train, X_test, y_train, y_test)
        
        ################################################################################################
        
        
        ################################### EXERCICI 3 #################################################
        
        exercici3(X_train, X_test, y_train, y_test)
        
        ################################################################################################
        
        
        sys.stdout = sys.__stdout__