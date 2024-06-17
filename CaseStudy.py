import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def Load_Data(Path):
    return pd.read_csv(Path)

def Prepare_Data(Data):
    weather_encoder = LabelEncoder()
    temperature_encoder = LabelEncoder()
    play_encoder = LabelEncoder()

    Data['Weather'] = weather_encoder.fit_transform(Data['Weather'])
    Data['Temperature'] = temperature_encoder.fit_transform(Data['Temperature'])
    Data['Play'] = play_encoder.fit_transform(Data['Play'])

    return Data, weather_encoder, temperature_encoder, play_encoder

def Training(X, y, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    return knn

def predict_play(knn, test_data, play_encoder):
    prediction = knn.predict(test_data)
    play_decision = play_encoder.inverse_transform(prediction)
    return play_decision[0]

def check_accuracy(X, y, k=3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    KNN = KNeighborsClassifier(n_neighbors=k)
    KNN.fit(X_train, y_train)
    y_pred = KNN.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def main():
    print("--------------Play Predictor--------------")

    Path = 'PlayPredictor.csv'
    Data = Load_Data(Path)
    Data, weather_encoder, temperature_encoder, play_encoder = Prepare_Data(Data)
    
    Features = Data[['Weather', 'Temperature']]
    Labels = Data['Play']

    KNN = Training(Features, Labels, k=3)
    test_data = [[0, 2]]
    play_decision = predict_play(KNN, test_data, play_encoder)
   
    print('Test Data:', test_data)
    print('Predicted Play Decision:',play_decision)

    for k in range(1, 6):
        accuracy = check_accuracy(Features, Labels, k)
        print('Accuracy for k={k}:',accuracy)

if __name__ == "__main__":
    main()
