import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def Load_Data(Path):
    return pandas.read_csv(Path)

def Prepare_Data(Data):
    weather_encoder = LabelEncoder()
    temperature_encoder = LabelEncoder()
    play_encoder = LabelEncoder()

    Data['Whether'] = weather_encoder.fit_transform(Data['Whether'])
    Data['Temperature'] = temperature_encoder.fit_transform(Data['Temperature'])
    Data['Play'] = play_encoder.fit_transform(Data['Play'])

    return Data, weather_encoder, temperature_encoder, play_encoder

def Training(Features, Labels, k=3):
    KNN = KNeighborsClassifier(n_neighbors=k)
    KNN.fit(Features, Labels)
    return KNN

def predict_play(KNN, test_data, play_encoder):
    prediction = KNN.predict(test_data)
    play_decision = play_encoder.inverse_transform(prediction)
    return play_decision[0]

def check_accuracy(Features, Labels, k=3):
    data_train, data_test, target_train, target_test = train_test_split(Features, Labels, test_size=0.5, random_state=42)
    KNN = KNeighborsClassifier(n_neighbors=k)
    KNN.fit(data_train, target_train)
    Prediction = KNN.predict(data_test)
    accuracy = accuracy_score(target_test, Prediction)
    return accuracy

def main():
    print("--------------Play Predictor--------------")

    Path = 'PlayPredictor.csv'
    Data = Load_Data(Path)
    Data, weather_encoder, temperature_encoder, play_encoder = Prepare_Data(Data)
    
    Features = Data[['Whether', 'Temperature']]
    Labels = Data['Play']

    KNN = Training(Features, Labels, k=3)
    test_data = [[0, 2]]
    play_decision = predict_play(KNN, test_data, play_encoder)
   
    print('Test Data:', test_data)
    print('Predicted Play Decision:',play_decision)

    for k in range(1, 6):

        Accuracy = check_accuracy(Features, Labels, k)
        print(f"Accuracy of classification algorithm with K = {k} : Neighbor classifier is",Accuracy*100,"%")


if __name__ == "__main__":
    main()
