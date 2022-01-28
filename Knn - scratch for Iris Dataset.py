import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def accuracy(y_true, y_pred):
    num_correct = 0
    for i in range(len(y_true)):
        if y_true[i]==y_pred[i]:
            num_correct+=1
    accuracy = num_correct/len(y_true)
    return accuracy

def predict(x_train,x_test,y_train,k):
    distance = []
    predicted= []
    for i in x_test.to_numpy():
        for j in x_train.to_numpy():
            dist = np.sqrt(np.sum((i-j)**2))
            distance.append(dist)
        p = np.argsort(distance)[:k]
        flower_dict ={"Iris-versicolor":0,"Iris-setosa":0,"Iris-virginica":0}
        for h in p:
            flower_dict[y_train.loc[h]]+=1
        distance = []
        predicted.append(max(flower_dict, key=flower_dict.get))
    return predicted

def color_plots(df,title):
    plt.figure()
    versicolor = [[],[]]
    virginica = [[],[]]
    setosa = [[],[]]

    for k in range(len(df["Species"])):
        if(df["Species"].loc[k]=="Iris-versicolor"):
            versicolor[0].append(df["SepalLengthCm"].loc[k])
            versicolor[1].append(df["SepalWidthCm"].loc[k])
        if(df["Species"].loc[k]=="Iris-virginica"):
            virginica[0].append(df["SepalLengthCm"].loc[k])
            virginica[1].append(df["SepalWidthCm"].loc[k])
        if(df["Species"].loc[k]=="Iris-setosa"):
            setosa[0].append(df["SepalLengthCm"].loc[k])
            setosa[1].append(df["SepalWidthCm"].loc[k])
    if(title!="None"):
        plt.scatter(versicolor[0],versicolor[1],c='tab:blue',label="versicolor")     
        plt.scatter(virginica[0],virginica[1],c='tab:red',label="virginica")
        plt.scatter(setosa[0],setosa[1],c='tab:orange',label="setosa")
        plt.xlabel("Sepal Length in cm")
        plt.ylabel("Sepal width in cm")
        plt.title(title)
        plt.legend()
    return versicolor,virginica,setosa

def predicted_plot(predicted,k,a):
    versicolor,virginica,setosa = color_plots(train_dataset,"None")
    wrongly_predicted = [[],[]]
    last_index = 80
    for l,m in zip(predicted,y_test):
        if(l==m):
            if(m=="Iris-virginica"):
                virginica[0].append(x_test["SepalLengthCm"].loc[last_index])
                virginica[1].append(x_test["SepalWidthCm"].loc[last_index])
            if(m=="Iris-setosa"):
                setosa[0].append(x_test["SepalLengthCm"].loc[last_index])
                setosa[1].append(x_test["SepalWidthCm"].loc[last_index])
            if(m=="Iris-versicolor"):
                versicolor[0].append(x_test["SepalLengthCm"].loc[last_index])
                versicolor[1].append(x_test["SepalWidthCm"].loc[last_index])
        else:
            wrongly_predicted[0].append(x_test["SepalLengthCm"].loc[last_index])
            wrongly_predicted[1].append(x_test["SepalWidthCm"].loc[last_index])
        
        last_index+=1
    plt.scatter(versicolor[0],versicolor[1],c='tab:blue',label="versicolor")     
    plt.scatter(virginica[0],virginica[1],c='tab:red',label="virginica")
    plt.scatter(setosa[0],setosa[1],c='tab:orange',label="setosa")
    plt.scatter(wrongly_predicted[0],wrongly_predicted[1],c='black',label="Wrong_Prediction",marker="x",s=30)
    #print(len(versicolor[0])+len(virginica[0])+len(setosa[0])+len(wrongly_predicted[0]))
    plt.title("Predicted | k = {} | Accuracy={}".format(k,a))
    plt.xlabel("Sepal Length in cm")
    plt.ylabel("Sepal width in cm")
    plt.legend()


df = pd.read_csv(r'C:\Users\Naveen\OneDrive\Desktop\Assigns\shuffled_iris.csv')

#df = df.sample(frac=1)
#df = df.drop(["Id"],axis=1)
df = df.reset_index().drop("index",axis=1)
#df.to_csv("shuffled_iris.csv")
train_dataset = df.loc[:80]
test_dataset = df.loc[80:]
x_train = train_dataset[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y_train = train_dataset["Species"]
x_test = test_dataset[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y_test = test_dataset["Species"]
KK = 5
v = predict(x_train,x_test,y_train,KK)
print("Accuarcy:",accuracy(list(y_test),v))

color_plots(df,"All Original Data Points")
color_plots(train_dataset,"Train Dataset")

predicted_plot(v,KK,accuracy(list(y_test),v))

k_values = [3,5,7,9,11,13]
Accuracy = []
for u in k_values:
    v = predict(x_train,x_test,y_train,u)
    Accuracy.append(accuracy(list(y_test),v))
#print(Accuracy)
plt.figure()
plt.xlim([1,14])
plt.ylim([0.92,1.0])
plt.xlabel("k-values")
plt.ylabel("Accuracy")

plt.bar(k_values,Accuracy) 
plt.show()

