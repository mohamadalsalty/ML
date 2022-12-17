import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier



if __name__ == "__main__":
    params = sys.argv
    
    
    if len(sys.argv) == 4:
        data = pd.read_csv("data.csv")
        x = data[["sport", "drink", "work"]]
        y = data["die"]

        model = RandomForestClassifier()
        model.fit(x.values, y)

        prediction = model.predict([[params[1], params[2], params[3]]])

        if prediction == 0:
            print("This person is going to live.")
        else:
            print("This person is going to die.")
