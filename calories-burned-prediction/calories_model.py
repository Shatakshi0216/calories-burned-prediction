import csv
import random
def load_data(filename):
    x = []
    y = []
    with open(filename,'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            gender = float(row["Gender"])       
            age = float(row["Age"])
            duration = float(row["Duration"])
            heart_rate = float(row["Heart_Rate"])
            calories = float(row["Calories"])

            x.append([gender, age, duration, heart_rate])
            y.append(calories)
        return x,y    

def initialize_weights(n):
    weights = [random.uniform(-0.01,0.01) for _ in range(n)]
    bias = 0.0
    return weights,bias

def predict(X, weights, bias):
    predictions = []

    for row in X:  # loop through each data point
        y_pred = 0  # start with 0 for each prediction

        # multiply each feature by its weight and add
        for i in range(len(weights)):
            y_pred += weights[i] * row[i]

        y_pred += bias  # finally, add the bias
        predictions.append(y_pred)  # store the prediction

    return predictions

def cost(x,y):
    m = len(y)
    mean_sq = 0
    for i in range (m):
        mean_sq += (x[i]-y[i])**2
    return (1/m)*mean_sq    

def gradient(x,y,weights,bias):
    m = len(y)
    n = len(weights)
    dj_dw = [0.0 for _ in range(n)]
    dj_db = 0.0
    for i in range (m):
        y_pred = sum([weights[j] * x[i][j] for j in range(n)]) + bias
        error = y_pred - y[i]
        for j in range(n):
            dj_dw[j] += (2 / m) * error * x[i][j]
            dj_db += (2 / m) * error
    return dj_dw,dj_db

def train(X, y, weights, bias, learning_rate=0.01, epochs=100):
    cost_history = []

    for epoch in range(epochs):
        # 1. Make predictions
        predictions = predict(X, weights, bias)

        # 2. Calculate gradients
        dj_dw = [0] * len(weights)
        dj_db = 0

        m = len(y)
        for i in range(m):
            error = predictions[i] - y[i]
            for j in range(len(weights)):
                dj_dw[j] += error * X[i][j]
            dj_db += error

        # 3. Average gradients
        for j in range(len(dj_dw)):
            dj_dw[j] = (2/m) * dj_dw[j]
        dj_db = (2/m) * dj_db

        # 4. Update weights and bias
        for j in range(len(weights)):
            weights[j] -= learning_rate * dj_dw[j]
        bias -= learning_rate * dj_db

        # 5. Calculate cost for this epoch
        epoch_cost = cost(predictions, y)
        cost_history.append(epoch_cost)

        # Optional: print every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Cost = {epoch_cost:.4f}")

    return weights, bias, cost_history


x, y = load_data("data.csv")
weights, bias = initialize_weights(len(x[0]))
weights, bias, history = train(x, y, weights, bias)

final_predictions = predict(x, weights, bias)
final_cost = cost(final_predictions, y)
print(f"\nFinal Cost after training: {final_cost:.4f}")

def predict_new_input(new_data, weights, bias):
    y_pred = sum([weights[i] * new_data[i] for i in range(len(weights))]) + bias
    return y_pred

# Example new input: gender, age, duration, heart_rate
new_input = [1, 23, 30.5, 140]  # Replace with real values
prediction = predict_new_input(new_input, weights, bias)
print(f"Calories burned prediction: {prediction:.2f}")
