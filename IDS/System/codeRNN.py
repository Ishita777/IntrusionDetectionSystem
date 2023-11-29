# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Load your dataset
# Replace 'your_dataset.csv' with the actual path to your dataset file
data = pd.read_csv('KDDTrain+.csv')

# Remove single quotes from column names
data.columns = data.columns.str.strip("'")

# Encode categorical variables
le = LabelEncoder()
data['protocol_type'] = le.fit_transform(data['protocol_type'])
data['service'] = le.fit_transform(data['service'])
data['flag'] = le.fit_transform(data['flag'])
data['class'] = le.fit_transform(data['class'])

# Separate features and labels
X = data.drop('class', axis=1)
y = data['class']

# Split the dataset into training and testing sets with 40% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for RNN (assuming sequences of length X_train.shape[1])
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Create a simple RNN model
model = Sequential()
model.add(SimpleRNN(128, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
predictions = (model.predict(X_test_reshaped) > 0.5).astype(int)  # Convert probabilities to binary predictions
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

