import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV files
branch_features = pd.read_csv('/kaggle/input/29-new-tree/branch_features_mean_new.csv')
external_features = pd.read_csv('/kaggle/input/29-new-tree/external_features.csv')
filename_species = pd.read_csv('/kaggle/input/29-new-tree/filename_species.csv')

# Merge the datasets on the 'filename' column
combined_data = pd.merge(branch_features, external_features, on='filename')
combined_data = pd.merge(combined_data, filename_species, on='filename')

# Print available species to verify correct names
print("Available species in the dataset:", combined_data['species'].unique())

# Define the species and their corresponding sample sizes for training and testing
species_sample_sizes = {
    'GUY': {'train': 8, 'test': 2},
    'IND': {'train': 8, 'test': 2},
    'MDD': {'train': 7, 'test': 2}
}

train_frames = []
test_frames = []

# Ensure there are enough samples for each species
for species, sizes in species_sample_sizes.items():
    species_data = combined_data[combined_data['species'] == species]
    if len(species_data) < sizes['train'] + sizes['test']:
        print(f"Not enough samples for {species}: Needed {sizes['train'] + sizes['test']}, Found {len(species_data)}")
        continue
    train_samples = species_data.sample(n=sizes['train'], random_state=42)
    remaining = species_data.drop(train_samples.index)
    test_samples = remaining.sample(n=sizes['test'], random_state=42)
    
    train_frames.append(train_samples)
    test_frames.append(test_samples)

if not train_frames or not test_frames:
    print("Insufficient data to proceed with training/testing. Please check the species names and availability.")
else:
    # Create training and testing datasets
    train_data = pd.concat(train_frames).reset_index(drop=True)
    test_data = pd.concat(test_frames).reset_index(drop=True)

    def prepare_data(data):
        label_encoder = LabelEncoder()
        data['species_label'] = label_encoder.fit_transform(data['species'])
        data = data.drop(columns=['filename', 'species'])
        scaler = StandardScaler()  # Changed to StandardScaler
        features = data.drop(columns=['species_label'])
        scaled_features = scaler.fit_transform(features)
        scaled_data = pd.DataFrame(scaled_features, columns=features.columns)
        scaled_data['species_label'] = data['species_label']
        X = scaled_data.drop(columns=['species_label'])
        y = scaled_data['species_label']
        return X, y, label_encoder

    X_train, y_train, train_label_encoder = prepare_data(train_data)
    X_test, y_test, test_label_encoder = prepare_data(test_data)

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500, activation='relu', solver='adam', random_state=42, alpha=0.001, early_stopping=True)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    cm_test = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=train_label_encoder.classes_, yticklabels=train_label_encoder.classes_, annot_kws={"size": 16})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Test Set)')
    plt.show()

    # Confusion matrix for the whole dataset
    X_total, y_total, _ = prepare_data(combined_data)
    y_pred_total = mlp.predict(X_total)
    cm_total = confusion_matrix(y_total, y_pred_total)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm_total, annot=True, fmt='d', cmap='Blues', xticklabels=train_label_encoder.classes_, yticklabels=train_label_encoder.classes_, annot_kws={"size": 16})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Whole Dataset)')
    plt.show()