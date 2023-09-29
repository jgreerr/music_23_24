from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import datasets


PROCESSED_DATASET_PATH = "/cs/home/stu/greer2jl/Documents/teleband-export/teleband-export/teleband_dataset_mp3"



# Idea of data: {RhythmScore, Cost Matrix} -> {1, 2, 3, 4, 5}

def get_cost_matrix(key, title):
    return title

def preprocess_data(data):
    new_data = pd.DataFrame(data)
    new_data = new_data.drop(columns=['tone', 'expression', 'grade'])
    new_data["cost_matrix"] = get_cost_matrix(new_data["key"] , new_data["title"])
    print(new_data)
    pass

def main():
    # Load in full dataset
    dataset = datasets.load_from_disk(PROCESSED_DATASET_PATH)
    dataset = preprocess_data(dataset)

    rf = RandomForestClassifier()
    # rf.fit(X_train, ["rhythm"])

main()