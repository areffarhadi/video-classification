import os
import csv
import random

def create_manifest(directory, output_csv_train, output_csv_test):
    # Dictionary to store files by emotion
    files_by_emotion = {}

    # Traverse the directory recursively
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp4'):
                # Full path of the file
                file_path = os.path.join(root, file)

                # Emotion is the name of the parent folder
                emotion = os.path.basename(root)

                # Add file to the corresponding emotion category
                if emotion not in files_by_emotion:
                    files_by_emotion[emotion] = []
                files_by_emotion[emotion].append(file_path)

    # Separate files into train and test splits
    train_data = []
    test_data = []

    for emotion, file_paths in files_by_emotion.items():
        # Shuffle files randomly
        random.shuffle(file_paths)

        # Split files into 90% train and 10% test
        split_index = int(len(file_paths) * 0.9)
        train_files = file_paths[:split_index]
        test_files = file_paths[split_index:]

        # Add to respective datasets
        train_data.extend([[file_path, emotion] for file_path in train_files])
        test_data.extend([[file_path, emotion] for file_path in test_files])

    # Write train data to CSV
    with open(output_csv_train, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_path', 'Emotion'])  
        writer.writerows(train_data)  

    # Write test data to CSV
    with open(output_csv_test, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_path', 'Emotion'])  
        writer.writerows(test_data)  

    print(f"Train manifest file created: {output_csv_train}")
    print(f"Test manifest file created: {output_csv_test}")


directory = "/home/user/aref/AV_data"  
output_csv_train = "train.csv"  
output_csv_test = "test.csv"  
create_manifest(directory, output_csv_train, output_csv_test)

