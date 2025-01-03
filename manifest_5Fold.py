import os
import csv
from collections import defaultdict

# Set paths for your directories
DS_PATH = '/media/rf/T9/DATABASE/ADS/audio' 
IS_PATH = '/media/rf/T9/DATABASE/IDS/audio'  

# Output directory for CSV files
OUTPUT_DIR = '/media/rf/T9/DATABASE/AV_Clips'


os.makedirs(OUTPUT_DIR, exist_ok=True)


def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


ds_files = list_files(DS_PATH)
is_files = list_files(IS_PATH)


all_files = [(os.path.join(DS_PATH, file), 'ADS') for file in ds_files] + \
            [(os.path.join(IS_PATH, file), 'IDS') for file in is_files]

# Dictionary to hold files per speaker per class
speaker_files_ads = defaultdict(list)
speaker_files_ids = defaultdict(list)

# Group files by speaker ID (first 4 characters of filename) and class
for file, label in all_files:
    speaker_id = file.split('/')[-1][:4]  # Assuming the first 4 characters are speaker ID
    if label == 'ADS':
        speaker_files_ads[speaker_id].append((file, label))
    else:
        speaker_files_ids[speaker_id].append((file, label))

# Ensure files are sorted per speaker for consistency
for speaker_id in speaker_files_ads:
    speaker_files_ads[speaker_id].sort()

for speaker_id in speaker_files_ids:
    speaker_files_ids[speaker_id].sort()

# Function to split data: 15 files for testing, 60 for training for each speaker, per class, repeated for 6 folds
def split_data_per_speaker_and_class(speaker_files_ads, speaker_files_ids, k_folds=6):
    train_test_splits = []

    # KFold splitting based on indices of utterances
    for fold in range(k_folds):
        train_set = []
        test_set = []
        
        # For each speaker, select 15 utterances per class for testing and the rest for training
        for speaker_id in speaker_files_ads.keys():
            # Get files for both ADS and IDS for the speaker
            files_ads = speaker_files_ads[speaker_id]
            files_ids = speaker_files_ids[speaker_id]
            
            # Rotate the test selection in each fold
            start_test_idx_ads = (fold * 15) % 75  # Modulo ensures we wrap around the 75 utterances for ADS
            start_test_idx_ids = (fold * 15) % 75  
            
            # Select 15 files for test from ADS and IDS
            test_files_ads = files_ads[start_test_idx_ads:start_test_idx_ads + 15]  # 15 files from ADS for test
            test_files_ids = files_ids[start_test_idx_ids:start_test_idx_ids + 15]  # 15 files from IDS for test
            
            # Select the remaining 60 files for training
            train_files_ads = [f for i, f in enumerate(files_ads) if i < start_test_idx_ads or i >= start_test_idx_ads + 15]
            train_files_ids = [f for i, f in enumerate(files_ids) if i < start_test_idx_ids or i >= start_test_idx_ids + 15]
            
            # Add the current speaker's files to the overall train/test sets
            test_set.extend(test_files_ads + test_files_ids)  # 30 test files (15 from ADS + 15 from IDS)
            train_set.extend(train_files_ads + train_files_ids)  # 120 train files (60 from ADS + 60 from IDS)

        train_test_splits.append((train_set, test_set))
    
    return train_test_splits

# Function to save train/test splits to CSV
def save_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_path', 'Emotion']) 
        writer.writerows(data)

# Perform the split for each fold (15 utterances per class for test, 60 per class for train)
fold_splits = split_data_per_speaker_and_class(speaker_files_ads, speaker_files_ids)

# Save each fold to CSV files
for fold, (train, test) in enumerate(fold_splits):
    train_filename = os.path.join(OUTPUT_DIR, f'train_fold_{fold+1}.csv')
    test_filename = os.path.join(OUTPUT_DIR, f'test_fold_{fold+1}.csv')
    
    # Save train and test data for the current fold
    save_to_csv(train, train_filename)
    save_to_csv(test, test_filename)
    
    print(f'Fold {fold+1} CSV files saved: train -> {train_filename}, test -> {test_filename}')

