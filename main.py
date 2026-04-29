from train import train_and_save_model

from verify_models import verify_model

if __name__ == "__main__":
    csv_files = ["data/merged_PM10_2017_2023.csv", "data/merged_PM25_2017_2023.csv", "data/merged_SO2_2017_2023.csv", "data/merged_NO2_2017_2023.csv", "data/merged_O3_2017_2023.csv", "data/merged_CO_2017_2023.csv"]
    index_name = ["PM10", "PM25", "SO2", "NO2", "O3", "CO"]
    test_files = ["data/test_PM10.csv", "data/test_PM25.csv", "data/test_SO2.csv", "data/test_NO2.csv", "data/test_O3.csv", "data/test_CO.csv"]
    for i in range(len(csv_files)):
        #train_and_save_model(csv_files[i], index_name[i])
        print(f"Verifying {index_name[i]} model...")
        print(f"Test file: {test_files[i]}")
        verify_model(f"models/{index_name[i]}_model.joblib", test_files[i], index_name[i])