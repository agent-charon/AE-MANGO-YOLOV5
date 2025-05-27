import pandas as pd
import os

class CaliperDataLoader:
    def __init__(self, csv_filepath, image_id_col='Image_ID', width_col='Width_cm', height_col='Length_cm'):
        """
        Loads mango size measurements from a CSV file.
        Args:
            csv_filepath (str): Path to the CSV file.
            image_id_col (str): Name of the column containing the image identifier.
            width_col (str): Name of the column for mango width (cm).
            height_col (str): Name of the column for mango height/length (cm).
        """
        if not os.path.exists(csv_filepath):
            raise FileNotFoundError(f"Caliper measurement file not found: {csv_filepath}")
        
        self.df = pd.read_csv(csv_filepath)
        self.image_id_col = image_id_col
        self.width_col = width_col
        self.height_col = height_col # Paper refers to length, width, thickness. Assume length = height for ellipse.

        # Basic validation
        if not all(col in self.df.columns for col in [image_id_col, width_col, height_col]):
            raise ValueError(f"CSV must contain columns: {image_id_col}, {width_col}, {height_col}")

        # The paper mentions 200 mangoes selected for size estimation.
        # Each mango would have an ID that links it to an image or a specific detected bounding box.
        # If one image has multiple mangoes, the ID needs to be more specific (e.g., image1_mango1).

    def get_measurements(self, image_id_or_mango_id):
        """
        Retrieves measurements for a specific image or mango ID.
        Returns a dictionary {'width_cm': val, 'height_cm': val} or None if not found.
        """
        record = self.df[self.df[self.image_id_col] == image_id_or_mango_id]
        if record.empty:
            return None
        # Assuming one entry per ID. If multiple, take the first.
        return {
            'width_cm': record.iloc[0][self.width_col],
            'height_cm': record.iloc[0][self.height_col]
        }

    def get_all_data(self):
        """Returns the entire DataFrame."""
        return self.df
    
    def get_specific_columns(self):
        """Returns only the relevant columns: ID, width, height."""
        return self.df[[self.image_id_col, self.width_col, self.height_col]]

if __name__ == '__main__':
    # Create a dummy CSV for testing
    dummy_csv_path = "temp_caliper_data.csv"
    data = {
        'Image_ID': ['mango_001', 'mango_002', 'mango_003', 'mango_001_obj2'], # Unique ID for each measured mango
        'Width_cm': [8.5, 9.1, 7.8, 8.2],
        'Length_cm': [10.2, 11.0, 9.5, 9.9], # 'Length_cm' used as height_col
        'Some_Other_Data': ['A', 'B', 'C', 'D']
    }
    dummy_df = pd.DataFrame(data)
    dummy_df.to_csv(dummy_csv_path, index=False)

    # Test CaliperDataLoader
    loader = CaliperDataLoader(dummy_csv_path, image_id_col='Image_ID', width_col='Width_cm', height_col='Length_cm')
    
    print("All data:")
    print(loader.get_all_data())

    print("\nRelevant columns:")
    print(loader.get_specific_columns())

    mango_id_to_test = 'mango_002'
    measurements = loader.get_measurements(mango_id_to_test)
    if measurements:
        print(f"\nMeasurements for {mango_id_to_test}:")
        print(f"  Width: {measurements['width_cm']} cm")
        print(f"  Height/Length: {measurements['height_cm']} cm")
    else:
        print(f"\nNo measurements found for {mango_id_to_test}")

    # Clean up dummy file
    os.remove(dummy_csv_path)