import os
import glob
import numpy as np
import yaml

class AnnotationLoader:
    def __init__(self, annotation_dir, image_width, image_height):
        """
        Loads YOLO format annotations.
        Args:
            annotation_dir (str): Directory containing YOLO annotation files (.txt).
            image_width (int): Width of the images these annotations correspond to.
            image_height (int): Height of the images these annotations correspond to.
        """
        self.annotation_dir = annotation_dir
        self.image_width = image_width
        self.image_height = image_height

    def get_annotation_path(self, image_filename):
        """Gets the annotation file path for a given image filename."""
        base_name = os.path.splitext(image_filename)[0]
        return os.path.join(self.annotation_dir, base_name + ".txt")

    def load_yolo_annotations(self, image_filename):
        """
        Loads YOLO annotations for a single image.
        Returns a list of dictionaries, each with {'class_id', 'x_center', 'y_center', 'width', 'height'}
        where coordinates are normalized (0-1).
        """
        annot_path = self.get_annotation_path(image_filename)
        annotations = []
        if not os.path.exists(annot_path):
            # print(f"Warning: Annotation file not found for {image_filename} at {annot_path}")
            return annotations

        with open(annot_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
                else:
                    print(f"Warning: Malformed line in {annot_path}: {line.strip()}")
        return annotations

    def yolo_to_pixel_bbox(self, yolo_bbox):
        """Converts a single YOLO bbox (normalized) to pixel coordinates [xmin, ymin, xmax, ymax]."""
        x_center_norm = yolo_bbox['x_center']
        y_center_norm = yolo_bbox['y_center']
        width_norm = yolo_bbox['width']
        height_norm = yolo_bbox['height']

        x_center_px = x_center_norm * self.image_width
        y_center_px = y_center_norm * self.image_height
        width_px = width_norm * self.image_width
        height_px = height_norm * self.image_height

        xmin = x_center_px - (width_px / 2)
        ymin = y_center_px - (height_px / 2)
        xmax = x_center_px + (width_px / 2)
        ymax = y_center_px + (height_px / 2)
        
        return [int(xmin), int(ymin), int(xmax), int(ymax)]

    def get_all_image_filenames_with_annotations(self):
        """Returns a list of image base names that have corresponding annotation files."""
        annot_files = glob.glob(os.path.join(self.annotation_dir, "*.txt"))
        # Assuming image files have common extensions like .jpg, .png
        # This part might need adjustment based on how image files are named/stored
        # For simplicity, just return base names from .txt files
        image_basenames = [os.path.splitext(os.path.basename(f))[0] for f in annot_files]
        return image_basenames


if __name__ == '__main__':
    # Create dummy config and annotation files for testing
    dummy_config_path = "temp_dataset_config.yaml"
    dummy_annot_dir = "temp_annots"
    os.makedirs(dummy_annot_dir, exist_ok=True)

    config_data = {
        'annotation_dir': dummy_annot_dir,
        'patch_size': 120, # Example
        'original_image_width': 1920,
        'original_image_height': 1080,
    }
    with open(dummy_config_path, 'w') as f:
        yaml.dump(config_data, f)

    # Create a dummy annotation file
    with open(os.path.join(dummy_annot_dir, "image1.txt"), 'w') as f:
        f.write("0 0.5 0.5 0.1 0.2\n") # class_id, x_center, y_center, width, height
        f.write("0 0.25 0.25 0.05 0.08\n")
    with open(os.path.join(dummy_annot_dir, "image2.txt"), 'w') as f:
        f.write("0 0.7 0.6 0.15 0.15\n")

    # Test AnnotationLoader
    with open(dummy_config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    loader = AnnotationLoader(
        annotation_dir=cfg['annotation_dir'],
        image_width=cfg['original_image_width'],
        image_height=cfg['original_image_height']
    )

    image_files_with_annots = loader.get_all_image_filenames_with_annotations()
    print(f"Image basenames with annotations: {image_files_with_annots}")

    for img_name in image_files_with_annots:
        # Assume image files are named like "image1.jpg"
        yolo_annots = loader.load_yolo_annotations(img_name + ".jpg") 
        print(f"\nAnnotations for {img_name}.jpg (YOLO format):")
        for annot in yolo_annots:
            print(annot)
            pixel_bbox = loader.yolo_to_pixel_bbox(annot)
            print(f"  Pixel bbox [xmin, ymin, xmax, ymax]: {pixel_bbox}")
            print(f"  Pixel width: {pixel_bbox[2]-pixel_bbox[0]}, Pixel height: {pixel_bbox[3]-pixel_bbox[1]}")


    # Clean up dummy files
    os.remove(os.path.join(dummy_annot_dir, "image1.txt"))
    os.remove(os.path.join(dummy_annot_dir, "image2.txt"))
    os.rmdir(dummy_annot_dir)
    os.remove(dummy_config_path)