import cv2
import os
import numpy as np
from tqdm import tqdm

class PatchExtractor:
    def __init__(self, patch_size, stride=None, padding_mode='constant', constant_values=0):
        """
        Extracts patches from images.
        Args:
            patch_size (int or tuple): Size of the square patch (patch_size, patch_size) or (h, w).
            stride (int or tuple, optional): Stride for patch extraction. If None, defaults to patch_size.
            padding_mode (str): OpenCV border type for padding if patches go out of bounds.
                                e.g., 'constant', 'replicate', 'reflect'.
            constant_values (int or tuple): Value for 'constant' padding.
        """
        if isinstance(patch_size, int):
            self.patch_h, self.patch_w = patch_size, patch_size
        else:
            self.patch_h, self.patch_w = patch_size

        if stride is None:
            self.stride_h, self.stride_w = self.patch_h, self.patch_w
        elif isinstance(stride, int):
            self.stride_h, self.stride_w = stride, stride
        else:
            self.stride_h, self.stride_w = stride
            
        self.padding_mode_map = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
            'reflect': cv2.BORDER_REFLECT,
            'reflect_101': cv2.BORDER_REFLECT_101, # Same as reflect without border doubling
            'wrap': cv2.BORDER_WRAP
        }
        self.cv2_padding_mode = self.padding_mode_map.get(padding_mode, cv2.BORDER_CONSTANT)
        self.constant_values = constant_values


    def extract_patches_from_image(self, image_path, output_dir, base_filename=None):
        """
        Extracts patches from a single image and saves them.
        Args:
            image_path (str): Path to the input image.
            output_dir (str): Directory to save extracted patches.
            base_filename (str, optional): Base name for saved patch files. If None, uses image filename.
        Returns:
            list: List of paths to the saved patches.
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return []

        img_h, img_w = img.shape[:2]
        if base_filename is None:
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        os.makedirs(output_dir, exist_ok=True)
        saved_patch_paths = []

        # Calculate padding needed if patches go out of bounds
        # For simplicity, this version will only extract full patches within image boundaries.
        # A more robust version would pad the image.
        # The paper says: "2800 image patches...each sized 120x120 pixels, extracted from 20 original images."
        # This implies patches are taken from within the image.

        patch_idx = 0
        for y in range(0, img_h - self.patch_h + 1, self.stride_h):
            for x in range(0, img_w - self.patch_w + 1, self.stride_w):
                patch = img[y:y + self.patch_h, x:x + self.patch_w]
                
                patch_filename = f"{base_filename}_patch_{y}_{x}.png" # Or .jpg
                patch_filepath = os.path.join(output_dir, patch_filename)
                
                try:
                    cv2.imwrite(patch_filepath, patch)
                    saved_patch_paths.append(patch_filepath)
                    patch_idx += 1
                except Exception as e:
                    print(f"Error saving patch {patch_filepath}: {e}")
        
        return saved_patch_paths

    def process_directory(self, input_image_dir, output_patch_dir):
        """
        Extracts patches from all images in a directory.
        """
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        all_image_paths = []
        for ext in image_extensions:
            all_image_paths.extend(glob.glob(os.path.join(input_image_dir, ext)))
        
        if not all_image_paths:
            print(f"No images found in {input_image_dir}")
            return

        print(f"Found {len(all_image_paths)} images to process.")
        for img_path in tqdm(all_image_paths, desc="Extracting patches"):
            self.extract_patches_from_image(img_path, output_patch_dir)


if __name__ == '__main__':
    # Create dummy image for testing
    dummy_img_dir = "temp_orig_images"
    dummy_patch_dir = "temp_extracted_patches"
    os.makedirs(dummy_img_dir, exist_ok=True)
    
    dummy_image = np.zeros((300, 400, 3), dtype=np.uint8) # H, W, C
    cv2.putText(dummy_image, "Test Img", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
    dummy_image_path = os.path.join(dummy_img_dir, "test_image1.png")
    cv2.imwrite(dummy_image_path, dummy_image)

    patch_size = 120 # As per paper
    stride = 60 # Example: 50% overlap

    extractor = PatchExtractor(patch_size=patch_size, stride=stride)
    
    # Test single image
    print(f"Extracting patches from single image: {dummy_image_path}")
    saved_paths = extractor.extract_patches_from_image(dummy_image_path, dummy_patch_dir)
    print(f"Saved {len(saved_paths)} patches to {dummy_patch_dir}")
    # for p in saved_paths: print(p)

    # Test directory processing (using the same dummy image dir)
    # print(f"\nExtracting patches from directory: {dummy_img_dir}")
    # extractor.process_directory(dummy_img_dir, dummy_patch_dir + "_from_dir")
    # print("Directory processing finished.")

    # Clean up
    import shutil
    if os.path.exists(dummy_img_dir): shutil.rmtree(dummy_img_dir)
    if os.path.exists(dummy_patch_dir): shutil.rmtree(dummy_patch_dir)
    # if os.path.exists(dummy_patch_dir + "_from_dir"): shutil.rmtree(dummy_patch_dir + "_from_dir")