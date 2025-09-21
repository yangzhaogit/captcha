#!/usr/bin/env python3

import os
import glob
import subprocess
import pathlib

def test_all_images():
    # Get all jpg files in the input directory
    input_dir = "sampleCaptchas/input"
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    
    results = []
    
    for img_path in image_files:
        # Extract image name
        img_name = pathlib.Path(img_path).name
        
        # Create output file path
        output_file = f"prediction_{img_name.replace('.jpg', '.txt')}"
        
        try:
            # Run captcha solver
            cmd = [
                "python", "captcha_solver.py", "infer",
                "--image", img_path,
                "--model", "captcha_model.npz", 
                "--out", output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Read the prediction
                with open(output_file, 'r') as f:
                    prediction = f.read().strip()
                
                results.append((img_name, prediction))
                print(f"{img_name}: {prediction}")
                
                # Clean up output file
                os.remove(output_file)
            else:
                print(f"Error processing {img_name}: {result.stderr}")
                
        except Exception as e:
            print(f"Exception processing {img_name}: {e}")
    
    return results

if __name__ == "__main__":
    print("Testing all images in sampleCaptchas/input/")
    print("=" * 50)
    results = test_all_images()
    print("=" * 50)
    print(f"Total images processed: {len(results)}")