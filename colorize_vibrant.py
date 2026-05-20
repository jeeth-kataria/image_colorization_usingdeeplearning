import os
import argparse
import cv2
import torch
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="VIBRANT Auto-Colorize: Using SOTA DDColor (2023) for deep-learned automatic colorization.")
    parser.add_argument("--target", type=str, required=True, help="Path to the target grayscale image.")
    parser.add_argument("--output", type=str, default=None, help="Path to save the output image.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    
    args = parser.parse_args()
    
    # Generate timestamped filename if none provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"vibrant_{timestamp}.png"

    print(f"Using device: {args.device}")
    print("Initializing VIBRANT AI (DDColor SOTA Model)...")
    
    try:
        # Load the DDColor pipeline from ModelScope
        # This model is specifically designed to prevent 'vague' colors using Dual Decoders
        colorization_pipeline = pipeline(Tasks.image_colorization, model='damo/cv_ddcolor_image-colorization', device=args.device)
        
        print(f"Colorizing {args.target}...")
        result = colorization_pipeline(args.target)
        
        # Extract and save the result
        output_img = result[OutputKeys.OUTPUT_IMG]
        cv2.imwrite(args.output, output_img)
        
        print(f"DONE! Vibrant colorization saved to: {args.output}")
        print("Note: This model uses ICCV 2023 Dual Decoder technology for superior realism.")
        
    except Exception as e:
        print(f"Error during colorization: {e}")
        print("Tip: Make sure you have an internet connection for the first run to download the model weights (approx 500MB).")

if __name__ == "__main__":
    main()
