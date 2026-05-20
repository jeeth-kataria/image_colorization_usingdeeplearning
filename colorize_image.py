import os
import argparse
import torch
import torchvision.transforms as transform_lib
from PIL import Image
import numpy as np
import cv2

import lib.TestTransforms as transforms
from models.ColorVidNet import ColorVidNet
from models.FrameColor import frame_colorization
from models.NonlocalNet import VGG19_pytorch, WarpNet
from utils.util import lab2rgb_transpose_mc, tensor_lab2rgb, uncenter_l
import cv2
from datetime import datetime

import lib.TestTransforms as transforms
...
def main():
    parser = argparse.ArgumentParser(description="Colorize a single image using an exemplar reference image.")
    parser.add_argument("--target", type=str, required=True, help="Path to the target grayscale image.")
    parser.add_argument("--ref", type=str, required=True, help="Path to the reference color image.")
    parser.add_argument("--output", type=str, default=None, help="Path to save the output image.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on (cuda or cpu).")

    args = parser.parse_args()

    # Generate timestamped filename if none provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"output_{timestamp}.png"

    device = torch.device(args.device)
    print(f"Using device: {device}")

    nonlocal_net = WarpNet(1)
    colornet = ColorVidNet(7)
    vggnet = VGG19_pytorch()

    # Load pre-trained weights
    vggnet.load_state_dict(torch.load('data/vgg19_conv.pth', map_location=device))
    nonlocal_net.load_state_dict(torch.load('checkpoints/video_moredata_l1/nonlocal_net_iter_76000.pth', map_location=device))
    colornet.load_state_dict(torch.load('checkpoints/video_moredata_l1/colornet_iter_76000.pth', map_location=device))

    nonlocal_net.eval().to(device)
    colornet.eval().to(device)
    vggnet.eval().to(device)
    
    for param in vggnet.parameters():
        param.requires_grad = False

    # Preprocessing transform
    # We remove CenterPad and CenterCrop because they crop the image (causing the 'zoom' effect)
    # Instead, we will resize directly to the model size and resize back at the end.
    model_size = (216 * 2, 384 * 2) # Height, Width
    
    transform_to_model = transforms.Compose([
        RGB2Lab(),
        ToTensor(),
        Normalize()
    ])

    # Load images
    target_img_orig = Image.open(args.target).convert('RGB')
    orig_width, orig_height = target_img_orig.size
    
    ref_img_orig = Image.open(args.ref).convert('RGB')

    # Resize to model input size (Height, Width)
    target_img = target_img_orig.resize((model_size[1], model_size[0]), Image.BILINEAR)
    ref_img = ref_img_orig.resize((model_size[1], model_size[0]), Image.BILINEAR)

    # Process reference image
    IB_lab_large = transform_to_model(ref_img).unsqueeze(0).to(device)
    IB_lab = torch.nn.functional.interpolate(IB_lab_large, scale_factor=0.5, mode="bilinear", align_corners=False)
    
    with torch.no_grad():
        I_reference_lab = IB_lab
        I_reference_l = I_reference_lab[:, 0:1, :, :]
        I_reference_ab = I_reference_lab[:, 1:3, :, :]
        I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1))
        features_B = vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
        
    # Process target image
    IA_lab_large = transform_to_model(target_img).unsqueeze(0).to(device)
    IA_lab = torch.nn.functional.interpolate(IA_lab_large, scale_factor=0.5, mode="bilinear", align_corners=False)
    IA_l = IA_lab[:, 0:1, :, :]
    
    I_last_lab_predict = torch.zeros_like(IA_lab).to(device)

    # Colorization
    print("Running colorization...")
    with torch.no_grad():
        I_current_ab_predict, _, _ = frame_colorization(
            IA_lab,
            I_reference_lab,
            I_last_lab_predict,
            features_B,
            vggnet,
            nonlocal_net,
            colornet,
            feature_noise=0,
            temperature=1e-10,
        )

    # Upsample and Post-process
    curr_bs_l = IA_lab_large[:, 0:1, :, :]
    curr_predict = torch.nn.functional.interpolate(I_current_ab_predict.data.cpu(), scale_factor=2, mode="bilinear", align_corners=False) * 1.25
    
    output_img_resized = lab2rgb_transpose_mc(curr_bs_l[0], curr_predict[0])
    
    # Resize back to original dimensions
    output_img = cv2.resize(output_img_resized, (orig_width, orig_height), interpolation=cv2.INTER_LANCZOS4)

    # Save output
    cv2.imwrite(args.output, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    print(f"Successfully saved colorized image to: {args.output} (Size: {orig_width}x{orig_height})")

if __name__ == "__main__":
    main()
