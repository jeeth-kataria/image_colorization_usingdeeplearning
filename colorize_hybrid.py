import os
import argparse
import torch
import torchvision.transforms as transform_lib
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from datetime import datetime
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

import lib.TestTransforms as transforms
from models.ColorVidNet import ColorVidNet
from models.FrameColor import frame_colorization
from models.NonlocalNet import VGG19_pytorch, WarpNet
from utils.util import lab2rgb_transpose_mc, tensor_lab2rgb, uncenter_l
from utils.util_distortion import Normalize, RGB2Lab, ToTensor

def boost_saturation(img_np, factor=1.2):
    img_pil = Image.fromarray(img_np)
    enhancer = ImageEnhance.Color(img_pil)
    return np.array(enhancer.enhance(factor))

def main():
    parser = argparse.ArgumentParser(description="HYBRID Refinement: Transformer Guess -> Exemplar Precision.")
    parser.add_argument("--target", type=str, required=True, help="Path to target grayscale image.")
    parser.add_argument("--output", type=str, default=None, help="Path to save output.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device.")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_name = os.path.splitext(os.path.basename(args.target))[0]
        args.output = f"{target_name}_hybrid_{timestamp}.png"

    print(f"Using device: {device}")

    # STAGE 1: Transformer Guess (DDColor)
    print("STAGE 1: Generating high-quality color guess using DDColor...")
    colorization_pipeline = pipeline(Tasks.image_colorization, model='damo/cv_ddcolor_image-colorization', device=args.device)
    result = colorization_pipeline(args.target)
    guessed_img_np = result[OutputKeys.OUTPUT_IMG] # BGR
    guessed_img_rgb = cv2.cvtColor(guessed_img_np, cv2.COLOR_BGR2RGB)
    guessed_img_pil = Image.fromarray(guessed_img_rgb)
    
    # STAGE 2: Exemplar Refinement
    print("STAGE 2: Refining colors using Exemplar Engine (Semantic Correspondence)...")
    
    # Load Refinement Networks
    nonlocal_net = WarpNet(1)
    colornet = ColorVidNet(7)
    vggnet = VGG19_pytorch()

    vggnet.load_state_dict(torch.load('data/vgg19_conv.pth', map_location=device))
    nonlocal_net.load_state_dict(torch.load('checkpoints/video_moredata_l1/nonlocal_net_iter_76000.pth', map_location=device))
    colornet.load_state_dict(torch.load('checkpoints/video_moredata_l1/colornet_iter_76000.pth', map_location=device))

    nonlocal_net.eval().to(device)
    colornet.eval().to(device)
    vggnet.eval().to(device)
    
    for param in vggnet.parameters():
        param.requires_grad = False

    # Prep
    model_size = (216 * 2, 384 * 2)
    transform_to_model = transforms.Compose([RGB2Lab(), ToTensor(), Normalize()])
    
    target_img_orig = Image.open(args.target).convert('RGB')
    orig_width, orig_height = target_img_orig.size
    
    # Resize images to model input size
    target_img = target_img_orig.resize((model_size[1], model_size[0]), Image.BILINEAR)
    ref_img = guessed_img_pil.resize((model_size[1], model_size[0]), Image.BILINEAR)

    # Process reference (the guess)
    IB_lab_large = transform_to_model(ref_img).unsqueeze(0).to(device)
    IB_lab = torch.nn.functional.interpolate(IB_lab_large, scale_factor=0.5, mode="bilinear", align_corners=False)
    
    with torch.no_grad():
        I_reference_lab = IB_lab
        I_reference_l = I_reference_lab[:, 0:1, :, :]
        I_reference_ab = I_reference_lab[:, 1:3, :, :]
        I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1))
        features_B = vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
        
    # Process target
    IA_lab_large = transform_to_model(target_img).unsqueeze(0).to(device)
    IA_lab = torch.nn.functional.interpolate(IA_lab_large, scale_factor=0.5, mode="bilinear", align_corners=False)
    IA_l = IA_lab[:, 0:1, :, :]
    I_last_lab_predict = torch.zeros_like(IA_lab).to(device)

    # Final Refinement
    with torch.no_grad():
        I_current_ab_predict, _, _ = frame_colorization(
            IA_lab, I_reference_lab, I_last_lab_predict, features_B,
            vggnet, nonlocal_net, colornet, feature_noise=0, temperature=1e-10,
        )

    # Final Post-process
    curr_bs_l = IA_lab_large[:, 0:1, :, :]
    curr_predict = torch.nn.functional.interpolate(I_current_ab_predict.data.cpu(), scale_factor=2, mode="bilinear", align_corners=False) * 1.25
    output_img_resized = lab2rgb_transpose_mc(curr_bs_l[0], curr_predict[0])
    
    # Back to original size
    output_img = cv2.resize(output_img_resized, (orig_width, orig_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Optional Polish
    output_img = cv2.bilateralFilter(output_img, d=7, sigmaColor=50, sigmaSpace=50)
    output_img = boost_saturation(output_img, 1.2)

    # Save
    cv2.imwrite(args.output, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    print(f"DONE! Hybrid colorization saved to: {args.output}")

if __name__ == "__main__":
    main()
