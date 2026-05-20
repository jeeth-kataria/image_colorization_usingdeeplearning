import os
import argparse
import torch
import torchvision.transforms as transform_lib
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import glob
from skimage.exposure import match_histograms
from datetime import datetime

import lib.TestTransforms as transforms
from models.ColorVidNet import ColorVidNet
from models.FrameColor import frame_colorization
from models.NonlocalNet import VGG19_pytorch, WarpNet
from utils.util import lab2rgb_transpose_mc, tensor_lab2rgb, uncenter_l
from utils.util_distortion import CenterPad, Normalize, RGB2Lab, ToTensor

def smart_color_transfer(source_img, target_img):
    """Adapt source colors to target histogram for natural blending."""
    src_arr = np.array(source_img)
    tgt_arr = np.array(target_img)
    # Match histograms for each channel
    matched = match_histograms(src_arr, tgt_arr, channel_axis=-1)
    return Image.fromarray(matched.astype(np.uint8))

def edge_preserving_smooth(img_np):
    """Clean up color bleeding using edge-aware filtering."""
    # Bilateral filter helps keep edges sharp while smoothing colors
    return cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)

def boost_saturation(img_np, factor=1.5):
    """Enhance the saturation of the final output."""
    img_pil = Image.fromarray(img_np)
    enhancer = ImageEnhance.Color(img_pil)
    return np.array(enhancer.enhance(factor))

def get_smart_descriptor(img_tensor, vggnet, device):
    """Extract a multi-layer semantic descriptor."""
    with torch.no_grad():
        # r32 for texture/style, r52 for high-level content
        feats = vggnet(img_tensor, ["r32", "r52"], preprocess=True)
        # Combine GAP of both layers
        d1 = torch.mean(feats[0], dim=(2, 3))
        d2 = torch.mean(feats[1], dim=(2, 3))
    return torch.cat((d1, d2), dim=1)

def main():
    parser = argparse.ArgumentParser(description="AI Brain Colorize: Deep semantic matching + Advanced color adaptation.")
    parser.add_argument("--target", type=str, required=True, help="Path to the target grayscale image.")
    parser.add_argument("--output", type=str, default=None, help="Path to save the output image.")
    parser.add_argument("--lib", type=str, default="sample_videos/ref", help="Path to the reference library.")
    parser.add_argument("--sat", type=float, default=1.4, help="Saturation boost factor.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    
    args = parser.parse_args()

    # Generate unique filename if none provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_name = os.path.splitext(os.path.basename(args.target))[0]
        args.output = f"{target_name}_auto_{timestamp}.png"

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load networks
    print("Awakening the AI Brain...")
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

    # Preprocessing
    model_size = (216 * 2, 384 * 2)
    transform_to_model = transforms.Compose([RGB2Lab(), ToTensor(), Normalize()])
    feat_size = (224, 224)

    # 1. Analyze target
    target_img_orig = Image.open(args.target).convert('RGB')
    orig_width, orig_height = target_img_orig.size
    
    print("Scanning image for semantic markers...")
    target_feat_img = target_img_orig.resize(feat_size, Image.BILINEAR)
    target_tensor = transform_lib.ToTensor()(target_feat_img).unsqueeze(0).to(device)
    target_desc = get_smart_descriptor(target_tensor, vggnet, device)

    # 2. Search library with Semantic Consensus
    ref_files = []
    for root, dirs, files in os.walk(args.lib):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                ref_files.append(os.path.join(root, file))
    
    if not ref_files:
        print(f"Error: Library '{args.lib}' is empty.")
        return

    best_ref_path = None
    max_sim = -1.0
    
    print(f"Searching library of {len(ref_files)} images for the best match...")
    for ref_path in ref_files:
        try:
            ref_img_tmp = Image.open(ref_path).convert('RGB').resize(feat_size, Image.BILINEAR)
            ref_tensor = transform_lib.ToTensor()(ref_feat_img).unsqueeze(0).to(device)
            ref_desc = get_smart_descriptor(ref_tensor, vggnet, device)
            
            sim = torch.nn.functional.cosine_similarity(target_desc, ref_desc).item()
            if sim > max_sim:
                max_sim = sim
                best_ref_path = ref_path
        except Exception as e:
            # If search fails, ensure we have a fallback
            if best_ref_path is None:
                best_ref_path = ref_path
            continue

    if best_ref_path is None:
        print("Error: Could not process any images in the library.")
        return

    print(f"Found optimal reference: {os.path.basename(best_ref_path)} (Confidence: {max_sim:.2f})")

    # 3. Smart Color Adaptation (The 'Secret Sauce')
    print("Adapting colors to target lighting environment...")
    ref_img_orig = Image.open(best_ref_path).convert('RGB')
    # Use advanced histogram matching to make the colors fit perfectly
    ref_img_adapted = smart_color_transfer(ref_img_orig, target_img_orig)

    # 4. Colorization Refinement
    target_img = target_img_orig.resize((model_size[1], model_size[0]), Image.BILINEAR)
    ref_img = ref_img_adapted.resize((model_size[1], model_size[0]), Image.BILINEAR)

    IB_lab_large = transform_to_model(ref_img).unsqueeze(0).to(device)
    IB_lab = torch.nn.functional.interpolate(IB_lab_large, scale_factor=0.5, mode="bilinear", align_corners=False)
    
    with torch.no_grad():
        I_reference_lab = IB_lab
        I_reference_l = I_reference_lab[:, 0:1, :, :]
        I_reference_ab = I_reference_lab[:, 1:3, :, :]
        I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1))
        features_B = vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
        
    IA_lab_large = transform_to_model(target_img).unsqueeze(0).to(device)
    IA_lab = torch.nn.functional.interpolate(IA_lab_large, scale_factor=0.5, mode="bilinear", align_corners=False)
    IA_l = IA_lab[:, 0:1, :, :]
    I_last_lab_predict = torch.zeros_like(IA_lab).to(device)

    print("Refining colors and fixing textures...")
    with torch.no_grad():
        I_current_ab_predict, _, _ = frame_colorization(
            IA_lab, I_reference_lab, I_last_lab_predict, features_B,
            vggnet, nonlocal_net, colornet, feature_noise=0, temperature=1e-10,
        )

    # 5. Final Polish (Smoothing + Saturation)
    curr_bs_l = IA_lab_large[:, 0:1, :, :]
    curr_predict = torch.nn.functional.interpolate(I_current_ab_predict.data.cpu(), scale_factor=2, mode="bilinear", align_corners=False) * 1.25
    output_img_resized = lab2rgb_transpose_mc(curr_bs_l[0], curr_predict[0])
    
    # Resize back
    output_img = cv2.resize(output_img_resized, (orig_width, orig_height), interpolation=cv2.INTER_LANCZOS4)

    # Smart Polish
    print("Applying final image polish...")
    output_img = edge_preserving_smooth(output_img)
    if args.sat > 1.0:
        output_img = boost_saturation(output_img, args.sat)

    # Save output
    cv2.imwrite(args.output, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    print(f"DONE! Result saved as: {args.output}")

if __name__ == "__main__":
    main()
