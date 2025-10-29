import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import zipfile
from pathlib import Path
import os
import json
from pillow_heif import register_heif_opener

# Register HEIF opener for PIL
register_heif_opener()

# === Password Configuration ===
# Get password from Streamlit secrets
CORRECT_PASSWORD = st.secrets["password"]

# === Watermark Configuration ===
WATERMARK_PATH = "badge.png"
BRAND_LOGOS_JSON = "brand-logos/brands.json"

# === Image Processing Functions ===
def remove_background(img):
    """
    Remove white/grey background, crop to square with margins, and apply adjustments.
    Preserves product details while removing neutral backgrounds.
    Returns processed image as numpy array.
    """
    # === Pre-processing: Enhance contrast to separate product from background ===
    img_preprocessed = img.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve contrast
    lab = cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_preprocessed = cv2.merge([l, a, b])
    img_preprocessed = cv2.cvtColor(img_preprocessed, cv2.COLOR_LAB2BGR)
    
    # === Convert to multiple color spaces ===
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use preprocessed image for better edge detection
    gray_preprocessed = cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2GRAY)
    
    # === Strategy 1: Detect neutral gray background (VERY AGGRESSIVE) ===
    # Cast a wide net for background detection
    a_centered = np.abs(a_channel.astype(np.int16) - 128) < 25
    b_centered = np.abs(b_channel.astype(np.int16) - 128) < 25
    low_saturation = s < 28
    very_light = v > 175
    neutral_gray = a_centered & b_centered & low_saturation & very_light
    not_background_mask = (~neutral_gray).astype(np.uint8) * 255
    
    # === Strategy 2: Preserve anything with color ===
    _, sat_mask = cv2.threshold(s, 5, 255, cv2.THRESH_BINARY)
    
    # === Strategy 3: Preserve darker objects ===
    _, dark_mask = cv2.threshold(v, 190, 255, cv2.THRESH_BINARY_INV)
    
    # === Strategy 4: Enhanced edge detection using preprocessed image ===
    edges = cv2.Canny(gray_preprocessed, 30, 100)
    edges_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    
    # === Strategy 5: Texture detection ===
    blur = cv2.GaussianBlur(l_channel, (21, 21), 0)
    texture = cv2.absdiff(l_channel, blur)
    _, texture_mask = cv2.threshold(texture, 4, 255, cv2.THRESH_BINARY)
    
    # === Combine masks - anything NOT neutral gray OR has features ===
    combined_mask = cv2.bitwise_or(not_background_mask, sat_mask)
    combined_mask = cv2.bitwise_or(combined_mask, dark_mask)
    combined_mask = cv2.bitwise_or(combined_mask, texture_mask)
    combined_mask = cv2.bitwise_or(combined_mask, edges_dilated)
    
    # === Morphological operations to clean mask (MORE CONSERVATIVE) ===
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    kernel_large = np.ones((9, 9), np.uint8)
    
    # Close small gaps in product
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    # Remove very small noise only
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    # Fill larger holes within product
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    # Slightly dilate to ensure we capture all product edges
    combined_mask = cv2.dilate(combined_mask, kernel_small, iterations=1)
    
    # === Find contours ===
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No objects detected in image")
    
    # === Filter small contours ===
    min_contour_area = 800
    contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    if not contours:
        raise ValueError("No significant objects detected in image")
    
    # === Find the largest contour (main product) ===
    main_contour = max(contours, key=cv2.contourArea)
    main_area = cv2.contourArea(main_contour)
    
    # === Keep contours that are significant or close to main contour ===
    final_contours = [main_contour]
    for contour in contours:
        if contour is main_contour:
            continue
        if cv2.contourArea(contour) > main_area * 0.02:
            final_contours.append(contour)
    
    # === Build final mask ===
    final_mask = np.zeros_like(combined_mask)
    for contour in final_contours:
        cv2.drawContours(final_mask, [contour], -1, 255, -1)
    
    # === Dilate mask MORE to preserve product edges ===
    final_mask = cv2.dilate(final_mask, kernel_small, iterations=3)
    
    # === Refine: Remove background aggressively ===
    # Target light neutral pixels more aggressively
    very_light_bg = gray > 185
    very_neutral_a = np.abs(a_channel.astype(np.int16) - 128) < 22
    very_neutral_b = np.abs(b_channel.astype(np.int16) - 128) < 22
    very_low_sat = s < 22
    strict_bg = very_light_bg & very_neutral_a & very_neutral_b & very_low_sat
    
    # Remove strict background from mask
    final_mask = cv2.bitwise_and(final_mask, (~strict_bg).astype(np.uint8) * 255)
    
    # === Get bounding box from CLEANED mask ===
    # Recalculate contours from cleaned mask for accurate bounding box
    clean_contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not clean_contours:
        # Fallback to original contours if cleaning removed everything
        all_points = np.vstack(final_contours)
    else:
        all_points = np.vstack(clean_contours)
    
    x, y, w, h = cv2.boundingRect(all_points)
    
    # Add padding (more generous to capture full product)
    padding = 15
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)
    
    # === Calculate square crop with very minimal margin (makes products largest possible) ===
    margin_ratio = 0.02  # Reduced from 0.03 to make products even larger
    object_ratio = 1 - 2 * margin_ratio
    max_dim = max(w, h)
    square_size = int(max_dim / object_ratio)
    
    # Use simple bounding box center for consistent centering
    # This works better than center of mass for irregularly shaped products
    center_x = x + w // 2
    center_y = y + h // 2
    
    crop_x1 = center_x - square_size // 2
    crop_y1 = center_y - square_size // 2
    crop_x2 = crop_x1 + square_size
    crop_y2 = crop_y1 + square_size
    
    # === Create white canvas ===
    canvas = np.ones((square_size, square_size, 3), dtype=np.uint8) * 255
    
    # === Adjust crop boundaries ===
    paste_x = max(0, -crop_x1)
    paste_y = max(0, -crop_y1)
    src_x1 = max(0, crop_x1)
    src_y1 = max(0, crop_y1)
    src_x2 = min(crop_x2, img.shape[1])
    src_y2 = min(crop_y2, img.shape[0])
    
    src_w = src_x2 - src_x1
    src_h = src_y2 - src_y1
    
    # === Extract regions ===
    source_region = img_rgb[src_y1:src_y2, src_x1:src_x2]
    mask_region = final_mask[src_y1:src_y2, src_x1:src_x2]
    
    # === Apply mask with MORE blur for smoother edges ===
    mask_blurred = cv2.GaussianBlur(mask_region, (9, 9), 0)
    mask_3channel = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2RGB) / 255.0
    
    # Blend with white background
    masked_source = (source_region * mask_3channel).astype(np.uint8)
    white_bg = np.ones_like(source_region) * 255
    final_region = (masked_source + white_bg * (1 - mask_3channel)).astype(np.uint8)
    
    # === Final cleanup: Aggressive background removal with strong product protection ===
    final_gray = cv2.cvtColor(final_region, cv2.COLOR_RGB2GRAY)
    final_hsv = cv2.cvtColor(final_region, cv2.COLOR_RGB2HSV)
    final_h, final_s, final_v = cv2.split(final_hsv)
    final_lab = cv2.cvtColor(final_region, cv2.COLOR_RGB2LAB)
    final_l, final_a, final_b = cv2.split(final_lab)
    
    # Create a strong product protection zone
    product_mask = mask_region > 30  # Very sensitive to include everything
    # Dilate significantly to protect product interior
    product_protected = cv2.dilate(product_mask.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=3)
    # Small erosion to allow edge cleanup only
    product_core = cv2.erode(product_protected, np.ones((3, 3), np.uint8), iterations=1)
    
    # Target 1: Very light, very neutral pixels
    nearly_white = (final_gray > 210) & \
                   (np.abs(final_a.astype(np.int16) - 128) < 15) & \
                   (np.abs(final_b.astype(np.int16) - 128) < 15) & \
                   (final_s < 15)
    
    # Target 2: Light gray background - VERY AGGRESSIVE
    light_gray_bg = (final_gray > 165) & \
                    (final_gray < 245) & \
                    (final_s < 25) & \
                    (np.abs(final_a.astype(np.int16) - 128) < 22) & \
                    (np.abs(final_b.astype(np.int16) - 128) < 22)
    
    # Only remove if NOT in protected product core
    to_whiten = (nearly_white | light_gray_bg) & (product_core == 0)
    final_region[to_whiten] = [255, 255, 255]
    
    # === Paste onto canvas ===
    canvas[paste_y:paste_y + src_h, paste_x:paste_x + src_w] = final_region
    
    # Convert to array for return (will apply adjustments later with sliders)
    return canvas


def has_transparency(pil_image):
    """
    Check if a PIL image has meaningful transparency (alpha channel with non-255 values).
    Returns True if image has transparency, False otherwise.
    """
    if pil_image.mode not in ('RGBA', 'LA', 'P'):
        return False
    
    # Convert P mode to RGBA to check alpha
    if pil_image.mode == 'P':
        if 'transparency' in pil_image.info:
            return True
        pil_image = pil_image.convert('RGBA')
    
    # Check if alpha channel has any non-255 values
    if pil_image.mode in ('RGBA', 'LA'):
        alpha = np.array(pil_image.split()[-1])
        # If any pixel has alpha < 250, consider it transparent
        return np.any(alpha < 250)
    
    return False


def process_transparent_image(img_rgb, margin_ratio=0.02):
    """
    Process an image that already has transparency.
    Creates square bounding box around non-transparent content and applies margins.
    Returns processed image as numpy array.
    """
    # Convert to RGBA if not already
    if len(img_rgb.shape) == 2:
        img_rgba = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGBA)
    elif img_rgb.shape[2] == 3:
        img_rgba = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2RGBA)
    else:
        img_rgba = img_rgb.copy()
    
    # Extract alpha channel
    alpha = img_rgba[:, :, 3]
    
    # Find bounding box of non-transparent pixels
    non_transparent = alpha > 10  # Threshold for near-transparent
    coords = np.argwhere(non_transparent)
    
    if len(coords) == 0:
        raise ValueError("Image appears to be completely transparent")
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Add padding
    padding = 15
    y_min = max(0, y_min - padding)
    x_min = max(0, x_min - padding)
    y_max = min(img_rgba.shape[0], y_max + padding)
    x_max = min(img_rgba.shape[1], x_max + padding)
    
    w = x_max - x_min
    h = y_max - y_min
    
    # Calculate square crop with margins
    object_ratio = 1 - 2 * margin_ratio
    max_dim = max(w, h)
    square_size = int(max_dim / object_ratio)
    
    # Calculate center
    center_x = x_min + w // 2
    center_y = y_min + h // 2
    
    crop_x1 = center_x - square_size // 2
    crop_y1 = center_y - square_size // 2
    crop_x2 = crop_x1 + square_size
    crop_y2 = crop_y1 + square_size
    
    # Create white canvas
    canvas = np.ones((square_size, square_size, 3), dtype=np.uint8) * 255
    
    # Adjust crop boundaries
    paste_x = max(0, -crop_x1)
    paste_y = max(0, -crop_y1)
    src_x1 = max(0, crop_x1)
    src_y1 = max(0, crop_y1)
    src_x2 = min(crop_x2, img_rgba.shape[1])
    src_y2 = min(crop_y2, img_rgba.shape[0])
    
    src_w = src_x2 - src_x1
    src_h = src_y2 - src_y1
    
    # Extract regions
    source_region = img_rgba[src_y1:src_y2, src_x1:src_x2]
    rgb_region = source_region[:, :, :3]
    alpha_region = source_region[:, :, 3:4] / 255.0
    
    # Blend with white background
    white_bg = np.ones_like(rgb_region) * 255
    final_region = (rgb_region * alpha_region + white_bg * (1 - alpha_region)).astype(np.uint8)
    
    # Paste onto canvas
    canvas[paste_y:paste_y + src_h, paste_x:paste_x + src_w] = final_region
    
    return canvas
    

def apply_image_adjustments(img_array, brightness_factor, saturation_factor, contrast_factor):
    """
    Apply brightness, saturation, and contrast adjustments to an image array.
    Factors should be in range 0.75 to 1.25 for -25% to +25% adjustment.
    """
    pil_img = Image.fromarray(img_array)
    
    # Apply brightness
    brightness_enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = brightness_enhancer.enhance(brightness_factor)
    
    # Apply saturation
    saturation_enhancer = ImageEnhance.Color(pil_img)
    pil_img = saturation_enhancer.enhance(saturation_factor)
    
    # Apply contrast
    contrast_enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = contrast_enhancer.enhance(contrast_factor)
    
    return np.array(pil_img).astype(np.uint8)

def apply_watermark(base_image_array, watermark_image):
    """
    Applies a watermark to an image and returns the result.
    """
    base_image = Image.fromarray(base_image_array).convert("RGBA")
    watermark = watermark_image.convert("RGBA")

    wm_width = base_image.width // 4
    wm_height = int(watermark.height * (wm_width / watermark.width))
    watermark = watermark.resize((wm_width, wm_height), Image.LANCZOS)

    alpha = watermark.split()[-1]
    alpha = alpha.point(lambda i: i * 0.85)
    watermark.putalpha(alpha)

    margin_ratio = 0.076
    margin_x = int(base_image.width * margin_ratio)
    margin_y = int(base_image.height * margin_ratio)
    pos_x = base_image.width - watermark.width - margin_x
    pos_y = margin_y
    position = (pos_x, pos_y)

    watermark_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    watermark_layer.paste(watermark, position, watermark)

    watermarked_image = Image.alpha_composite(base_image, watermark_layer)
    return watermarked_image.convert("RGB")


def apply_brand_watermark(base_image_array, brand_watermark_image):
    """
    Applies a brand watermark to the top LEFT corner of an image.
    Uses same margin and opacity as main watermark.
    Brand logo is centered in a square frame and vertically aligned with main badge.
    """
    base_image = Image.fromarray(base_image_array).convert("RGBA")
    watermark = brand_watermark_image.convert("RGBA")

    # Calculate size for square frame (same width as main watermark)
    square_size = base_image.width // 4
    
    # Create a transparent square frame
    square_frame = Image.new("RGBA", (square_size, square_size), (0, 0, 0, 0))
    
    # Resize brand logo to fit within square while maintaining aspect ratio
    # Leave some padding (90% of square size)
    max_logo_size = int(square_size * 0.9)
    watermark.thumbnail((max_logo_size, max_logo_size), Image.LANCZOS)
    
    # Center the logo within the square frame
    logo_x = (square_size - watermark.width) // 2
    logo_y = (square_size - watermark.height) // 2
    square_frame.paste(watermark, (logo_x, logo_y), watermark)
    
    # Apply opacity to the entire square frame
    alpha = square_frame.split()[-1]
    alpha = alpha.point(lambda i: i * 0.85)
    square_frame.putalpha(alpha)

    # Position in top left corner with same margin as main watermark
    margin_ratio = 0.076
    margin_x = int(base_image.width * margin_ratio)
    margin_y = int(base_image.height * margin_ratio)
    pos_x = margin_x
    pos_y = margin_y
    position = (pos_x, pos_y)

    watermark_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    watermark_layer.paste(square_frame, position, square_frame)

    watermarked_image = Image.alpha_composite(base_image, watermark_layer)
    return watermarked_image.convert("RGB")


def load_watermark():
    """
    Load watermark from badge.png in repository root.
    Returns PIL Image or None if not found.
    """
    if Path(WATERMARK_PATH).exists():
        return Image.open(WATERMARK_PATH)
    return None


def load_brand_logos():
    """
    Load brand logos information from JSON file.
    Returns dict with brand names as keys and filepaths as values, or None if not found.
    """
    if Path(BRAND_LOGOS_JSON).exists():
        try:
            with open(BRAND_LOGOS_JSON, 'r') as f:
                data = json.load(f)
                # Create a dictionary mapping brand names to filepaths
                brands = {item['name']: item['filepath'] for item in data['data']}
                return brands
        except Exception as e:
            st.error(f"Error loading brand logos JSON: {str(e)}")
            return None
    return None


def load_brand_logo_image(filepath):
    """
    Load a specific brand logo image from filepath.
    Returns PIL Image or None if not found.
    """
    if Path(filepath).exists():
        try:
            return Image.open(filepath)
        except Exception as e:
            st.error(f"Error loading brand logo from {filepath}: {str(e)}")
            return None
    return None


def load_image_file(uploaded_file):
    """
    Load image from uploaded file, handling various formats including JFIF, AVIF, and HEIC.
    Converts transparent backgrounds to white.
    Automatically corrects image orientation based on EXIF data.
    HEIC images are automatically resized to max 3000px to improve processing speed.
    JPG/JPEG images are automatically resized to max 900px to improve processing speed.
    Returns tuple: (image as numpy array in BGR format, has_transparency flag)
    """
    try:
        # For HEIC, AVIF, JFIF and other PIL-supported formats, use PIL first then convert to OpenCV
        file_ext = uploaded_file.name.lower().split('.')[-1]
        
        if file_ext in ['heic', 'heif', 'avif', 'jfif', 'jpg', 'jpeg', 'png']:
            # Use PIL to open these formats
            pil_image = Image.open(uploaded_file)
            
            # Check for transparency BEFORE any conversions
            has_trans = has_transparency(pil_image)
            
            # Correct orientation based on EXIF data
            try:
                from PIL import ImageOps
                pil_image = ImageOps.exif_transpose(pil_image)
            except Exception:
                pass  # If EXIF orientation fails, continue with original
            
            # HEIC Pre-processing: Resize large images for faster processing
            if file_ext in ['heic', 'heif']:
                max_dimension = 900
                width, height = pil_image.size
                
                if width > max_dimension or height > max_dimension:
                    if width > height:
                        new_width = max_dimension
                        new_height = int((max_dimension / width) * height)
                    else:
                        new_height = max_dimension
                        new_width = int((max_dimension / height) * width)
                    
                    pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            # JPG/JPEG Pre-processing: Resize for faster processing
            if file_ext in ['jpg', 'jpeg']:
                max_dimension = 900
                width, height = pil_image.size
                
                if width > max_dimension or height > max_dimension:
                    if width > height:
                        new_width = max_dimension
                        new_height = int((max_dimension / width) * height)
                    else:
                        new_height = max_dimension
                        new_width = int((max_dimension / height) * width)
                    
                    pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert transparent backgrounds to white
            if pil_image.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                white_bg = Image.new('RGB', pil_image.size, (255, 255, 255))
                # If image has transparency, paste it on white background
                if pil_image.mode == 'P':
                    pil_image = pil_image.convert('RGBA')
                white_bg.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode in ('RGBA', 'LA') else None)
                pil_image = white_bg
            elif pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # Convert PIL to numpy array (RGB)
            img_array = np.array(pil_image)
            # Convert RGB to BGR for OpenCV
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            # Standard OpenCV decoding for common formats
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            
            # Check for transparency before orientation correction
            has_trans = (img is not None and len(img.shape) == 3 and img.shape[2] == 4)
            
            # Correct orientation for standard formats
            try:
                from PIL import ImageOps
                uploaded_file.seek(0)  # Reset file pointer
                pil_temp = Image.open(uploaded_file)
                pil_temp = ImageOps.exif_transpose(pil_temp)
                img_array = np.array(pil_temp)
                
                # Convert based on mode
                if pil_temp.mode == 'RGB':
                    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                elif pil_temp.mode == 'RGBA':
                    # Handle transparency
                    bgr = cv2.cvtColor(img_array[:, :, :3], cv2.COLOR_RGB2BGR)
                    alpha = img_array[:, :, 3:4] / 255.0
                    white_bg = np.ones_like(bgr) * 255
                    img = (bgr * alpha + white_bg * (1 - alpha)).astype(np.uint8)
                else:
                    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            except Exception:
                # Fallback if EXIF processing fails
                pass
            
            # Check if image has alpha channel (transparency)
            if img is not None and len(img.shape) == 3 and img.shape[2] == 4:
                # Image has alpha channel - convert to white background
                bgr = img[:, :, :3]
                alpha = img[:, :, 3:4] / 255.0
                
                white_bg = np.ones_like(bgr) * 255
                
                img = (bgr * alpha + white_bg * (1 - alpha)).astype(np.uint8)
            elif img is not None and len(img.shape) == 2:
                # Grayscale image, convert to BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        if img is None:
            raise ValueError(f"Failed to load image: {uploaded_file.name}")
            
        return img, has_trans
        
    except Exception as e:
        raise ValueError(f"Error loading {uploaded_file.name}: {str(e)}")
        

# === Streamlit App ===
def main():
    st.set_page_config(
        page_title="Grassroots Photobooth Processor",
        page_icon="📸",
        layout="wide"
    )

    # Initialize session state for authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Initialize session state for tips modal
    if 'show_tips' not in st.session_state:
        st.session_state.show_tips = False

    # Login page
    if not st.session_state.authenticated:
        st.title("🔒 Grassroots Photobooth Image Processor")
        st.write("Please enter the password to access the image processor.")
        
        password = st.text_input("Password", type="password", key="password_input")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("Login", type="primary"):
                if password == CORRECT_PASSWORD:
                    st.session_state.authenticated = True
                    st.session_state.show_tips = True  # Show tips on first login
                    st.rerun()
                else:
                    st.error("❌ Incorrect password. Please try again.")
        
        st.info("💡 Contact the administrator if you need access.")
        return

    # Main application (after authentication)
    st.title("📸 Grassroots Photobooth Image Processor")
    
    # Tips modal - shows after login
    if st.session_state.show_tips:
        with st.container():
            st.info("### 💡 Tips for Best Results")
            st.markdown("""
            **For optimal background removal:**
            
            1. **📏 Manually crop your images** to include as little background as possible before uploading. 
               This significantly improves the effectiveness of the background removal algorithm.
            
            2. **🔆 Adjust Pre-Processing Settings** if you encounter issues with background removal:
               - Use the **Pre-Processing Brightness Boost** slider to lighten dark photos or darken overly bright ones
               - Adjust the **Pre-Processing Contrast** slider to enhance the distinction between product and background
               - These settings are available after you upload images, right before processing
            
            3. **✨ Best practices:**
               - Use well-lit photos with even lighting
               - Avoid shadows on the background when possible
               - Keep the product clearly separated from the background
            """)
            
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button("Got it! ✓", type="primary", use_container_width=True):
                    st.session_state.show_tips = False
                    st.rerun()
    
    # Logout button in sidebar
    with st.sidebar:
        st.write("### Settings")
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.rerun()
        
        # Add button to show tips again
        if st.button("💡 Show Tips"):
            st.session_state.show_tips = True
            st.rerun()
        
        st.write("---")
        st.write("### Image Adjustments")
        
        # Brightness slider
        brightness_pct = st.slider(
            "Brightness",
            min_value=-25,
            max_value=25,
            value=3,
            step=1,
            format="%d%%",
            help="Adjust image brightness from -25% to +25%"
        )
        brightness_factor = 1.0 + (brightness_pct / 100.0)
        
        # Saturation slider
        saturation_pct = st.slider(
            "Saturation",
            min_value=-25,
            max_value=25,
            value=3,
            step=1,
            format="%d%%",
            help="Adjust color saturation from -25% to +25%"
        )
        saturation_factor = 1.0 + (saturation_pct / 100.0)
        
        # Contrast slider
        contrast_pct = st.slider(
            "Contrast",
            min_value=-25,
            max_value=25,
            value=2,
            step=1,
            format="%d%%",
            help="Adjust image contrast from -25% to +25%"
        )
        contrast_factor = 1.0 + (contrast_pct / 100.0)
        
        st.write("---")
        st.write("### Instructions")
        st.write("1. Upload photos to process")
        st.write("2. Adjust brightness, saturation, and contrast above")
        st.write("3. Choose processing mode:")
        st.write("   - **Remove Background + Watermark**: Full processing with background removal")
        st.write("   - **Watermark Only**: Just adds watermark to existing photos")
        st.write("   - **Remove Background Only**: Removes background without watermark")
        st.write("4. Download processed images")
        
        st.write("---")
        st.write("### Supported Formats")
        st.write("JPG, JPEG, PNG, JFIF, AVIF, HEIC")
        st.caption("📱 HEIC images are automatically resized to 3000px max for faster processing")
        st.caption("📷 JPG/JPEG images are automatically resized to 900px max for faster processing")
        
        st.write("---")
        st.write("### Watermark Status")
        watermark = load_watermark()
        if watermark:
            st.success("✅ Watermark loaded from badge.png")
            st.image(watermark, caption="Current Watermark", width=150)
        else:
            st.warning("⚠️ No badge.png found in repository")

    # Check if watermark exists
    watermark_image = load_watermark()
    if not watermark_image:
        st.error("❌ **Error:** badge.png not found in the repository root directory.")
        st.info("Please add a badge.png file to your repository and redeploy the app.")
        return
    
    # Photo upload
    st.write("## Upload Photos to Process")
    uploaded_files = st.file_uploader(
        "Upload one or more photos (JPG, PNG, JFIF, AVIF, HEIC)",
        type=['png', 'jpg', 'jpeg', 'jfif', 'avif', 'heic', 'heif'],
        accept_multiple_files=True,
        key="photos"
    )
    
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} photo(s) uploaded**")
        
        # Pre-processing brightness adjustment
        st.write("### Pre-Processing Adjustment")
        pre_brightness_pct = st.slider(
            "Pre-Processing Brightness Boost",
            min_value=-25,
            max_value=50,
            value=0,
            step=1,
            format="%d%%",
            help="Adjust brightness BEFORE background removal (useful for dark or overly bright photos)"
        )
        pre_brightness_factor = 1.0 + (pre_brightness_pct / 100.0)
        
        # Pre-processing contrast adjustment
        pre_contrast_pct = st.slider(
            "Pre-Processing Contrast",
            min_value=-25,
            max_value=25,
            value=0,
            step=1,
            format="%d%%",
            help="Adjust contrast BEFORE background removal"
        )
        pre_contrast_factor = 1.0 + (pre_contrast_pct / 100.0)
        
        # Processing mode selection
        st.write("### Select Processing Mode")
        col1, col2, col3 = st.columns(3)
        with col1:
            process_full = st.button("🎨 Remove Background + Watermark", type="primary", use_container_width=True)
        with col2:
            process_watermark_only = st.button("🏷️ Watermark Only", use_container_width=True)
        with col3:
            process_background_only = st.button("✂️ Remove Background Only", use_container_width=True)
        
        if process_full or process_watermark_only or process_background_only:
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            processed_images = []
            
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Read image using the new loader function
                    img, has_transparency = load_image_file(uploaded_file)
                    
                    # Apply pre-processing brightness and contrast if needed
                    if pre_brightness_pct != 0 or pre_contrast_pct != 0:
                        img_rgb_pre = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_adjusted_pre = apply_image_adjustments(img_rgb_pre, pre_brightness_factor, 1.0, pre_contrast_factor)
                        img = cv2.cvtColor(img_adjusted_pre, cv2.COLOR_RGB2BGR)
                    
                    if process_watermark_only:
                        # Watermark only mode - convert to RGB array
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # If image has transparency, process it to fit margins
                        if has_transparency:
                            img_rgb = process_transparent_image(img_rgb)
                        
                            # Apply adjustments
                            img_adjusted = apply_image_adjustments(img_rgb, brightness_factor, saturation_factor, contrast_factor)
                            final_image = apply_watermark(img_adjusted, watermark_image)
                    elif process_background_only:
                        # Background removal only - no watermark
                        if has_transparency:
                            # Image already has transparency, just process to fit margins
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            processed = process_transparent_image(img_rgb)
                        else:
                            # Remove background normally
                            processed = remove_background(img)
                        
                        # Apply adjustments
                        final_image_array = apply_image_adjustments(processed, brightness_factor, saturation_factor, contrast_factor)
                        final_image = Image.fromarray(final_image_array)
                    else:
                        # Full processing mode - remove background + adjustments + watermark
                        if has_transparency:
                            # Image already has transparency, just process to fit margins
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            processed = process_transparent_image(img_rgb)
                        else:
                            # Remove background normally
                            processed = remove_background(img)
                        
                        # Apply adjustments
                        processed_adjusted = apply_image_adjustments(processed, brightness_factor, saturation_factor, contrast_factor)
                        final_image = apply_watermark(processed_adjusted, watermark_image)
                    
                    # Convert to bytes for download
                    img_byte_arr = io.BytesIO()
                    final_image.save(img_byte_arr, format='JPEG', quality=95)
                    img_byte_arr.seek(0)
                    
                    # Ensure filename has .jpg extension
                    original_name = uploaded_file.name
                    name_without_ext = os.path.splitext(original_name)[0]
                    jpg_name = f"{name_without_ext}.jpg"
                    
                    processed_images.append({
                        'name': jpg_name,
                        'data': img_byte_arr.getvalue(),
                        'image': final_image
                    })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            
            status_text.text("✅ Processing complete!")
            
            # Store in session state
            st.session_state.processed_images = processed_images
            
    # Display and download processed images
    if 'processed_images' in st.session_state and st.session_state.processed_images:
        st.write("---")
        st.write("## Download Processed Images")
        
        # Create zip file for bulk download
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for img_data in st.session_state.processed_images:
                zip_file.writestr(f"processed_{img_data['name']}", img_data['data'])
        
        zip_buffer.seek(0)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.download_button(
                label="📦 Download All as ZIP",
                data=zip_buffer.getvalue(),
                file_name="processed_photos.zip",
                mime="application/zip",
                type="primary"
            )
        
        # Brand watermark section
        st.write("---")
        st.write("## Add Brand Watermark (Optional)")
        st.write("Add a brand logo to the top left corner of your processed images.")
        
        # Load brand logos
        brand_logos = load_brand_logos()
        
        if brand_logos:
            col1, col2 = st.columns([2, 2])
            
            with col1:
                # Dropdown for brand selection
                brand_names = ["None"] + sorted(brand_logos.keys())
                selected_brand = st.selectbox(
                    "Select Brand Logo",
                    options=brand_names,
                    key="brand_selector"
                )
            
            with col2:
                # Show preview of selected brand logo with grey background
                if selected_brand != "None":
                    brand_logo_path = brand_logos[selected_brand]
                    brand_logo_img = load_brand_logo_image(brand_logo_path)
                    if brand_logo_img:
                        # Create a medium grey background for preview
                        preview_size = (200, 200)
                        grey_bg = Image.new('RGB', preview_size, color=(128, 128, 128))
                        
                        # Resize logo to fit preview while maintaining aspect ratio
                        logo_rgba = brand_logo_img.convert('RGBA')
                        logo_rgba.thumbnail((180, 180), Image.LANCZOS)
                        
                        # Center logo on grey background
                        x = (preview_size[0] - logo_rgba.width) // 2
                        y = (preview_size[1] - logo_rgba.height) // 2
                        grey_bg.paste(logo_rgba, (x, y), logo_rgba)
                        
                        st.image(grey_bg, caption=f"{selected_brand} Logo", width=150)
            
            # Show "Add Brand Watermark" button if a brand is selected
            if selected_brand != "None":
                if st.button("🏷️ Add Brand Watermark", type="primary", use_container_width=True):
                    brand_logo_path = brand_logos[selected_brand]
                    brand_logo_img = load_brand_logo_image(brand_logo_path)
                    
                    if brand_logo_img:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        branded_images = []
                        
                        for idx, img_data in enumerate(st.session_state.processed_images):
                            try:
                                status_text.text(f"Adding brand watermark to {img_data['name']}...")
                                
                                # Convert existing image to array
                                img_array = np.array(img_data['image'])
                                
                                # Apply brand watermark
                                final_image = apply_brand_watermark(img_array, brand_logo_img)
                                
                                # Convert to bytes for download
                                img_byte_arr = io.BytesIO()
                                final_image.save(img_byte_arr, format='JPEG', quality=95)
                                img_byte_arr.seek(0)
                                
                                # Update name to indicate brand
                                original_name = img_data['name']
                                name_without_ext = os.path.splitext(original_name)[0]
                                branded_name = f"{name_without_ext}_branded.jpg"
                                
                                branded_images.append({
                                    'name': branded_name,
                                    'data': img_byte_arr.getvalue(),
                                    'image': final_image
                                })
                                
                                progress_bar.progress((idx + 1) / len(st.session_state.processed_images))
                                
                            except Exception as e:
                                st.error(f"❌ Error adding brand watermark to {img_data['name']}: {str(e)}")
                        
                        status_text.text("✅ Brand watermarking complete!")
                        
                        # Store branded images in session state
                        st.session_state.branded_images = branded_images
                        st.rerun()
                    else:
                        st.error(f"❌ Could not load brand logo: {brand_logo_path}")
        else:
            st.warning("⚠️ Brand logos JSON file not found. Please add brand-logos/brands.json to your repository.")
        
        st.write("---")
        st.write("### Preview and Individual Downloads")
        
        # Display processed images in a grid
        cols = st.columns(3)
        for idx, img_data in enumerate(st.session_state.processed_images):
            with cols[idx % 3]:
                st.image(img_data['image'], caption=img_data['name'], use_container_width=True)
                st.download_button(
                    label=f"⬇️ Download",
                    data=img_data['data'],
                    file_name=f"processed_{img_data['name']}",
                    mime="image/jpeg",
                    key=f"download_{idx}"
                )
    
    # Display branded images if they exist
    if 'branded_images' in st.session_state and st.session_state.branded_images:
        st.write("---")
        st.write("## Download Branded Images")
        
        # Create zip file for branded images
        zip_buffer_branded = io.BytesIO()
        with zipfile.ZipFile(zip_buffer_branded, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for img_data in st.session_state.branded_images:
                zip_file.writestr(f"branded_{img_data['name']}", img_data['data'])
        
        zip_buffer_branded.seek(0)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.download_button(
                label="📦 Download All Branded as ZIP",
                data=zip_buffer_branded.getvalue(),
                file_name="branded_photos.zip",
                mime="application/zip",
                type="primary"
            )
        
        st.write("### Branded Images Preview")
        
        # Display branded images in a grid
        cols = st.columns(3)
        for idx, img_data in enumerate(st.session_state.branded_images):
            with cols[idx % 3]:
                st.image(img_data['image'], caption=img_data['name'], use_container_width=True)
                st.download_button(
                    label=f"⬇️ Download",
                    data=img_data['data'],
                    file_name=f"branded_{img_data['name']}",
                    mime="image/jpeg",
                    key=f"download_branded_{idx}"
                )
    
    elif uploaded_files:
        st.info("👆 Choose a processing mode above to begin!")
    else:
        st.info("👆 Upload photos above to get started!")


if __name__ == "__main__":
    main()
