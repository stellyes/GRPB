import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import zipfile
from pathlib import Path
import os

# === Password Configuration ===
# Get password from Streamlit secrets
CORRECT_PASSWORD = st.secrets["password"]

# === Watermark Configuration ===
WATERMARK_PATH = "badge.png"

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


def load_watermark():
    """
    Load watermark from badge.png in repository root.
    Returns PIL Image or None if not found.
    """
    if Path(WATERMARK_PATH).exists():
        return Image.open(WATERMARK_PATH)
    return None


def load_image_file(uploaded_file):
    """
    Load image from uploaded file, handling various formats including JFIF and AVIF.
    Returns image as numpy array in BGR format for OpenCV processing.
    """
    try:
        # For AVIF and other PIL-supported formats, use PIL first then convert to OpenCV
        file_ext = uploaded_file.name.lower().split('.')[-1]
        
        if file_ext in ['avif', 'jfif']:
            # Use PIL to open these formats
            pil_image = Image.open(uploaded_file)
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # Convert PIL to numpy array (RGB)
            img_array = np.array(pil_image)
            # Convert RGB to BGR for OpenCV
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            # Standard OpenCV decoding for common formats
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
        if img is None:
            raise ValueError(f"Failed to load image: {uploaded_file.name}")
            
        return img
        
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
                    st.rerun()
                else:
                    st.error("❌ Incorrect password. Please try again.")
        
        st.info("💡 Contact the administrator if you need access.")
        return

    # Main application (after authentication)
    st.title("📸 Grassroots Photobooth Image Processor")
    
    # Logout button in sidebar
    with st.sidebar:
        st.write("### Settings")
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
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
        st.write("4. Download processed images")
        
        st.write("---")
        st.write("### Supported Formats")
        st.write("JPG, JPEG, PNG, JFIF, AVIF")
        
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
        "Upload one or more photos (JPG, PNG, JFIF, AVIF)",
        type=['png', 'jpg', 'jpeg', 'jfif', 'avif'],
        accept_multiple_files=True,
        key="photos"
    )
    
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} photo(s) uploaded**")
        
        # Pre-processing brightness adjustment
        st.write("### Pre-Processing Adjustment")
        pre_brightness_pct = st.slider(
            "Pre-Processing Brightness Boost",
            min_value=0,
            max_value=25,
            value=0,
            step=1,
            format="%d%%",
            help="Brighten the image BEFORE background removal (useful for dark photos)"
        )
        pre_brightness_factor = 1.0 + (pre_brightness_pct / 100.0)
        
        # Processing mode selection
        st.write("### Select Processing Mode")
        col1, col2 = st.columns(2)
        with col1:
            process_full = st.button("🎨 Remove Background + Watermark", type="primary", use_container_width=True)
        with col2:
            process_watermark_only = st.button("🏷️ Watermark Only", use_container_width=True)
        
        if process_full or process_watermark_only:
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            processed_images = []
            
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Read image using the new loader function
                    img = load_image_file(uploaded_file)
                    
                    # Apply pre-processing brightness if needed
                    if pre_brightness_pct > 0:
                        img_rgb_pre = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_brightened = apply_image_adjustments(img_rgb_pre, pre_brightness_factor, 1.0, 1.0)
                        img = cv2.cvtColor(img_brightened, cv2.COLOR_RGB2BGR)
                    
                    if process_watermark_only:
                        # Watermark only mode - convert to RGB array
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # Apply adjustments
                        img_adjusted = apply_image_adjustments(img_rgb, brightness_factor, saturation_factor, contrast_factor)
                        final_image = apply_watermark(img_adjusted, watermark_image)
                    else:
                        # Full processing mode - remove background + adjustments + watermark
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
    
    elif uploaded_files:
        st.info("👆 Choose a processing mode above to begin!")
    else:
        st.info("👆 Upload photos above to get started!")


if __name__ == "__main__":
    main()
