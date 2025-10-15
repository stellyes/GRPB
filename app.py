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
    # === Convert to multiple color spaces ===
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # === Strategy 1: Detect neutral gray background specifically ===
    # Gray backgrounds have low saturation AND are centered around 128 in a/b channels AND are light
    a_centered = np.abs(a_channel.astype(np.int16) - 128) < 10
    b_centered = np.abs(b_channel.astype(np.int16) - 128) < 10
    low_saturation = s < 12
    very_light = v > 200
    neutral_gray = a_centered & b_centered & low_saturation & very_light
    not_background_mask = (~neutral_gray).astype(np.uint8) * 255
    
    # === Strategy 2: Include anything with decent saturation (colored) ===
    _, sat_mask = cv2.threshold(s, 10, 255, cv2.THRESH_BINARY)
    
    # === Strategy 3: Include darker objects ===
    _, dark_mask = cv2.threshold(v, 215, 255, cv2.THRESH_BINARY_INV)
    
    # === Strategy 4: Edge detection for product boundaries ===
    edges = cv2.Canny(img, 25, 90)
    edges_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=3)
    
    # === Strategy 5: Texture detection ===
    blur = cv2.GaussianBlur(l_channel, (21, 21), 0)
    texture = cv2.absdiff(l_channel, blur)
    _, texture_mask = cv2.threshold(texture, 4, 255, cv2.THRESH_BINARY)
    
    # === Combine masks - anything NOT neutral gray OR has features ===
    combined_mask = cv2.bitwise_or(not_background_mask, sat_mask)
    combined_mask = cv2.bitwise_or(combined_mask, dark_mask)
    combined_mask = cv2.bitwise_or(combined_mask, texture_mask)
    combined_mask = cv2.bitwise_or(combined_mask, edges_dilated)
    
    # === Morphological operations to clean mask ===
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    kernel_large = np.ones((7, 7), np.uint8)
    
    # Close small gaps
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=3)
    # Remove small noise
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    # Fill larger holes
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    
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
    
    # === Dilate mask for smooth edges ===
    final_mask = cv2.dilate(final_mask, kernel_small, iterations=2)
    
    # === Refine: Remove only truly neutral gray pixels ===
    # This catches background that made it through
    very_light_bg = gray > 210
    very_neutral_a = np.abs(a_channel.astype(np.int16) - 128) < 8
    very_neutral_b = np.abs(b_channel.astype(np.int16) - 128) < 8
    very_low_sat = s < 8
    strict_bg = very_light_bg & very_neutral_a & very_neutral_b & very_low_sat
    
    # Remove strict background from mask
    final_mask = cv2.bitwise_and(final_mask, (~strict_bg).astype(np.uint8) * 255)
    
    # === Get bounding box ===
    all_points = np.vstack(final_contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    # Add padding
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)
    
    # === Calculate square crop with margin ===
    margin_ratio = 0.09
    object_ratio = 1 - 2 * margin_ratio
    max_dim = max(w, h)
    square_size = int(max_dim / object_ratio)
    
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
    
    # === Apply mask with blur for smooth edges (2-4 pixel blur) ===
    mask_blurred = cv2.GaussianBlur(mask_region, (7, 7), 0)
    mask_3channel = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2RGB) / 255.0
    
    # Blend with white background
    masked_source = (source_region * mask_3channel).astype(np.uint8)
    white_bg = np.ones_like(source_region) * 255
    final_region = (masked_source + white_bg * (1 - mask_3channel)).astype(np.uint8)
    
    # === Final cleanup: Remove residual background noise ===
    final_gray = cv2.cvtColor(final_region, cv2.COLOR_RGB2GRAY)
    final_hsv = cv2.cvtColor(final_region, cv2.COLOR_RGB2HSV)
    final_h, final_s, final_v = cv2.split(final_hsv)
    final_lab = cv2.cvtColor(final_region, cv2.COLOR_RGB2LAB)
    final_l, final_a, final_b = cv2.split(final_lab)
    
    # Target 1: Very light, very neutral pixels (nearly white/gray)
    nearly_white = (final_gray > 230) & \
                   (np.abs(final_a.astype(np.int16) - 128) < 8) & \
                   (np.abs(final_b.astype(np.int16) - 128) < 8)
    
    # Target 2: Light gray pixels with low saturation (background noise) - MORE AGGRESSIVE
    light_gray_bg = (final_gray > 190) & \
                    (final_gray < 245) & \
                    (final_s < 15) & \
                    (np.abs(final_a.astype(np.int16) - 128) < 15) & \
                    (np.abs(final_b.astype(np.int16) - 128) < 15)
    
    # Combine and apply
    to_whiten = nearly_white | light_gray_bg
    final_region[to_whiten] = [255, 255, 255]
    
    # === Paste onto canvas ===
    canvas[paste_y:paste_y + src_h, paste_x:paste_x + src_w] = final_region
    
    # === Convert to PIL for adjustments ===
    pil_img = Image.fromarray(canvas)
    
    # === Increase brightness by 3% ===
    brightness_enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = brightness_enhancer.enhance(1.03)
    
    # === Increase saturation by 3% ===
    saturation_enhancer = ImageEnhance.Color(pil_img)
    pil_img = saturation_enhancer.enhance(1.03)
    
    # === Increase contrast by 2% ===
    contrast_enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = contrast_enhancer.enhance(1.02)
    
    # === Convert back to array ===
    result = np.array(pil_img).astype(np.uint8)
    
    return result

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
        st.write("### Instructions")
        st.write("1. Upload photos to process")
        st.write("2. Click 'Process All Images'")
        st.write("3. Download processed images")
        
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
        
        if st.button("🎨 Process All Images", type="primary"):
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            processed_images = []
            
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Read image using the new loader function
                    img = load_image_file(uploaded_file)
                    
                    # Process image
                    processed = remove_background(img)
                    
                    # Apply watermark
                    final_image = apply_watermark(processed, watermark_image)
                    
                    # Convert to bytes for download
                    img_byte_arr = io.BytesIO()
                    final_image.save(img_byte_arr, format='JPEG', quality=95)
                    img_byte_arr.seek(0)
                    
                    processed_images.append({
                        'name': uploaded_file.name,
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
        st.info("👆 Click 'Process All Images' to begin processing!")
    else:
        st.info("👆 Upload photos above to get started!")


if __name__ == "__main__":
    main()
