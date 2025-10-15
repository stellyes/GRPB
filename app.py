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
    Preserves product details like shadows while removing background gradients.
    Returns processed image as numpy array.
    """
    # === Convert to RGB and HSV ===
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    
    # === Convert to LAB color space ===
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    
    # === Strategy 1: Saturation-based mask (backgrounds are desaturated) ===
    # Product elements usually have more color saturation than gray backgrounds
    _, sat_mask = cv2.threshold(s, 15, 255, cv2.THRESH_BINARY)
    
    # === Strategy 2: Luminance-based mask (darker than background) ===
    # Background is typically light gray/white (high luminance)
    _, lum_mask = cv2.threshold(v, 200, 255, cv2.THRESH_BINARY_INV)
    
    # === Strategy 3: Edge detection to find product boundaries ===
    # Products have strong edges, backgrounds don't
    edges = cv2.Canny(img, 30, 100)
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    
    # === Strategy 4: Color distance from pure gray ===
    # Gray background has a ≈ 128, b ≈ 128 in LAB space
    a_dist = np.abs(a_channel.astype(np.int16) - 128)
    b_dist = np.abs(b_channel.astype(np.int16) - 128)
    color_dist = a_dist + b_dist
    _, color_mask = cv2.threshold(color_dist.astype(np.uint8), 8, 255, cv2.THRESH_BINARY)
    
    # === Strategy 5: Texture detection ===
    # Background is smooth, products have texture/detail
    blur = cv2.GaussianBlur(l_channel, (15, 15), 0)
    texture = cv2.absdiff(l_channel, blur)
    _, texture_mask = cv2.threshold(texture, 5, 255, cv2.THRESH_BINARY)
    
    # === Combine all masks with OR operation ===
    combined_mask = cv2.bitwise_or(sat_mask, lum_mask)
    combined_mask = cv2.bitwise_or(combined_mask, color_mask)
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
    # Fill holes
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
    
    # === Keep contours that are either large or close to the main contour ===
    final_contours = [main_contour]
    for contour in contours:
        if contour is main_contour:
            continue
        # Keep if it's at least 2% of main contour size (for shadows, drips, etc.)
        if cv2.contourArea(contour) > main_area * 0.02:
            final_contours.append(contour)
    
    # === Build final mask ===
    final_mask = np.zeros_like(combined_mask)
    for contour in final_contours:
        cv2.drawContours(final_mask, [contour], -1, 255, -1)
    
    # === Dilate mask slightly for smooth edges ===
    final_mask = cv2.dilate(final_mask, kernel_small, iterations=2)
    
    # === Additional pass: Remove remaining gray background pixels ===
    # Create a strict background removal mask based on luminance and saturation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, strict_mask = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY_INV)
    _, sat_strict = cv2.threshold(s, 10, 255, cv2.THRESH_BINARY)
    strict_bg_removal = cv2.bitwise_or(strict_mask, sat_strict)
    
    # Combine with our existing mask
    final_mask = cv2.bitwise_and(final_mask, strict_bg_removal)
    
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
    
    # === Apply mask with slight blur for smooth edges ===
    mask_blurred = cv2.GaussianBlur(mask_region, (3, 3), 0)
    mask_3channel = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2RGB) / 255.0
    
    # Blend with white background
    masked_source = (source_region * mask_3channel).astype(np.uint8)
    white_bg = np.ones_like(source_region) * 255
    final_region = (masked_source + white_bg * (1 - mask_3channel)).astype(np.uint8)
    
    # === Post-process: Replace any remaining light gray pixels with pure white ===
    gray_final = cv2.cvtColor(final_region, cv2.COLOR_RGB2GRAY)
    light_pixels = gray_final > 230
    final_region[light_pixels] = [255, 255, 255]
    
    # === Paste onto canvas ===
    canvas[paste_y:paste_y + src_h, paste_x:paste_x + src_w] = final_region
    
    # === Convert to PIL for adjustments ===
    pil_img = Image.fromarray(canvas)
    
    # === Increase saturation by 4% ===
    saturation_enhancer = ImageEnhance.Color(pil_img)
    pil_img = saturation_enhancer.enhance(1.04)
    
    # === Increase brightness by 2.75% ===
    brightness_enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = brightness_enhancer.enhance(1.0275)
    
    # === Slight contrast enhancement ===
    contrast_enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = contrast_enhancer.enhance(1.03)
    
    # === Levels adjustment ===
    img = np.array(pil_img).astype(np.float32)
    img = np.clip(img * (255.0 / 219.0), 0, 255)
    result = img.astype(np.uint8)
    
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
        "Upload one or more photos",
        type=['png', 'jpg', 'jpeg'],
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
                    
                    # Read image
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
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