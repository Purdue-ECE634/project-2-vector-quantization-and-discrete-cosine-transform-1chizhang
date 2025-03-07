import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.fftpack import dct, idct
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def load_image(image_path, grayscale=False):
    """Load an image and convert to grayscale if specified."""
    if grayscale:
        img = Image.open(image_path).convert('L')
        return np.array(img)
    else:
        img = Image.open(image_path)
        return np.array(img)


def dct2(block):
    """2D DCT Transform."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    """2D Inverse DCT Transform."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def extract_blocks(image, block_size=8):
    """Extract all block_size x block_size blocks from the image for each channel."""
    if len(image.shape) == 3:  # Color image
        h, w, channels = image.shape

        # Adjust image dimensions to be divisible by block_size
        h_blocks = h // block_size
        w_blocks = w // block_size

        # Crop image to be divisible by block_size
        image = image[:h_blocks * block_size, :w_blocks * block_size, :]

        # Initialize blocks for each channel
        blocks = [np.zeros((h_blocks * w_blocks, block_size, block_size)) for _ in range(channels)]

        # Extract blocks for each channel
        for c in range(channels):
            block_idx = 0
            for i in range(0, h_blocks * block_size, block_size):
                for j in range(0, w_blocks * block_size, block_size):
                    blocks[c][block_idx] = image[i:i + block_size, j:j + block_size, c]
                    block_idx += 1

        return blocks, (h_blocks, w_blocks, block_size, channels)

    else:  # Grayscale image
        h, w = image.shape

        # Adjust image dimensions to be divisible by block_size
        h_blocks = h // block_size
        w_blocks = w // block_size

        # Crop image to be divisible by block_size
        image = image[:h_blocks * block_size, :w_blocks * block_size]

        # Extract blocks
        blocks = np.zeros((h_blocks * w_blocks, block_size, block_size))
        block_idx = 0
        for i in range(0, h_blocks * block_size, block_size):
            for j in range(0, w_blocks * block_size, block_size):
                blocks[block_idx] = image[i:i + block_size, j:j + block_size]
                block_idx += 1

        return [blocks], (h_blocks, w_blocks, block_size, 1)


def keep_top_k_coefficients(dct_block, k):
    """Keep only the top K DCT coefficients in zigzag order."""
    # Create a zigzag mask
    mask = np.zeros_like(dct_block)
    block_size = dct_block.shape[0]

    # Fill the zigzag pattern
    count = 0
    for s in range(2 * block_size - 1):
        if s < block_size:
            # Upper triangle
            for i in range(min(s + 1, block_size)):
                j = s - i
                if j < block_size:
                    if count < k:
                        mask[i, j] = 1
                    count += 1
        else:
            # Lower triangle
            for i in range(s - block_size + 1, block_size):
                j = s - i
                if j >= 0:
                    if count < k:
                        mask[i, j] = 1
                    count += 1

    # Apply the mask
    return dct_block * mask


def dct_compression(image_path, k_values=[2, 4, 8, 16, 32], block_size=8, grayscale=False):
    """Compress an image using DCT and keeping only the top K coefficients."""
    # Load the image
    image = load_image(image_path, grayscale)

    # Extract blocks for each channel
    blocks, (h_blocks, w_blocks, block_size, channels) = extract_blocks(image, block_size)

    results = {}  # Use a dictionary to store results by K value

    # Process for each K value
    for k in k_values:
        if channels == 1:  # Grayscale
            # DCT transform and keep top K coefficients
            reconstructed_blocks = np.zeros_like(blocks[0])
            for i in range(blocks[0].shape[0]):
                dct_block = dct2(blocks[0][i])
                filtered_dct_block = keep_top_k_coefficients(dct_block, k)
                reconstructed_blocks[i] = idct2(filtered_dct_block)

            # Reconstruct the image
            reconstructed_image = np.zeros((h_blocks * block_size, w_blocks * block_size))
            block_idx = 0
            for i in range(h_blocks):
                for j in range(w_blocks):
                    reconstructed_image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = \
                    reconstructed_blocks[block_idx]
                    block_idx += 1

            # Clip values to valid range [0, 255]
            reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

            # Evaluate the compression
            original_cropped = image[:h_blocks * block_size, :w_blocks * block_size]
            psnr_value = psnr(original_cropped, reconstructed_image)
            ssim_value = ssim(original_cropped, reconstructed_image, data_range=255)

        else:  # Color image
            # Initialize reconstructed image
            reconstructed_image = np.zeros((h_blocks * block_size, w_blocks * block_size, channels), dtype=np.uint8)

            # Process each channel separately
            for c in range(channels):
                reconstructed_blocks = np.zeros_like(blocks[c])
                for i in range(blocks[c].shape[0]):
                    dct_block = dct2(blocks[c][i])
                    filtered_dct_block = keep_top_k_coefficients(dct_block, k)
                    reconstructed_blocks[i] = idct2(filtered_dct_block)

                # Reconstruct this channel
                block_idx = 0
                for i in range(h_blocks):
                    for j in range(w_blocks):
                        reconstructed_image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size,
                        c] = np.clip(reconstructed_blocks[block_idx], 0, 255)
                        block_idx += 1

            # Clip values to valid range [0, 255]
            reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

            # Evaluate the compression
            original_cropped = image[:h_blocks * block_size, :w_blocks * block_size, :]
            # Calculate metrics using the multichannel PSNR and SSIM
            psnr_value = psnr(original_cropped, reconstructed_image, data_range=255)
            # For color images, we calculate SSIM for each channel and take the average
            ssim_value = np.mean([ssim(original_cropped[:, :, c], reconstructed_image[:, :, c], data_range=255)
                                  for c in range(channels)])

        # Store results in dictionary by K value
        results[k] = (original_cropped, reconstructed_image, psnr_value, ssim_value)

    return results


def display_custom_layout(results):
    """
    Display results in a specific layout:
    Top row: Original, K=4, K=16
    Bottom row: K=2, K=8, K=32
    """
    plt.figure(figsize=(15, 10))

    # Get original image (same for all k values)
    original = list(results.values())[0][0]

    # Top row
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray' if len(original.shape) == 2 else None)
    plt.title('Original Image')
    plt.axis('off')

    # K=4
    if 4 in results:
        plt.subplot(2, 3, 2)
        plt.imshow(results[4][1], cmap='gray' if len(results[4][1].shape) == 2 else None)
        plt.title(f'K=4, PSNR: {results[4][2]:.2f}, SSIM: {results[4][3]:.4f}')
        plt.axis('off')

    # K=16
    if 16 in results:
        plt.subplot(2, 3, 3)
        plt.imshow(results[16][1], cmap='gray' if len(results[16][1].shape) == 2 else None)
        plt.title(f'K=16, PSNR: {results[16][2]:.2f}, SSIM: {results[16][3]:.4f}')
        plt.axis('off')

    # Bottom row
    # K=2
    if 2 in results:
        plt.subplot(2, 3, 4)
        plt.imshow(results[2][1], cmap='gray' if len(results[2][1].shape) == 2 else None)
        plt.title(f'K=2, PSNR: {results[2][2]:.2f}, SSIM: {results[2][3]:.4f}')
        plt.axis('off')

    # K=8
    if 8 in results:
        plt.subplot(2, 3, 5)
        plt.imshow(results[8][1], cmap='gray' if len(results[8][1].shape) == 2 else None)
        plt.title(f'K=8, PSNR: {results[8][2]:.2f}, SSIM: {results[8][3]:.4f}')
        plt.axis('off')

    # K=32
    if 32 in results:
        plt.subplot(2, 3, 6)
        plt.imshow(results[32][1], cmap='gray' if len(results[32][1].shape) == 2 else None)
        plt.title(f'K=32, PSNR: {results[32][2]:.2f}, SSIM: {results[32][3]:.4f}')
        plt.axis('off')

    plt.tight_layout()
    is_color = len(original.shape) > 2
    plt.savefig(f"DCT_Compression_{'Color' if is_color else 'Gray'}.png")
    plt.show()


def analyze_dct_results(results):
    """Analyze and plot the PSNR and SSIM values vs K."""
    k_values = sorted(list(results.keys()))
    psnr_values = [results[k][2] for k in k_values]
    ssim_values = [results[k][3] for k in k_values]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(k_values, psnr_values, 'o-')
    plt.xlabel('Number of DCT Coefficients (K)')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR vs Number of DCT Coefficients')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(k_values, ssim_values, 'o-')
    plt.xlabel('Number of DCT Coefficients (K)')
    plt.ylabel('SSIM')
    plt.title('SSIM vs Number of DCT Coefficients')
    plt.grid(True)

    plt.tight_layout()
    original = list(results.values())[0][0]
    is_color = len(original.shape) > 2
    plt.savefig(f"DCT_Analysis_{'Color' if is_color else 'Gray'}.png")
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Path to your image
    image_path = "D:\\Code_run\\sample_image\\girl.png"

    # Choose whether to process as grayscale or color
    grayscale = False  # Set to False for color processing, True for grayscale

    # Compress the image using different K values
    dct_results = dct_compression(image_path, k_values=[2, 4, 8, 16, 32], grayscale=grayscale)

    # Display the results with the custom layout
    display_custom_layout(dct_results)

    # Analyze the results
    analyze_dct_results(dct_results)