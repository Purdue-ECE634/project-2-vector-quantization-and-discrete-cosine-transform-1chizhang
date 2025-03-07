import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.cluster import KMeans
import glob
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def load_grayscale_image(image_path):
    """Load an image and convert to grayscale."""
    img = Image.open(image_path).convert('L')
    return np.array(img)


def extract_blocks(image, block_size=4):
    """Extract all block_size x block_size blocks from the image."""
    h, w = image.shape
    # Adjust image dimensions to be divisible by block_size
    h_blocks = h // block_size
    w_blocks = w // block_size

    # Crop image to be divisible by block_size
    image = image[:h_blocks * block_size, :w_blocks * block_size]

    # Extract blocks
    blocks = []
    for i in range(0, h_blocks * block_size, block_size):
        for j in range(0, w_blocks * block_size, block_size):
            block = image[i:i + block_size, j:j + block_size]
            blocks.append(block.flatten())  # Flatten block to 1D vector

    return np.array(blocks), (h_blocks, w_blocks, block_size)


def train_codebook(blocks, codebook_size):
    """Train a codebook using K-means clustering."""
    # Ensure we don't have more clusters than samples
    codebook_size = min(codebook_size, len(blocks))

    kmeans = KMeans(n_clusters=codebook_size, random_state=42, n_init=10)
    kmeans.fit(blocks)
    codebook = kmeans.cluster_centers_
    return codebook


def quantize_image(blocks, codebook, image_shape):
    """Quantize the image using the trained codebook."""
    h_blocks, w_blocks, block_size = image_shape

    # Find the nearest codebook vector for each block
    # Instead of using KMeans.predict, calculate distances manually
    # to avoid the n_samples >= n_clusters error
    distances = np.zeros((len(blocks), len(codebook)))

    for i, block in enumerate(blocks):
        for j, code in enumerate(codebook):
            distances[i, j] = np.sum((block - code) ** 2)

    # Get the index of the closest codebook vector for each block
    indices = np.argmin(distances, axis=1)

    # Reconstruct the image
    reconstructed_blocks = codebook[indices]

    # Reshape blocks back to 2D
    reconstructed_image = np.zeros((h_blocks * block_size, w_blocks * block_size))

    block_idx = 0
    for i in range(0, h_blocks):
        for j in range(0, w_blocks):
            block = reconstructed_blocks[block_idx].reshape(block_size, block_size)
            reconstructed_image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = block
            block_idx += 1

    return reconstructed_image


def evaluate_compression(original, reconstructed):
    """Evaluate the compression using PSNR and SSIM."""
    psnr_value = psnr(original, reconstructed)
    ssim_value = ssim(original, reconstructed,data_range=255)
    return psnr_value, ssim_value


def vector_quantization(image_path, codebook_size=128, block_size=4, training_images=None):
    """Perform vector quantization on an image."""
    # Load the image
    image = load_grayscale_image(image_path)

    # Extract blocks from the image
    blocks, image_shape = extract_blocks(image, block_size)

    if training_images is None:
        # Use the image itself for training
        codebook = train_codebook(blocks, codebook_size)
    else:
        # Use a collection of images for training
        training_blocks = []
        for img_path in training_images:
            training_img = load_grayscale_image(img_path)
            training_blocks_img, _ = extract_blocks(training_img, block_size)
            training_blocks.append(training_blocks_img)
        training_blocks = np.vstack(training_blocks)
        codebook = train_codebook(training_blocks, codebook_size)

    # Quantize the image
    reconstructed_image = quantize_image(blocks, codebook, image_shape)

    # Evaluate the compression
    h_blocks, w_blocks, block_size = image_shape
    original_cropped = image[:h_blocks * block_size, :w_blocks * block_size]
    psnr_value, ssim_value = evaluate_compression(original_cropped, reconstructed_image)

    return original_cropped, reconstructed_image, psnr_value, ssim_value, codebook


def display_results(original, reconstructed, title, psnr_value, ssim_value,image_path):
    """Display the original and reconstructed images with PSNR and SSIM values."""
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed, cmap='gray')
    plt.title(f'{title}\nPSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f}')
    plt.axis('off')

    plt.tight_layout()
    # plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.savefig(f"{image_path.split('.')[0]}_{title.replace(' ', '_')}.png")
    plt.show()


# Example usage:
# Paths should be adjusted according to your directory structure
if __name__ == "__main__":
    # Part 1.1: Vector Quantization with different codebook sizes
    image_path = "D:\\Code_run\\sample_image\\girl.png"

    # Single image codebook with size L=128
    original, reconstructed_128, psnr_128, ssim_128, _ = vector_quantization(
        image_path, codebook_size=128
    )
    display_results(original, reconstructed_128, "VQ with L=128", psnr_128, ssim_128,image_path)

    # Single image codebook with size L=256
    original, reconstructed_256, psnr_256, ssim_256, _ = vector_quantization(
        image_path, codebook_size=256
    )
    display_results(original, reconstructed_256, "VQ with L=256", psnr_256, ssim_256,image_path)

    # Get list of training images (assuming they're in the same directory)
    sample_dir = os.path.dirname(image_path)
    training_images = glob.glob(os.path.join(sample_dir, "*.png"))
    #sort the images
    training_images.sort()
    training_images = [img for img in training_images if img != image_path][:10]  # Use up to 10 different images
    training_images[9] = 'D:\\Code_run\\sample_image\\img12.tif'
    print(training_images)

    if training_images:
        print(f"Using {len(training_images)} images for training the multi-image codebook")

        # Multi-image codebook with size L=128
        original, reconstructed_multi_128, psnr_multi_128, ssim_multi_128, _ = vector_quantization(
            image_path, codebook_size=128, training_images=training_images
        )
        display_results(original, reconstructed_multi_128, "VQ with Multi-Image Codebook L=128",
                        psnr_multi_128, ssim_multi_128,image_path)

        # Multi-image codebook with size L=256
        original, reconstructed_multi_256, psnr_multi_256, ssim_multi_256, _ = vector_quantization(
            image_path, codebook_size=256, training_images=training_images
        )
        display_results(original, reconstructed_multi_256, "VQ with Multi-Image Codebook L=256",
                        psnr_multi_256, ssim_multi_256,image_path)