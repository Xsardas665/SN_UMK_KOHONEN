import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def initialize_palette(n_colors):
    # Initialize a color palette with random values
    return np.random.rand(n_colors, 3)

def find_best_matching_unit(image_pixel, palette):
    # Find the index of the unit closest to the given pixel in the color space
    distances = np.linalg.norm(image_pixel - palette, axis=1)
    return np.argmin(distances)

def update_palette(palette, image_pixel, bmu_index, learning_rate):
    # Update the palette based on the closest unit for a given pixel
    palette[bmu_index] += learning_rate * (image_pixel - palette[bmu_index])
    return palette

def map_image_to_palette(image, palette):
    # Map colors in the image to the nearest neighbors in the palette
    mapped_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            bmu_index = find_best_matching_unit(image[i, j], palette)
            mapped_image[i, j] = palette[bmu_index]
    return mapped_image

def kohonen_algorithm(image, n_colors, epochs, learning_rate):
    # Initialize the palette
    palette = initialize_palette(n_colors)

    # Kohonen algorithm iterations
    for epoch in range(epochs):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Find the closest unit for a given pixel and update the palette
                bmu_index = find_best_matching_unit(image[i, j], palette)
                palette = update_palette(palette, image[i, j], bmu_index, learning_rate)

    return palette

def main():
    # Load the image
    image = plt.imread('1.jpg')  # Provide the path to your own image

    # Set parameters
    n_colors = int(input("Enter the number of target colors: "))
    epochs = int(input("Enter the number of epochs: "))
    learning_rate = float(input("Enter the learning rate: "))

    # Perform the Kohonen algorithm
    palette = kohonen_algorithm(image, n_colors, epochs, learning_rate)

    # Map colors in the image to the nearest neighbors in the palette
    mapped_image = map_image_to_palette(image, palette)

    # Display the original and transformed images
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    
    plt.subplot(1, 2, 2)
    plt.title("Transformed Image")
    plt.imshow(mapped_image)
    
    plt.show()

if __name__ == "__main__":
    main()
