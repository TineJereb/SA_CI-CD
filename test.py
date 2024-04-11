import cv2
import numpy as np
from naloga3 import kmeans, meanshift

def test_kmeans():
    # Testiranje funkcije kmeans
    image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)  # Naključna slika
    segmented_image = kmeans(image, 5, 10, 3)  # Preverimo delovanje funkcije
    assert segmented_image.shape == image.shape  # Preverimo, ali je izhodna slika enake oblike kot vhodna

def test_meanshift():
    # Testiranje funkcije meanshift
    image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)  # Naključna slika
    shifted_image = meanshift(image, 5, 3, 8)  # Preverimo delovanje funkcije
    assert shifted_image.shape == image.shape  # Preverimo, ali je izhodna slika enake oblike kot vhodna

if __name__ == "__main__":
    test_kmeans()
    test_meanshift()
    print("Vsi testi uspešno opravljeni.")
