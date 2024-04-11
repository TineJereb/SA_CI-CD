import numpy as np
import math
import cv2
import numba
from numba import jit
from numba import prange

@jit(nopython=True)
def euclidean_distance(point1, point2, dimension):
    '''Compute the Euclidean distance between two points considering coordinates and color values.'''

    if dimension == 5:
        x1, y1, R1, G1, B1 = point1
        x2, y2, R2, G2, B2 = point2
        
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (R1 - R2) ** 2 + (G1 - G2) ** 2 + (B1 - B2) ** 2)
    elif dimension == 3:
        R1, G1, B1 = point1
        R2, G2, B2 = point2
        return math.sqrt((R1 - R2) ** 2 + (G1 - G2) ** 2 + (B1 - B2) ** 2)

    return 0

def kmeans(image, k, iterations, dimenzija_centra):
    '''Performs image segmentation using the k-means algorithm.'''
    izbira = 'naključno'
    centers = izracunaj_centre(image, izbira, dimenzija_centra, k, T=20)
    centers = [np.array(center, dtype=np.float64) for center in centers]

    segmented_image = np.zeros_like(image)
    pixel_counts = np.zeros((k,), dtype=np.int32)
    center_sums = np.zeros((k, dimenzija_centra), dtype=np.float64)

    for _ in prange(iterations):
        # Reset pixel counts and center sums for each iteration
        pixel_counts.fill(0)
        center_sums.fill(0)

        # Assign pixels to centers and accumulate pixel colors
        for i in prange(image.shape[0]):
            for j in prange(image.shape[1]):
                if dimenzija_centra == 3:
                    pixel = image[i, j]
                elif dimenzija_centra == 5:
                    B, G, R = image[i, j]
                    pixel = np.array([i, j, B, G, R], dtype=np.uint16)

                nearest_center_index = np.argmin([euclidean_distance(pixel, c, dimenzija_centra) for c in centers])

                pixel_counts[nearest_center_index] += 1
                center_sums[nearest_center_index] += pixel

        # Update center colors by averaging the accumulated colors
        for idx in prange(k):
            if pixel_counts[idx] > 0:
                centers[idx] = center_sums[idx] / pixel_counts[idx]

    # Assign each pixel the color of its corresponding center
    for i in prange(image.shape[0]):
        for j in prange(image.shape[1]):
            if dimenzija_centra == 3:
                pixel = image[i, j]
            elif dimenzija_centra == 5:
                B, G, R = image[i, j]
                pixel = np.array([i,j,B, G, R], dtype=np.uint16)

            nearest_center_index = np.argmin([euclidean_distance(pixel, c, dimenzija_centra) for c in centers])
            if dimenzija_centra == 3:
                B,G,R = centers[nearest_center_index]
            elif dimenzija_centra == 5:
                _,_,B,G,R = centers[nearest_center_index]
                
            segmented_image[i, j] = [B,G,R]

    return segmented_image


@jit(nopython=True)
def meanshift(image, h, dimension,radius, max_iterations=10, convergence=0.0001):
    rows, cols, _ = image.shape
    shifted_image = np.zeros_like(image)

    for i in prange(rows):
        for j in prange(cols):
            
            if dimension == 3:
                current_point = image[i, j].astype(np.float32)
            elif dimension == 5:
                B,G,R = image[i,j]
                current_point = np.array([i,j,B,G,R], dtype=np.float32)

            iterations = 0

            while iterations < max_iterations:
                new_point = np.zeros_like(current_point)
                total_weight = 0.0

                for m in range(max(0, i - radius), min(rows, i + radius + 1)):
                    for n in range(max(0, j - radius), min(cols, j + radius + 1)):
                        
                        if dimension == 3:
                            neighbor_point = image[m, n].astype(np.float32)
                        elif dimension == 5:
                            B,G,R = image[m,n].astype(np.float32)
                            neighbor_point = np.array([m,n,B,G,R], dtype=np.float32)

                        distance = euclidean_distance(current_point, neighbor_point,dimension)
                        weight = jedro(distance, h)

                        new_point += weight * neighbor_point
                        total_weight += weight

                new_point /= total_weight
                
                if euclidean_distance(current_point, new_point,dimension) < convergence:
                    break

                current_point = new_point
                iterations += 1
            if dimension == 3:
                shifted_image[i, j] = current_point
            elif dimension == 5:
                _,_,B,G,R = current_point
                shifted_image[i,j] = np.array([B,G,R], dtype=np.float32)

    return shifted_image

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        slika, centri, k, dimenzija = param

        if len(centri) < k:  # Check if maximum clicks is not reached
            if dimenzija == 3:
                centri.append(slika[y,x].astype(int))
            elif dimenzija == 5:
                B,G,R = slika[y,x].astype(int)
                centri.append([x,y,B,G,R])
            cv2.imshow('Izbira centrov', slika)

        
        # Check if maximum clicks is reached, if yes, close the window
        if len(centri) == k:
            cv2.destroyWindow('Izbira centrov')

def izracunaj_centre(slika, izbira, dimenzija_centra, k, T):
    '''Izračuna centre za metodo kmeans.'''
    if izbira == 'ročno':
        # Inicializacija praznega seznama za shranjevanje centrov
        centri = []
        
        # Prikaz slike in čakanje na klike uporabnika
        cv2.namedWindow('Izbira centrov')
        cv2.setMouseCallback('Izbira centrov', click_event, [slika, centri, k, dimenzija_centra])
        cv2.imshow('Izbira centrov', slika)

        while True:
            key = cv2.waitKey(0)
            if key != 1:  # 1 je ASCII koda za levi klik miške
                break

        cv2.destroyAllWindows()
        # Vračanje izbranih centrov
        
        return centri
    
    elif izbira == 'naključno':
        
        # Generiramo naključne centre, ki niso preblizu med seboj
        centri = []
        while len(centri) < k:
            # Naključno izberemo koordinate centra
            
            rand_x = np.random.randint(0,slika.shape[0])
            rand_y = np.random.randint(0,slika.shape[1])
            B,G,R = slika[rand_x,rand_y].astype(int)

            if dimenzija_centra == 5:
                nov_center = np.array([rand_x,rand_y,B,G,R], dtype=np.float64)
            elif dimenzija_centra == 3:
                nov_center = [B,G,R]
            
            # Preverimo, če je seznam centri prazen ali če je nov center dovolj oddaljen od ostalih
            if not centri or all(euclidean_distance(center, nov_center, dimenzija_centra) >= T for center in centri):
                centri.append(nov_center)
        
        return centri

def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

@jit(nopython=True)
def jedro(u, h):
    '''Izračuna vrednost gaussovega jedra.'''
    return (1 / (h * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (u / h) ** 2)


if __name__ == "__main__":
    
    # Load the image
    image_original = cv2.imread("zelenjava.jpg")
    image = image_original.copy()
    image_original = resize_image(image_original,256,256)
    image_original = resize_image(image_original,512,512)
    image = resize_image(image,256,256)
    
    #image = kmeans(image,3,10,3)
    #image_3 = meanshift(image,1,5,10)
    image_5 = meanshift(image,10,5,10)

    
    #image_3 = resize_image(image_3,512,512)
    image_5 = resize_image(image_5,512,512)

    #cv2.imshow('Obdelana slika 3', image_3)
    cv2.imshow('Obdelana slika 5', image_5)

    cv2.imshow('Originalna slika', image_original)

    cv2.waitKey(0)
    cv2.destroyAllWindows()