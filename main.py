import cv2
import numpy as np
from tqdm import tqdm
from imageio.v3 import imread, imwrite
from skimage.util import view_as_windows
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np
import cv2

def calculate_energy(image, saliency_map, depth_map, entropy_energy_normalized):
    if image.ndim != 2:
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = image
    
    gray_image = gray_img.astype(np.float32)
    

    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    energy_edge = np.sqrt(sobel_x**2 + sobel_y**2)
    

    avg_gradient_blocks = np.zeros((24, 24))   

    block_size_x = gray_image.shape[1] // 24
    block_size_y = gray_image.shape[0] // 24
    

    for i in range(24):
        for j in range(24):
            block = energy_edge[i*block_size_y:(i+1)*block_size_y, j*block_size_x:(j+1)*block_size_x]
            avg_gradient_blocks[i, j] = np.mean(block)
    

    flattened_indices = np.argsort(avg_gradient_blocks, axis=None)[-87:]
    largest_blocks_indices = np.unravel_index(flattened_indices, avg_gradient_blocks.shape)
    

    adjusted_energy_edge = energy_edge.copy()
    

    for i, j in zip(*largest_blocks_indices):
        block = energy_edge[i*block_size_y:(i+1)*block_size_y, j*block_size_x:(j+1)*block_size_x]
        adjusted_energy_edge[i*block_size_y:(i+1)*block_size_y, j*block_size_x:(j+1)*block_size_x] = 4.5 * block
    cv2.imshow('adjusted_energy_edge', adjusted_energy_edge.astype(np.uint8))
    cv2.waitKey(5) 

    avg_entropy_blocks = np.zeros((24, 24))
    avg_depth_blocks = np.zeros((24, 24))
    
    for i in range(24):
        for j in range(24):
            entropy_block = entropy_energy_normalized[i*block_size_y:(i+1)*block_size_y, j*block_size_x:(j+1)*block_size_x]
            depth_block = depth_map[i*block_size_y:(i+1)*block_size_y, j*block_size_x:(j+1)*block_size_x]
            avg_entropy_blocks[i, j] = np.mean(entropy_block)
            avg_depth_blocks[i, j] = np.mean(depth_block)
    
    overall_mean_depth = np.mean(avg_depth_blocks)
    overall_mean_entropy = np.mean(avg_entropy_blocks)

    
    adjusted_entropy = entropy_energy_normalized.copy()
    for i in range(24):
        for j in range(24):
            if avg_depth_blocks[i, j] < overall_mean_depth and overall_mean_entropy < avg_entropy_blocks[i, j]:
                block = entropy_energy_normalized[i*block_size_y:(i+1)*block_size_y, j*block_size_x:(j+1)*block_size_x]
                adjusted_entropy[i*block_size_y:(i+1)*block_size_y, j*block_size_x:(j+1)*block_size_x] = 2 * block
    
    cv2.imshow('adjusted_entropy', adjusted_entropy.astype(np.uint8))
    cv2.waitKey(10) 

    avg_brightness_blocks = np.zeros((24, 24))
    

    for i in range(24):
        for j in range(24):
            block = saliency_map[i*block_size_y:(i+1)*block_size_y, j*block_size_x:(j+1)*block_size_x]
            avg_brightness_blocks[i, j] = np.mean(block)
    

    flattened_indices_brightness = np.argsort(avg_brightness_blocks, axis=None)[-120:]
    largest_brightness_blocks_indices = np.unravel_index(flattened_indices_brightness, avg_brightness_blocks.shape)
    

    adjusted_saliency_map = saliency_map.copy()
    

    for i, j in zip(*largest_brightness_blocks_indices):
        block = saliency_map[i*block_size_y:(i+1)*block_size_y, j*block_size_x:(j+1)*block_size_x]
        adjusted_saliency_map[i*block_size_y:(i+1)*block_size_y, j*block_size_x:(j+1)*block_size_x] = 1.5 * block
       
    cv2.imshow('adjusted_saliency_map', adjusted_saliency_map.astype(np.uint8))
    cv2.waitKey(10) 
    otsu_threshold, binary_image  = cv2.threshold(depth_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    depth_map_float = depth_map.astype(np.float32)
    adjusted_depth_map = depth_map_float.copy()
    adjusted_depth_map[depth_map > otsu_threshold] *= 7
    cv2.imshow('adjusted_depth_map', adjusted_depth_map.astype(np.uint8))
    cv2.waitKey(10)  

    energy_combined = (
        1.0 * adjusted_energy_edge +               
        1.0 * adjusted_saliency_map +              
        1.0 * adjusted_depth_map  +               
        0.5 * -adjusted_entropy                          
    )
    
    energy_combined = cv2.normalize(energy_combined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    

    cv2.imshow('Combined Energy Map', energy_combined.astype(np.uint8))
    cv2.waitKey(10)  
    
    return energy_combined


def find_seam(energy):
    rows, cols = energy.shape
    seam = np.zeros(rows, dtype=np.uint32)
    dp = np.zeros_like(energy)
    dp[0] = energy[0]

    for i in range(1, rows):
        for j in range(cols):
            min_energy = dp[i-1, j]
            if j > 0:
                min_energy = min(min_energy, dp[i-1, j-1])
            if j < cols - 1:
                min_energy = min(min_energy, dp[i-1, j+1])
            dp[i, j] = energy[i, j] + min_energy

    seam[rows - 1] = np.argmin(dp[rows - 1])
    for i in range(rows - 2, -1, -1):
        j = seam[i + 1]
        min_energy = dp[i, j]
        if j > 0 and dp[i, j - 1] < min_energy:
            min_energy = dp[i, j - 1]
            seam[i] = j - 1
        elif j < cols - 1 and dp[i, j + 1] < min_energy:
            min_energy = dp[i, j + 1]
            seam[i] = j + 1
        else:
            seam[i] = j

    return seam

def remove_seam(image, seam):
    rows, cols = image.shape[:2]
    
    if image.ndim == 3:
        new_image = np.zeros((rows, cols - 1, 3), dtype=image.dtype)
        for i in range(rows):
            j = seam[i]
            new_image[i, :j] = image[i, :j]
            new_image[i, j:] = image[i, j + 1:]
    else:
        new_image = np.zeros((rows, cols - 1), dtype=image.dtype)
        for i in range(rows):
            j = seam[i]
            new_image[i, :j] = image[i, :j]
            new_image[i, j:] = image[i, j + 1:]
    
    return new_image

def visualize_seam(image, seam):
    for i in range(image.shape[0]):
        image[i, seam[i]] = [0, 255, 0]  

def visualize_seam_on_saliency(saliency_map, seam):
    for i in range(saliency_map.shape[0]):
        saliency_map[i, seam[i]] = 255 

def visualize_seam_on_depth(depth_map, seam):
    for i in range(depth_map.shape[0]):
        depth_map[i, seam[i]] = 255 

def seam_carve(image, num_seams, saliency_map, depth_map, entropy_energy_normalized):
    carved_image = np.copy(image)
    energy = calculate_energy(carved_image, saliency_map, depth_map, entropy_energy_normalized)
    
    cv2.namedWindow("Seam Carving Progress", cv2.WINDOW_NORMAL)
    cv2.imshow("Seam Carving Progress", carved_image)
    cv2.waitKey(10)  
    
    for iteration in tqdm(range(num_seams), desc="Removing seams", leave=True):
        seam = find_seam(energy.astype(np.uint32))
        visualize_seam(carved_image, seam)
        
        visualize_seam_on_saliency(saliency_map, seam)
        visualize_seam_on_depth(depth_map, seam)
        
        cv2.imshow("Seam Carving Progress", carved_image)
        cv2.imshow("Depth Map with Seam", depth_map)
        cv2.waitKey(10)  
        
        carved_image = remove_seam(carved_image, seam)
        saliency_map = remove_seam(saliency_map, seam)
        entropy_energy_normalized = remove_seam(entropy_energy_normalized, seam)
        depth_map = remove_seam(depth_map, seam)

        energy = calculate_energy(carved_image, saliency_map, depth_map, entropy_energy_normalized)
    
    cv2.destroyAllWindows()
    return carved_image

def get_file_paths(category):
    base_path = "Samples dataset/"
    input_image_path = f"{base_path}{category}/{category}.png"
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    entropy_energy = calculate_entropy(image)

    entropy_energy_normalized = cv2.normalize(entropy_energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(f'{base_path}/{category}/{category}_entropy_energy_normalized.png', entropy_energy_normalized) 






    saliency_map_path = f"{base_path}/{category}/{category}_SMap.png"
    depth_map_path = f"{base_path}/{category}/{category}_DMap.png"
    
    entropy_energy_path = f"{base_path}/{category}/{category}_entropy_energy_normalized.png"  
    energy_map_path = f"{base_path}/{category}/{category}_energy_map.png"  
    return input_image_path, saliency_map_path, depth_map_path, entropy_energy_path, energy_map_path

def calculate_entropy(image, window_size=3):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    def entropy(window):
        _, counts = np.unique(window, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs))

    h, w = image.shape

    padded_image = np.pad(image, window_size // 2, mode='reflect')

    windows = view_as_windows(padded_image, (window_size, window_size))


    entropy_map = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            entropy_map[i, j] = entropy(windows[i, j])

    return entropy_map

if __name__ == "__main__":
    
    user_input = input("Enter category (Diana, Baby, Snowman): ")
    input_image_path, saliency_map_path, depth_map_path, entropy_energy_path, energy_map_path = get_file_paths(user_input)

    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    saliency_map = cv2.imread(saliency_map_path, cv2.IMREAD_GRAYSCALE)
    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    saliency_map = saliency_map.astype(np.float32)

    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


    entropy_energy_normalized = cv2.imread(entropy_energy_path, cv2.IMREAD_GRAYSCALE)

    num_seams_to_remove = int(input("How many column you need to delete(like: 214) :"))
    
    carved_image = seam_carve(image, num_seams_to_remove, saliency_map, depth_map, entropy_energy_normalized)
    cv2.imwrite(f'Samples dataset//{user_input}/{user_input}_output.png', carved_image)