import h5py
import numpy as np
from matplotlib import pyplot as plt
import random
import os
import cv2
from matplotlib import image as mpimg
import traceback
from skimage import io, color
from skimage.metrics import structural_similarity as compare_ssim
import scipy.io
try:
    import cupy
except:
    pass
import datetime
from IPython import display
from matplotlib import pyplot as plt
#from mri_cs import column2matrix, matrix2column, prefiltering, radial_lines_samples_rows
import numpy as np
import os
import pathlib
import pydot
from scipy.io import loadmat
import tensorflow as tf
import time
import zipfile
from PIL import Image
import glob
import pandas as pd
import matplotlib.image as mpimg
from tensorflow.keras.utils import plot_model



def open_single_file(file_name):
    try:
        hf = h5py.File(file_name)
    except:
        return
    volume_kspace = hf['kspace'][()]
    return volume_kspace

def open_single_mat(file_name):
    mat_data = scipy.io.loadmat(file_name)
    volume_data = mat_data['x']
    try:
        return [volume_data[:, :, i] for i in range(volume_data.shape[2])]
    except:
        return


def count_files(folder):
    items = os.listdir(folder)

    files = [item for item in items if os.path.isfile(os.path.join(folder, item))]

    return len(files)

def see_k_space(k, title):
    
    
    
    # k = np.abs(k)
    k = k - np.mean(k)
    k = np.log(np.abs(k) + 10e-8)

    k = (k - np.min(k)) / (np.max(k) - np.min(k))

    plt.title(title)
    plt.imshow(k, cmap='gray')
    plt.axis('off')
    plt.show()


def see_mask(mask):
    plt.imshow(mask, cmap='gray', interpolation='none')
    plt.title("Mask")
    plt.axis('off')
    plt.show()


def inverse_transform(k_space):
    image = abs(np.fft.ifft2(k_space))
    image = np.fft.ifftshift(image)
    return image

def direct_transform(image):
    k_space = np.fft.fft2(image)
    return k_space

def apply_undersampling_mask_random(dim, mask_percentage=0.7):
    matrix = np.ones((dim, dim))
    num_columns_to_mask = int(dim * mask_percentage)
    mask_index = random.sample(range(dim), num_columns_to_mask)
    matrix[:, mask_index] = 0
    
    return matrix

def apply_undersampling_mask_hf(dim, start_interval, interval):
    num_columns = dim
    center = int(num_columns/2)
    start_pos = center + start_interval
    start_neg = center - start_interval
    matrix = np.ones((dim, dim))
    for col in range(start_pos, num_columns, interval):
        matrix[:, col] = 0
    for col in range(start_neg, 0, -interval):
        matrix[:, col] = 0
             
    return matrix


def save_both(image1, image2, save_path):
    print(image1.dtype)
    print(image2.dtype)
    combined_image = np.concatenate((image1, image2), axis=1)
    print(combined_image.dtype)
    merged_images = []
    merged_images.append(combined_image)  # Armazena a imagem processada



    return merged_images



def k_space_greater_dimension(input_folder, n_slice, data_set_size):
    altura_max = 0
    largura_max = 0
    counter = 0
    for filename in os.listdir(input_folder):
        if filename.endswith(".h5"):
            file_path = os.path.join(input_folder, filename)
            try:
                k_space = open_single_file(file_path)
                altura, largura = k_space[n_slice].shape[:2]
                if altura > altura_max:
                    altura_max = altura

                if largura > largura_max:
                    largura_max = largura

                counter += 1     
                if counter >= data_set_size:
                    print("Counter: ", counter)
                    break

            except Exception as e:
                print(f"Erro ao processar o arquivo {filename}")
                print(f"Ocorreu um erro: {e}")


    if altura_max >= largura_max:
        return altura_max
    if altura_max < largura_max:
        return largura_max
    

def resize_matrix_zeros(dim, matrix):
    zero_matrix = np.zeros((dim, dim))
    altura, largura = matrix.shape[:2]
    pos_i = (dim - altura) // 2
    pos_j = (dim - largura) // 2

    matrix_crop = matrix[:min(altura, dim), :min(largura, dim)]

    pos_i = (dim - matrix_crop.shape[0]) // 2
    pos_j = (dim - matrix_crop.shape[1]) // 2

    zero_matrix[pos_i:pos_i + matrix_crop.shape[0], pos_j:pos_j + matrix_crop.shape[1]] = matrix_crop

    return zero_matrix


def next_pow_2(number):
    if (number & (number - 1)) == 0:
        return number

    most_sig_bit = 1
    while most_sig_bit < number:
        most_sig_bit <<= 1

    return most_sig_bit


def resize_matrix_zeros_pow_2(dim, matrix):
    new_dim = next_pow_2(dim)
    zero_matrix = np.zeros((new_dim, new_dim))
    altura, largura = matrix.shape[:2]
    pos_i = (new_dim - altura) // 2
    pos_j = (new_dim - largura) // 2

    zero_matrix[pos_i:pos_i + altura, pos_j:pos_j + largura] = matrix

    return zero_matrix
    
def fill_ones_row(matrix, coordinate1, coordinate2):
    if coordinate1[0] != coordinate2[0]:
        raise ValueError("The coordinates must be in the same row.")

    row = coordinate1[0]
    start_column, end_column = min(coordinate1[1], coordinate2[1]), max(coordinate1[1], coordinate2[1])

    matrix[row, start_column:end_column+1] = 1

def fill_ones_column(matrix, coordinate1, coordinate2):
    if coordinate1[1] != coordinate2[1]:
        raise ValueError("The coordinates must be in the same column.")

    column = coordinate1[1]
    row_start, row_end = min(coordinate1[0], coordinate2[0]), max(coordinate1[0], coordinate2[0])

    matrix[row_start:row_end+1, column] = 1

def create_spiral(width, height, space=1, mode = 'center'):
    spiral_matrix = np.zeros((height, width), dtype=int)
    row, col = spiral_matrix.shape

    central_row = row // 2
    central_col = col // 2

    if row % 2 == 0:
        central_row -= 1
    if col % 2 == 0:
        central_col -= 1

    spiral_matrix[central_row, central_col] = 1

    counter = 1

    while(counter < row and counter < col):
        try:
            # up
            central_row_new = central_row - counter
            spiral_matrix[central_row_new, central_col] = 1
            fill_ones_column(spiral_matrix, [central_row, central_col], [central_row_new, central_col])
            central_row = central_row_new
            counter += space

            # right
            central_col_new = central_col + counter
            spiral_matrix[central_row, central_col_new] = 1
            fill_ones_row(spiral_matrix, [central_row, central_col], [central_row, central_col_new])
            central_col = central_col_new
            counter += space

            # down
            central_row_new = central_row + counter
            spiral_matrix[central_row_new, central_col] = 1
            fill_ones_column(spiral_matrix, [central_row, central_col], [central_row_new, central_col])
            central_row = central_row_new
            counter += space

            # left
            central_col_new = central_col - counter
            spiral_matrix[central_row, central_col_new] = 1
            fill_ones_row(spiral_matrix, [central_row, central_col], [central_row, central_col_new])
            central_col = central_col_new
            counter += space
        except:
            break

    if mode == 'full':
        start = counter // 2
        counter = space + 1

        central_row = row // 2
        central_col = col // 2

        start_down = central_row + start

        start_up = central_row - start



        while (counter < row):
            try:
                spiral_matrix[start_down - 5 + counter, :] = 1

                if start_up - counter >= 0:
                    spiral_matrix[start_up - counter, :] = 1
                counter += (space*2)
                
            except:
                break

    return spiral_matrix

def crop_matrix(matrix):
    height, width = matrix.shape

    smaller_dimension = min(height, width)

    center_height = height // 2
    center_width = width // 2
    half_smaller_dimension = smaller_dimension // 2

    start_crop_height = center_height - half_smaller_dimension
    end_crop_height = start_crop_height + smaller_dimension

    start_crop_width = center_width - half_smaller_dimension
    end_crop_width = start_crop_width + smaller_dimension

    cropped_matrix = matrix[start_crop_height:end_crop_height, start_crop_width:end_crop_width]

    return cropped_matrix

def apply_undersampling_mask_spiral(dim, space = 1, spiral_mode_undersampling = 1, spiral_mode = 'center'):
    height = dim
    width = dim
    spiral_matrix = create_spiral(width, height, space, spiral_mode)

    if spiral_mode_undersampling == 1:
        spiral_matrix = ~spiral_matrix


    return spiral_matrix

def slices_in_volume(volume, n, offset, limit):
    if n < 1:
        raise ValueError("O inteiro deve ser maior ou igual a 1")
    
    length = len(volume)
    print("Slice length: ",length)
    if length < 20:
        raise ValueError("O vetor deve ter pelo menos 20 elementos")

    adjusted_length = length - limit  # Desconsidera os 10 primeiros e 10 últimos valores
    slices = adjusted_length / n
    slices_array = []

    for i in range(n):
        start_slice = int(offset + i * slices)  # Ajusta o índice inicial
        end_slice = int(offset + (i + 1) * slices)
        if end_slice > length - offset:
            end_slice = length - offset

        segment = volume[start_slice:end_slice]
        middle_slice = start_slice + len(segment) // 2
        slices_array.append(middle_slice)

    return slices_array

def k_space_percentage(mask, dim):
    matrix = np.ones((dim, dim))
    matrix_sum = matrix.sum()
    mask_sum = mask.sum()

    mask_sum = (mask_sum)*100/matrix_sum

    print("Total poupado do espaço-k: ", str(mask_sum) + '%')
    print("Total de sub-amostragem: ", str(100 - mask_sum) + '%')


def save_all(train_dataset, train_folder, test_folder, val_folder,mask = 'hf', 
             start_interval = 10, interval = 2, train_size = 100, test_size = 20, max_dim = 512, space = 1, 
             spiral_mode_undersampling = 1, spiral_mode = 'center', n_artifacts = 0, mask_percentage = 0.7, turns = 8, number_angles = 20, 
             val_size = 0,
             test_start_interval = 10, test_interval = 2, test_space = 1, test_mask_percentage = 0.7,
             test_turns = 8, test_number_angles = 20, offset = 15, show_mask = 0, limit = 20, show_k_space = 0):

    images_dataset = []
    train_dataset_array, test_dataset_array, val_dataset_array = [], [], []
    data_volume_size = count_files(train_dataset)
    mask_percentage = 1 - mask_percentage
    check = (train_size + val_size + test_size) // data_volume_size
    n_slice = check + 1
    data_set_size = train_size + test_size
    image_counter = 1
    input_folder = train_dataset
    output_folder = train_folder
    mode = 'train'

    file = os.listdir(train_dataset)

    file = [f for f in file if os.path.isfile(os.path.join(train_path, f))]


    if max_dim == -1 and filename.endswith(".h5"):
        max_dim = k_space_greater_dimension(input_folder, n_slice, data_set_size)
    print("Greater Dimension: ", max_dim)



    if mask == 'random':
        mask_matrix = apply_undersampling_mask_random(max_dim, mask_percentage)
    if mask == 'hf':
        mask_matrix = apply_undersampling_mask_hf(max_dim, start_interval, interval)
    if mask == 'spiral':
        mask_matrix = apply_undersampling_mask_spiral(dim = max_dim, space = space, 
                                                      spiral_mode_undersampling = spiral_mode_undersampling, 
                                                      spiral_mode =  spiral_mode)
    if mask == 'spiral4':
        k_row = max_dim
        k_col = max_dim
        mask_matrix = spiral_trajectory_example_4arms(rows = k_row, columns = k_col, turns = turns)
        mask_matrix = np.fft.ifftshift(mask_matrix)
        

    if mask == 'spiral1':
        k_row = max_dim
        k_col = max_dim
        mask_matrix = spiral_trajectory_example_1arm(rows = k_row, columns = k_col, turns = turns)
        mask_matrix = np.fft.ifftshift(mask_matrix)

    if mask == 'radial':
        k_row = max_dim
        k_col = max_dim
        mask_matrix = radial_lines_example(rows = k_row, columns = k_col, number_angles = number_angles)
        mask_matrix = np.fft.ifftshift(mask_matrix)
       
    if show_mask == 1:
        see_mask(mask_matrix)


    k_space_percentage(mask_matrix, max_dim)
    

    for filename in os.listdir(input_folder):
        if filename.endswith(".h5") or filename.endswith(".mat"):
            file_path = os.path.join(input_folder, filename)
            try:
                
                if filename.endswith(".h5"):
                    k_space_vol = open_single_file(file_path)
                    slices_array = slices_in_volume (k_space_vol, n_slice, offset, limit)

                if filename.endswith(".mat"):
                    mat_vol = open_single_mat(file_path)
                    slices_array = slices_in_volume (mat_vol, n_slice, offset, limit)


                print("Slice array: ",slices_array)

                for i in slices_array:
                    if filename.endswith(".h5"):
                        k_space = k_space_vol[i]
                        inversa = inverse_transform(k_space)

                    
                    if filename.endswith(".mat"):
                        inversa = mat_vol[i]
                    

                    inversa = resize_matrix_zeros(max_dim, inversa)
                    k_space = np.fft.ifftshift(direct_transform(inversa))

                    if show_k_space == 1:
                        see_k_space(k_space, "k-space")
                    # altura, largura = inversa.shape[:2]
                    # print("Altura Imagem: ", altura)
                    # print("Largura Imagem: ", largura)
            
                    if n_artifacts >= 1:
                        inversa = apply_artifacts(inversa, n_artifacts)
                        k_space = direct_transform(inversa)


                    if mask == 'random':
                        mask_matrix = apply_undersampling_mask_random(max_dim, mask_percentage)

                    sub_k_space = k_space * mask_matrix
                        
                    if show_k_space == 1:
                        see_k_space(sub_k_space, "Undersampled k-space")


                    inversa_sub = np.fft.ifftshift(inverse_transform(sub_k_space))

                
                    # if aplly_artifacts == 1:
                    #     inversa_sub = np.fft.ifftshift(inversa_sub)

                    # inversa_sub = resize_matrix_zeros(max_dim, inversa_sub)

                
                    
                
                    img_name = f"{image_counter}"
                    # print("Normal: ", np.sum(inversa))
                    # print("Sub: ", np.sum(inversa_sub))
                    # print("Total: ", np.sum(inversa) + np.sum(inversa_sub))
                    save_path = os.path.join(output_folder, f"{img_name}" + '.png')
                    combined_image = save_both(inversa, inversa_sub, save_path)
                    images_dataset.append(combined_image)

                    print(image_counter)
                    image_counter += 1
                    print(filename)

                    if  image_counter >= train_size + 1 and mode == 'train':
                        break
                    if  image_counter >= val_size + 1 and mode == 'val':
                        break
                    if  image_counter >= test_size + 1 and mode == 'test':
                        break
                
            except Exception as e:
                print(f"Erro ao processar o arquivo {filename}")
                print(f"Ocorreu um erro: {e}")
                traceback.print_exc()  # Imprime o traceback completo

            if  image_counter >= train_size + 1 and mode == 'train':
                train_dataset_array = images_dataset
                print("Acabou treino")
                images_dataset = []
                if val_size > 0:
                    mode = 'val'
                    image_counter = 1
                    output_folder = val_folder
                else:
                    image_counter = 1
                    output_folder = test_folder
                    mode = 'test'
                    interval = test_interval
                    start_interval = test_start_interval
                    space = test_space
                    mask_percentage = test_mask_percentage
                    turns = test_turns
                    number_angles = test_number_angles
                    

            if  image_counter >= val_size + 1 and mode == 'val':
                val_dataset_array = images_dataset
                print("Acabou validação")
                images_dataset = []
                image_counter = 1
                output_folder = test_folder
                mode = 'test'
                interval = test_interval
                start_interval = test_start_interval
                space = test_space
                mask_percentage = test_mask_percentage
                turns = test_turns
                number_angles = test_number_angles

            if  image_counter >= test_size + 1 and mode == 'test':
                image_counter = 1
                test_dataset_array = images_dataset
                print("Acabou teste")
                images_dataset = []
                return train_dataset_array, val_dataset_array, test_dataset_array

        


# def resize_image(image, target_height=256, target_width=256):
#     resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

#     return resized_image
                

def delete_folder_content(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        contents = os.listdir(folder_path)
        
        for content in contents:
            content_path = os.path.join(folder_path, content)
            if os.path.isfile(content_path):
                os.remove(content_path)


def apply_artifacts(image, n_artifacts = 1):
    
    height, width = image.shape[:2]
    
    result_image = np.copy(image)
    
    for _ in range(n_artifacts):
        center_x, center_y = np.random.randint(0, width), np.random.randint(0, height)
    
        artifact_radius = np.random.randint(10, 30)
    
        artifact_intensity = np.random.randint(50, 200)
        
        artifact_intensity = (artifact_intensity*max(max(row) for row in image)) / 150
    
        coord_x, coord_y = np.meshgrid(np.arange(width), np.arange(height))
    
        distance_to_center = np.sqrt((coord_x - center_x)**2 + (coord_y - center_y)**2)
        
        artifact_image = np.zeros((height, width), dtype=np.float64)
        
        artifact_image[distance_to_center <= artifact_radius] = artifact_intensity
    
        artifact_image[distance_to_center <= artifact_radius] = artifact_intensity
        
        result_image = result_image + artifact_image
        
    return result_image


def matrix2column(x):
    rows = x.shape[0]
    columns = x.shape[1]
    N = rows * columns
    y = np.zeros(shape = (N, ))
    y = y + 1j * y
    y[:, ] = np.asarray(np.reshape(x, (N, )))
    return y, rows, columns

def column2matrix(x, rows, columns):
    y = np.zeros(shape = (rows, columns))
    y = y + 1j * y
    x = np.reshape(x, [rows, columns])
    y[:, :] = x[:, :]
    return y

def radial_lines_samples_rows(rows, columns, angles):
    pi = np.pi
    I = np.zeros(shape=[rows * columns, 1])
    xmin = -float(columns) / 2.0
    xmax = float(columns) / 2.0
    ymin = -float(rows) / 2.0
    ymax = float(rows) / 2.0
    for k in range(0, angles.shape[0]):
        angle = angles[k]
        if (angle >= -pi / 4 and angle <= pi / 4) or \
                (angle >= 3 * pi / 4 and angle <= pi):
            x_ = np.linspace(xmin, xmax, columns)
            y_ = np.tan(angle) * x_
        else:
            y_ = np.linspace(ymin, ymax, rows)
            x_ = 1.0 / np.tan(angle) * y_
        i_ = np.round(-y_ - float(rows) / 2.0 + float(rows))
        j_ = np.round(x_ + float(columns) / 2.0)
        ii = np.where(np.logical_and(np.logical_and(
            np.greater_equal(i_, 0),
            np.less(i_, rows)),
            np.logical_and(np.greater_equal(j_, 0),
                           np.less(j_, columns))))
        i_ = i_[ii]
        j_ = j_[ii]
        indexes = list(i_ + j_ * rows)
        indexes = [int(ell) for ell in indexes]
        I[indexes, 0] = 1
    I = np.reshape(I, [rows, columns], order="F").copy()
    I = np.fft.ifftshift(I)
    I[0, 0] = 1
    i, j = np.where(I == 1)
    i = list(i)
    j = list(j)
    I = np.reshape(I, [rows * columns, 1], order="F").copy()
    samples_rows = list(np.where(I == 1))
    samples_rows = samples_rows[0]
    return samples_rows, i, j

def radial_lines_example(rows = 512, columns = 512, number_angles = 20):
    angles = np.linspace(0, np.pi, number_angles + 1)
    angles = angles[: number_angles]
    samples_rows, _, _ = radial_lines_samples_rows(rows, columns, angles)
    mask = np.zeros(shape=[rows * columns, 1])
    mask[samples_rows] = 1
    mask = np.real(column2matrix(mask, rows, columns))
    # plt.imshow(np.fft.fftshift(mask), cmap='gray')
    # plt.show()

    return mask

def spiral_trajectory_samples_rows(rows, columns, starting_angles = np.array([0]), numbers_turns = np.array([4]), r = np.linspace(0, 1, 1000000)):
    t = np.linspace(0, 1, len(r))
    x = np.zeros(shape = [len(starting_angles) * len(r), ]);
    y = np.zeros(shape = [len(starting_angles) * len(r), ]);
    n = 0
    for k in range (0, len(starting_angles)):
        a = starting_angles[k];
        turns = numbers_turns[k]
        x[n : n + len(r)] = np.cos(2 * np.pi * turns * t + a) * r
        y[n : n + len(r)] = np.sin(2 * np.pi * turns * t + a) * r
        n += len(r)
    x = (x / 2.0 + 0.5) * (columns - 1)
    y = (y / 2.0 + 0.5) * (rows - 1)
    i = np.round(rows - y)
    j = np.round(x)
    i = [int(i[n]) - 1 for n in range(0, len(i))]
    j = [int(j[n]) for n in range(0, len(j))]
    I = np.zeros(shape = (rows, columns))
    for k in range(0, len(i)):
        if i[k] >= 0 and i[k] < rows and j[k] >= 0 and j[k] < columns:
            I[i[k], j[k]] = 1
    I = np.fft.fftshift(I)
    I, _, _ = matrix2column(I)
    samples_rows = np.where(I > 0)[0]
    samples_rows = np.sort(samples_rows)
    return samples_rows, i, j

def spiral_trajectory_example_1arm(rows = 512, columns = 512, turns= 4):
    samples_rows, _, _ = \
    spiral_trajectory_samples_rows\
    (rows, columns, starting_angles = np.array([0]), numbers_turns = np.array([turns]), r = np.linspace(0, 1, 1000000))
    mask = np.zeros(shape=[rows * columns, 1])
    mask[samples_rows] = 1
    mask = np.reshape(mask, [rows, columns]).copy()
    # plt.imshow(np.fft.fftshift(mask), cmap='gray')
    # plt.show()

    return mask

def spiral_trajectory_example_4arms(rows = 512, columns = 512, turns = 8):
    samples_rows, _, _ = \
    spiral_trajectory_samples_rows\
    (rows, columns, starting_angles = np.array([0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]), numbers_turns = np.array([turns, turns, turns, turns]), r = np.linspace(0, 1, 1000000))
    mask = np.zeros(shape=[rows * columns, 1])
    mask[samples_rows] = 1
    mask = np.reshape(mask, [rows, columns]).copy()
    # plt.imshow(np.fft.fftshift(mask), cmap='gray')
    # plt.show()
    
    return mask



def SSIM(image1, image2):
  
    image1_gray = color.rgb2gray(image1)
    image2_gray = color.rgb2gray(image2)

    
    ssim = compare_ssim(image1_gray, image2_gray, data_range=image2_gray.max() - image2_gray.min())

    return ssim

def batch_similarity(folder):

    ssim_array = []

    for filename in os.listdir(folder):
        if filename.endswith('.jpg') and os.path.isfile(os.path.join(folder, filename)):
            file_path = os.path.join(folder, filename)
            image = io.imread(file_path)

            w = np.shape(image)[1]
            w = w // 2

            input_image = image[:, w:, :]
            real_image = image[:, :w, :]
            
            # plt.imshow(input_image)
            # plt.show() 
 
            ssim_v = SSIM(real_image, input_image)

            ssim_array.append(ssim_v)
  
    ssim_m = sum(ssim_array)/len(ssim_array)

    return ssim_m


def expand_dimensions(image_array):
    for n in range(len(image_array)): 
        if len(image_array[n].shape) == 2:
            image_array[n] = np.expand_dims(image_array[n], axis=-1)
            image_array[n] = np.repeat(image_array[n], 3, axis=-1)

    return image_array

def expand_to_rgb(image_array):
    if len(image_array.shape) == 2:  # Verifica se a imagem é grayscale (2D)
        image_array = np.expand_dims(image_array, axis=-1)  # Adiciona canal extra -> (256, 512, 1)
        image_array = np.repeat(image_array, 3, axis=-1)  # Repete para 3 canais -> (256, 512, 3)
    return image_array

def normalize_images(dataset_array):
    normalized_array = []
    for img in dataset_array:
        min_val = np.min(img)
        max_val = np.max(img)
        
        # Evita divisão por zero caso a imagem tenha valores constantes
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val) * 255
        
        # Converte para uint8 para exibição correta
        img = img.astype(np.uint8)
        
        normalized_array.append(img)
    
    return np.array(normalized_array)

if __name__ == '__main__':

    train_path = r'C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\GANs\cgans\dev\mricganspace_150\datasets\birn_anatomical_part\train'
    test_path = r'C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\GANs\cgans\dev\mricganspace_150\datasets\birn_anatomical_part\test'
    val_path = r'C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\GANs\cgans\dev\mricganspace_150\datasets\birn_anatomical_part\val'
    train_mat_path = r'C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\GANs\cgans\dev\mricganspace_150\datasets\birn_anatomical_part\DataSet'
    file_name = r'D:\singlecoil_train'
    
    train_dataset = r'D:\DataSet\singlecoil_train'
    train_mat_dataset = r'C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\GANs\cgans\dev\mricganspace_150\datasets\birn_anatomical_part\DataSet'
    
    
    delete_folder_content(train_path)
    
    delete_folder_content(test_path)

    delete_folder_content(val_path)
    
    train_dataset_array, val_dataset_array, test_dataset_array = save_all(train_dataset = train_dataset, train_folder = train_path, test_folder = test_path, val_folder = val_path, 
             mask ='spiral4', 
             start_interval = 8, interval = 1, train_size = 100,  val_size = 10, test_size = 20, max_dim = 512, space = 1, 
             spiral_mode_undersampling = 0, 
             spiral_mode = 'full', n_artifacts = 0, mask_percentage = 0.3, turns = 16, number_angles = 60,
             test_start_interval = 8, test_interval = 1, test_space = 1, test_mask_percentage = 0.5,
             test_turns = 16, test_number_angles = 32, show_mask = 0, show_k_space = 0)
    
    # print("SSIM treino: ", batch_similarity(train_path))
    # print("SSIM validação: ", batch_similarity(val_path))
    # print("SSIM teste: ", batch_similarity(test_path))

    print("Dataset train size: ", len(train_dataset_array))
    print("Dataset val size: ", len(val_dataset_array))
    print("Dataset test size: ", len(test_dataset_array))
    print("Train image size: ", np.array(train_dataset_array[0]).shape)
    print("Val image size: ", np.array(val_dataset_array[0]).shape)
    print("Test image size: ", np.array(test_dataset_array[0]).shape)

    print("Valores mínimos e máximos da imagem:", np.min(train_dataset_array[0]), np.max(train_dataset_array[0]))

    train_dataset_array = np.array(train_dataset_array).squeeze()
    val_dataset_array = np.array(val_dataset_array).squeeze()
    test_dataset_array = np.array(test_dataset_array).squeeze()

    print("Train image size: ", np.array(train_dataset_array[0]).shape)
    print("Val image size: ", np.array(val_dataset_array[0]).shape)
    print("Test image size: ", np.array(test_dataset_array[0]).shape)


    
    train_dataset_array = np.array([expand_to_rgb(img) for img in train_dataset_array])
    val_dataset_array = np.array([expand_to_rgb(img) for img in val_dataset_array])
    test_dataset_array = np.array([expand_to_rgb(img) for img in test_dataset_array])


    print("Valores mínimos e máximos da imagem:", np.min(train_dataset_array[0]), np.max(train_dataset_array[0]))

    print("train_dataset_array shape: ", train_dataset_array[0].shape)
    print("val_dataset_array shape: ", val_dataset_array[0].shape)
    print("test_dataset_array shape: ", test_dataset_array[0].shape)


    

    # image = np.array(train_dataset_array[0])  # Remove a dimensão extra
    # plt.imshow(image, cmap='gray')
    # plt.title("Imagem Processada")
    # plt.show()

    train_dataset_array = normalize_images(train_dataset_array)
    val_dataset_array = normalize_images(val_dataset_array)
    test_dataset_array = normalize_images(test_dataset_array)



import gc

tf.keras.backend.clear_session() 
gc.collect() 

mode = 'train'
save_checkpoint = 0

def count_files(folder):
    items = os.listdir(folder)

    files = [item for item in items if os.path.isfile(os.path.join(folder, item))]

    return len(files)

if mode == 'train' or mode == 'test':
    show_intermediate_images = False
    show_predicted_images_after_training = False


if mode == 'train' or mode == 'test':
    dataset_name = "birn_anatomical_part"

if mode == 'train' or mode == 'test':
    dataset_path = 'C:/Users/amaur/OneDrive/Área de Trabalho/Eletrônica/Mestrado/GANs/cgans/dev/mricganspace_150/datasets/' + dataset_name + '/'
    n_cont = count_files(dataset_path + 'train')
    print("Number of train images: ", n_cont)


if mode == 'train':
    sample_image = tf.convert_to_tensor(np.array(train_dataset_array[0]))
    sample_image = tf.cast(sample_image, tf.float32)
    print(sample_image.shape)

if mode == 'train':
    if show_intermediate_images:
        print("Valores mínimos e máximos da sample_image:", np.min(sample_image), np.max(sample_image))
        plt.figure()
        plt.imshow(sample_image / 255, cmap='grey')
        plt.title("Example of image")
        plt.show()



def load(image_array):
    # Se a imagem for 2D (grayscale), expande para 3D (fake RGB)

    # plt.figure()
    # plt.imshow(image_array, cmap='grey')
    # plt.title("opa")
    # plt.show()

    # print("Antes da conversão:")
    # print("Shape:", image_array.shape)
    # print("Valores únicos:", np.unique(image_array))

    # print("Depois da conversão para RGB:")
    # print("Shape:", image_array.shape)
    # print("Valores únicos:", np.unique(image_array))

    # plt.figure()
    # plt.imshow(image_array[:, :, 2], cmap='grey')
    # plt.title("opa2")
    # plt.show()

    

    w = image_array.shape[1]  # Largura da imagem
    w = w // 2  # Divide pela metade para separar

    # Separa a imagem real (esquerda) e a imagem input (direita)
    real_image = image_array[:, :w, :]
    input_image = image_array[:, w:, :]

    # Converte para float32 para garantir que está no formato adequado para o modelo
    input_image = tf.convert_to_tensor(input_image)
    real_image = tf.convert_to_tensor(real_image)
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)



    print("Formato tensor: ", np.shape(input_image))
    print("Formato tensor: ", type(input_image))

    return input_image, real_image

if mode == 'train' or mode == 'test':
    input1, output1 = load(train_dataset_array[0])
    IMG_HEIGHT, IMG_WIDTH = input1.shape[:2]
    print("Image Height: ", IMG_HEIGHT)
    print("Image Width: ", IMG_WIDTH)
    if show_intermediate_images:
        plt.figure()
        plt.imshow(input1)
        plt.title("Example of image to be used in the cGAN input")
        plt.show()
        plt.figure()
        plt.imshow(output1)
        plt.title("Example of image to be generated in the cGAN output")
        plt.show()

if mode == 'train':
    inp, re = load(train_dataset_array[1])
    print("Input Shape: ", inp.shape)
    if show_intermediate_images:
        # Casting to int for matplotlib to display the images
        plt.figure()
        plt.imshow(inp / 255, cmap='grey')
        plt.title("Example of image to be used in the cGAN input")
        plt.show()
        plt.figure()
        plt.imshow(re / 255, cmap='grey')
        plt.title("Example of image to be generated in the cGAN output")
        plt.show()

if mode == 'train' or mode == 'test':
    # The facade training set consist of 400 images
    BUFFER_SIZE = len(train_dataset_array)
    print("BUFFER_SIZE = ", BUFFER_SIZE)
    # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
    BATCH_SIZE = 1

# Esse conjunto de funções aumenta o tamanho das imagens e depois corta trechos da imagem aumentada do tamanho da imagem original, isso permite
#maior variedade de dados de treinamento inserindo leves variaçoes na imagem

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(\
    stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image[0], cropped_image[1]

# Faz um teste, abrindo a segunda imagem e realizando as opreações resize e random_crop
if mode == 'train':
    input1, output1 = load(np.array(train_dataset_array[0]))
    input1, output1 = resize(input1, output1, IMG_HEIGHT + 30, IMG_WIDTH + 30)
    input1, output1 = random_crop(input1, output1)
    if show_intermediate_images:
        plt.figure()
        plt.imshow(input1 / 255, cmap='grey')
        plt.title("Example of image to be used in the cGAN input")
        plt.show()
        plt.figure()
        plt.imshow(output1 / 255, cmap='grey')
        plt.title("Example of image to be generated in the cGAN output")
        plt.show()

# Normaliza os valores para ficarem entre -1 e 1
def normalize(input_image, real_image):


    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image



# Usa as funções resize e random_crop  para aumentar a variedade das imagens além de espelhar algumas imagens.
@tf.function()
def random_jitter(input_image, real_image):

    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT + 30, IMG_WIDTH + 30)
    input_image, real_image = random_crop(input_image, real_image)

    # Aleatoriamente espelha algumas imagens em volta do eixo vertical, aumentando a variedade e diminuindo overfitting

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    return input_image, real_image

if mode == 'train':
    data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                                            tf.keras.layers.RandomRotation(0.2),])

def augment_images(input, target):
    input = data_augmentation(input)
    target = data_augmentation(target)
    return input, target

# Visualiza a imagem após aplicar a função random_jitter
if mode == 'train':
    if show_intermediate_images:
        plt.figure(figsize=(6, 6))
        for i in range(4):
            rj_inp, rj_re = random_jitter(inp, re)
            plt.subplot(2, 2, i + 1)
            plt.imshow(rj_inp / 255.0)
            plt.axis('off')
        plt.show()

# As três funções abaixo carregam os conjuntos de dados de treino, teste e validação respectivamente, realizando as operações de random_jitter e normalização
def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    # input_image, real_image = augment_images(input_image, real_image) Rotacionar as imagens estava criando muito ruído e atrapalhando o treinamento pois as imagens sempre estão na mesma posição
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image

def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image

def load_image_val(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image

# Mapeia e separa os conjuntos em batches
if mode == 'train':
    # train_dataset = tf.data.Dataset.list_files(dataset_path + 'train/*.png')
    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset_array)
    train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    try:
        # val_dataset = tf.data.Dataset.list_files(dataset_path + 'val/*.png')
        val_dataset = tf.data.Dataset.from_tensor_slices(val_dataset_array)
    except tf.errors.InvalidArgumentError:
        print("ERROR ERROR ERROR ERROR ERROR ERROR")
    val_dataset = val_dataset.map(load_image_val, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)



if mode == 'train' or mode == 'test':
    try:
        # test_dataset = tf.data.Dataset.list_files(dataset_path + 'test/*.png')
        test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset_array)
    except tf.errors.InvalidArgumentError:
        print("ERROR")
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)



if mode == 'train':
    OUTPUT_CHANNELS = 3

# Função que cria o modelo downsample
def downsample(filters, size, apply_batchnorm = True, kernel_regularizer=None):

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()

    result.add(tf.keras.layers.Conv2D(filters, size, strides = 2, padding = 'same',
                                        kernel_initializer = initializer,
                                        kernel_regularizer = kernel_regularizer,
                                        use_bias = not apply_batchnorm))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    return result

# Apresenta uma imagem após aplicar o downsample

if mode == 'train':
    down_model = downsample(3, 4)
    down_result = down_model(tf.expand_dims(inp, 0))
    print(inp.shape)
    print(down_result.shape)
    print(np.min(inp))
    print(np.max(inp))
    print(np.min(down_result))
    print(np.max(down_result))
    x = down_result[0, :, :, :]
    x -= np.min(x)
    x /= np.max(x)
    print(np.min(inp))
    print(np.max(inp))
    print(np.min(x))
    print(np.max(x))
    if show_intermediate_images:
        plt.imshow(inp / 255.0)
        plt.imshow(x)

# Apresenta uma imagem que será usada como input da GAN, com a realização de operações de resize e random_crop
# Também apresenta uma imagem após a realização do downsample

if mode == 'train':
    input1, output1 = load(np.array(train_dataset_array[0]))
    input1, output1 = resize(input1, output1, IMG_HEIGHT + 30, IMG_WIDTH + 30)
    input1, output1 = random_crop(input1, output1)
    if show_intermediate_images:
        plt.figure()
        plt.imshow(input1 / 255.0)
        plt.title("Example of image to be used in the cGAN input")
        plt.show()
        input1, output1 = normalize(input1, output1)
        down_result = down_model(tf.expand_dims(input1, 0))
        # plt.figure()
        # plt.imshow((down_result + 1) / 2 * 255.0)
        # plt.title("Example of the image after the downsample model (before training)")
        # plt.show()
        print(input1.shape)
        print(down_result.shape)
        x = np.zeros(shape = (down_result.shape[1], down_result.shape[2], down_result.shape[3]))
        print(x.shape)
        x[:, :, :] = down_result[0, :, :, :]
        plt.figure()
        plt.imshow(x)
        plt.title("Example of the downsample module output (before training)")
        plt.show()

# Função que cria o modelo upsample

def upsample(filters, size, apply_dropout=False):

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()

    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                padding='same',
                kernel_initializer=initializer,
                use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())
    return result

# Cria um modelo upsample
if mode == 'train':
    up_model = upsample(3, 4)
    up_result = up_model(down_result)
    print (up_result.shape)

# Cria o gerador baseado em um modelo U-net com uma sequência de downsamples e upsamples que evidencia as características da imagem
def Generator():
    inputs = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])
    down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (batch_size, 128, 128, 64)
    downsample(128, 4), # (batch_size, 64, 64, 128)
    downsample(256, 4), # (batch_size, 32, 32, 256)
    downsample(512, 4), # (batch_size, 16, 16, 512)
    downsample(512, 4), # (batch_size, 8, 8, 512)
    downsample(512, 4), # (batch_size, 4, 4, 512)
    downsample(512, 4), # (batch_size, 2, 2, 512)
    downsample(512, 4), # (batch_size, 1, 1, 512)
    ]
    up_stack = [
    upsample(512, 4, apply_dropout=True), # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (batch_size, 8, 8, 1024)
    upsample(512, 4), # (batch_size, 16, 16, 1024)
    upsample(256, 4), # (batch_size, 32, 32, 512)
    upsample(128, 4), # (batch_size, 64, 64, 256)
    upsample(64, 4), # (batch_size, 128, 128, 128)
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
    strides=2,
    padding='same',
    kernel_initializer=initializer,
    activation='tanh') # (batch_size, 256, 256, 3)
    x = inputs
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1]) # should this be indented one more level?
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x) # should this be indented one more level?
    return tf.keras.Model(inputs=inputs, outputs=x)

# Cria uma visualização gráfica do modelo gerador
# if mode == 'train':
#     generator = Generator()
#     tf.keras.utils.plot_model(generator, to_file = 'generator.png', show_shapes=True, dpi=64)
#     if show_intermediate_images:
#         x = plt.imread('generator.png')
#         plt.imshow(x, cmap = 'gray')
#         plt.show()


# Aplica o modelo a uma imagem exemplo e apresenta o resultado
if mode == 'train':
    generator = Generator()
    generator.summary()
    gen_output = generator(inp[tf.newaxis, ...], training=False)
    if show_intermediate_images:
        plt.imshow(gen_output[0, ...])


if mode == 'train':
    plot_path = os.path.join(r"C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\GANs\cgans\dev\mricganspace_150\Plot", "unet_diagram.png")
    plot_model(generator, to_file=plot_path, show_shapes=True, expand_nested=True)

    print(f"Diagrama salvo em: {plot_path}")


if mode == 'train':
    LAMBDA = 150

if mode == 'train':
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# retorna os valores de loss do generator que são usados para ajustar a rede
def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

# Função que cria o modelo discriminador
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    # Define as entradas com base em IMG_HEIGHT e IMG_WIDTH
    inp = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3], name='target_image')

    # Calcula a diferença absoluta entre as imagens
    diff = tf.keras.layers.Subtract()([tar, inp])  # Calcula tar - inp

    # Concatena as imagens de entrada, alvo e a diferença
    x = tf.keras.layers.concatenate([inp, tar, diff])  # (batch_size, IMG_HEIGHT, IMG_WIDTH, 9)

    # Define o down_stack com dropout e regularização L2
    down1 = downsample(64, 4, False)(x) # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (batch_size, 32, 32, 256)

    # Zero padding para manter as dimensões ao aplicar convoluções finais
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
    kernel_initializer=initializer,
    use_bias=False)(zero_pad1) # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
    kernel_initializer=initializer)(zero_pad2) # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

# visualiza o modelo discriminador de forma gráfica
if mode == 'train':
    discriminator = Discriminator()
    discriminator.summary()
#     plot_path = os.path.join('/content/gdrive/MyDrive/Mestrado/Códigos/Plot/', "disc_diagram.png")
#     plot_model(discriminator, to_file=plot_path, show_shapes=True, expand_nested=True)

#     print(f"Diagrama salvo em: {plot_path}")


# Apresenta a saida do discriminador de forma gráfica
# if mode == 'train':
#     disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
#     if show_intermediate_images:
#         plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
#         plt.colorbar()

# Calcula a perda (loss) do modelo discriminador
def discriminator_loss(disc_real_output, disc_generated_output):
 real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

 generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

 total_disc_loss = real_loss + generated_loss

 return total_disc_loss

# Instância o otimizador Adam para os modelos

if mode == 'train':
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


# Cria um objeto que salva o estado dos modelos e dos otimizadores em checkpoints
if mode == 'train':
    checkpoint_dir = r"C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\GANs\cgans\dev\mricganspace_150\Checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)


    # latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    # if latest_checkpoint and os.path.exists(latest_checkpoint + ".index"):
    #     checkpoint.restore(latest_checkpoint).expect_partial()
    #     print(f"Checkpoint restored from: {latest_checkpoint}")
    # else:
    #     print("Checkpoint not found")

# Gera o conjunto de imagens resultado em uma única imagem

def generate_images(model, test_input, tar, save_path):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

def generate_images_file(model, test_input, tar, file_path, n, comp=0):
    prediction = model(test_input, training=True)

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        save_path = f"{file_path}_{title[i]}.png"
        img = display_list[i].numpy()  # Ajusta a escala para [0,1]
        img = np.clip(img, 0, 1)  # Garante que os valores estejam dentro de [0,1]
        mpimg.imsave(save_path, img, cmap='gray')

    if comp == 1:
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow(display_list[i])
            plt.axis('off')
            if i == 2:
                plt.savefig(file_path + '.png', dpi=300, bbox_inches = 'tight')

# if mode == 'train':
#     if show_intermediate_images:
#         for example_input, example_target in test_dataset.take(1):
#          generate_images(generator, example_input, example_target)

# Cria logs com detalhamento dos treinamentos
if mode == 'train':
    log_dir="logs/"
    summary_writer = tf.summary.create_file_writer(
     log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Realiza o treinamento do modelo
@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Gera a imagem de saída do gerador
        gen_output = generator(input_image, training=True)

        # Recebe os pares de imagem reais e falsos com os respectivos rótulos
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        # Calcula as perdas do gerador e do discriminador
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        # Calcula os gradientes e aplica a otimização
        generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    # Calcula a acurácia do discriminador para as imagens reais e geradas
    real_accuracy = tf.reduce_mean(tf.cast(disc_real_output > 0.5, tf.float32))
    fake_accuracy = tf.reduce_mean(tf.cast(disc_generated_output <= 0.5, tf.float32))
    disc_accuracy = (real_accuracy + fake_accuracy) / 2

    # Registra os resultados no TensorBoard
    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
        tf.summary.scalar('disc_loss', disc_loss, step=step//1000)
        tf.summary.scalar('disc_accuracy', disc_accuracy, step=step//1000)

    # Retorna as perdas e acurácias
    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss, disc_accuracy

# Realiza a validação do modelo
@tf.function
def val_step(input_image, target, step):
    # Gera a saída do gerador
    gen_output = generator(input_image, training=False)

    # Calcula as saídas do discriminador para as imagens reais e geradas
    disc_real_output = discriminator([input_image, target], training=False)
    disc_generated_output = discriminator([input_image, gen_output], training=False)

    # Calcula as perdas do gerador e do discriminador
    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    # Calcula a acurácia do discriminador para as imagens reais e geradas
    real_accuracy = tf.reduce_mean(tf.cast(disc_real_output > 0.5, tf.float32))
    fake_accuracy = tf.reduce_mean(tf.cast(disc_generated_output <= 0.5, tf.float32))
    disc_accuracy = (real_accuracy + fake_accuracy) / 2

    # Registra os resultados no TensorBoard
    with summary_writer.as_default():
        tf.summary.scalar('val_gen_total_loss', gen_total_loss, step=step//1000)
        tf.summary.scalar('val_gen_gan_loss', gen_gan_loss, step=step//1000)
        tf.summary.scalar('val_gen_l1_loss', gen_l1_loss, step=step//1000)
        tf.summary.scalar('val_disc_loss', disc_loss, step=step//1000)
        tf.summary.scalar('val_disc_accuracy', disc_accuracy, step=step//1000)

    # Retorna as perdas e acurácias
    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss, disc_accuracy

# Apaga checkpoints antigos caso um novo seja criado

def remove_old_checkpoints(checkpoint_dir, keep_latest=1):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "ckpt-*"))
    checkpoints.sort()


    for checkpoint in checkpoints[:-keep_latest]:
        os.remove(checkpoint)
        print(f"Checkpoint removed: {checkpoint}")

import time
import tensorflow as tf

def fit(train_ds, val_ds, steps):
    start = time.time()

    # Listas para armazenar métricas de treinamento
    gen_total_loss_list = []
    gen_gan_loss_list = []
    gen_l1_loss_list = []
    disc_loss_list = []
    disc_accuracy_list = []

    # Listas para armazenar métricas de validação (serão atualizadas apenas a cada 100 épocas)
    val_gen_total_loss_list = []
    val_gen_gan_loss_list = []
    val_gen_l1_loss_list = []
    val_disc_loss_list = []
    val_disc_accuracy_list = []

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    if latest_checkpoint:
        print(f"Checkpoint encontrado: {latest_checkpoint}")
    else:
        print("Nenhum checkpoint encontrado. O treinamento será iniciado do zero.")

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        # Realiza o treinamento normal
        gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss, disc_accuracy = train_step(input_image, target, step)

        # Salva métricas de treinamento
        gen_total_loss_list.append(gen_total_loss.numpy())
        gen_gan_loss_list.append(gen_gan_loss.numpy())
        gen_l1_loss_list.append(gen_l1_loss.numpy())
        disc_loss_list.append(disc_loss.numpy())
        disc_accuracy_list.append(disc_accuracy.numpy())

        # Executa a validação APENAS a cada 100 épocas
        if (step + 1) % 100 == 0:
            val_gen_total_loss, val_gen_gan_loss, val_gen_l1_loss, val_disc_loss, val_disc_accuracy = 0, 0, 0, 0, 0
            val_steps = 0

            for val_input_image, val_target in val_ds:
                v_gen_total_loss, v_gen_gan_loss, v_gen_l1_loss, v_disc_loss, v_disc_accuracy = val_step(val_input_image, val_target, step)
                # print(v_gen_total_loss)
                val_gen_total_loss += v_gen_total_loss
                val_gen_gan_loss += v_gen_gan_loss
                val_gen_l1_loss += v_gen_l1_loss
                val_disc_loss += v_disc_loss
                val_disc_accuracy += v_disc_accuracy
                val_steps += 1

            # Calcula médias das métricas de validação
            val_gen_total_loss /= val_steps
            val_gen_gan_loss /= val_steps
            val_gen_l1_loss /= val_steps
            val_disc_loss /= val_steps
            val_disc_accuracy /= val_steps

            # Adiciona os valores médios às listas de validação
            val_gen_total_loss_list.append(val_gen_total_loss.numpy())
            val_gen_gan_loss_list.append(val_gen_gan_loss.numpy())
            val_gen_l1_loss_list.append(val_gen_l1_loss.numpy())
            val_disc_loss_list.append(val_disc_loss.numpy())
            val_disc_accuracy_list.append(val_disc_accuracy.numpy())

            print(f"Validação na época {step + 1}: Gen Loss {val_gen_total_loss.numpy():.4f}, Disc Loss {val_disc_loss.numpy():.4f}")

        print(f"{step.numpy()}/{steps - 1}")

        # Salva o checkpoint no final do treinamento
        if step + 1 == int(steps * 1) and save_checkpoint == 1:
            remove_old_checkpoints(checkpoint_dir, keep_latest=1)
            checkpoint.save(file_prefix=checkpoint_prefix)
            print("Checkpoint Salvo!")

    return {
        "steps": steps,
        "train_metrics": {
            "gen_total_loss": gen_total_loss_list,
            "gen_gan_loss": gen_gan_loss_list,
            "gen_l1_loss": gen_l1_loss_list,
            "disc_loss": disc_loss_list,
            "disc_accuracy": disc_accuracy_list,
        },
        "val_metrics": {
            "gen_total_loss": gen_total_loss_list,
            "gen_gan_loss": gen_gan_loss_list,
            "gen_l1_loss": gen_l1_loss_list,
            "disc_loss": disc_loss_list,
            "disc_accuracy": disc_accuracy_list,
        }
    }

# Apresenta os tipos de cada um dos conjuntos de dados
if mode == 'train':
    print("BUFFER_SIZE = ", BUFFER_SIZE)
    print("Type train_dataset: ", type(train_dataset))
    print("Shape train_dataset: ", np.shape(train_dataset))
    print("Type test_dataset: ", type(test_dataset))
    print("Shape test_dataset: ", np.shape(test_dataset))
    print("Type val_dataset: ", type(val_dataset))
    print("Shape val_dataset: ", np.shape(val_dataset))

# Aplica o modelo as conjuntos de treino e validação obtendo as métricas de qualidade
if mode == 'train':
        print('Starting the training stage.')
        epochs = 3000




        ##################
        result = fit(train_dataset, val_dataset, steps=epochs)

        # steps, gen_total_loss_list, gen_gan_loss_list, gen_l1_loss_list, disc_loss_list, disc_accuracy_list, val_gen_total_loss_list, val_gen_gan_loss_list, val_gen_l1_loss_list, val_disc_loss_list, val_disc_accuracy_list = fit(train_dataset, val_dataset, steps=epochs)
        ##################
        print('Training finished.')

        gen_total_loss_list = result["train_metrics"]["gen_total_loss"]
        gen_gan_loss_list = result["train_metrics"]["gen_gan_loss"]
        gen_l1_loss_list = result["train_metrics"]["gen_l1_loss"]
        disc_loss_list = result["train_metrics"]["disc_loss"]
        disc_accuracy_list = result["train_metrics"]["disc_accuracy"]

        # Métricas de validação
        val_gen_total_loss_list = result["val_metrics"]["gen_total_loss"]
        val_gen_gan_loss_list = result["val_metrics"]["gen_gan_loss"]
        val_gen_l1_loss_list = result["val_metrics"]["gen_l1_loss"]
        val_disc_loss_list = result["val_metrics"]["disc_loss"]
        val_disc_accuracy_list = result["val_metrics"]["disc_accuracy"]




# if mode == 'train':
#     # Restoring the latest checkpoint in checkpoint_dir
#     checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Salva o modelo

if mode == 'train':
    generator.save(r'C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\GANs\cgans\dev\mricganspace_150\Modelo\generator_model.keras')

# Salva as iamgens de qualquer plot
def save_plot(x, y_train, y_val, xlabel, ylabel, title, filename, label1, label2):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_train, label=label1)
    plt.plot(x, y_val, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(plot_folder, filename), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def pad_with_zeros(arr1, arr2):
    len1, len2 = len(arr1), len(arr2)

    if len1 >= len2:
        return arr1

    # Criar um array de saída preenchido com zeros
    new_arr = np.zeros(len2, dtype=int)

    # Determinar os índices onde os valores de arr1 serão inseridos
    insert_positions = np.linspace(0, len2 - 1, num=len1, dtype=int)

    # Inserir os valores de arr1 nas posições calculadas
    for i, pos in enumerate(insert_positions):
        new_arr[pos] = arr1[i]

    return new_arr

# Salva as imagens dos plots de loss para o gerador e para o discriminador

if mode == 'train':

    plot_folder = r"C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\GANs\cgans\dev\mricganspace_150\Plot"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # Imprime informações de debug
    print("type gen: ", type(gen_total_loss_list))
    print("type disc: ", type(disc_loss_list))
    print("type val: ", type(val_gen_total_loss_list))
    print("size gen: ", len(gen_total_loss_list))
    print("size disc: ", len(disc_loss_list))
    print("size val: ", len(val_gen_total_loss_list))
    print(val_gen_total_loss_list)

    # Define os steps
    steps = list(range(len(gen_total_loss_list)))

    val_gen_total_loss_list = pad_with_zeros(val_gen_total_loss_list, gen_total_loss_list)
    val_disc_loss_list = pad_with_zeros(val_disc_loss_list, disc_loss_list)

    print("Ajuste no tamanho")
    print("type gen: ", type(gen_total_loss_list))
    print("type disc: ", type(disc_loss_list))
    print("type val: ", type(val_gen_total_loss_list))
    print("size gen: ", len(gen_total_loss_list))
    print("size disc: ", len(disc_loss_list))
    print("size val: ", len(val_gen_total_loss_list))
    print(val_gen_total_loss_list)

    # Salva gráficos
    save_plot(steps, gen_total_loss_list, val_gen_total_loss_list, 'Step', 'Loss', 'Generator Total Loss', 'Generator_Total_Loss.png', 'Treinamento', 'Validação')
    save_plot(steps, disc_loss_list, val_disc_loss_list, 'Step', 'Loss', 'Discriminator Loss', 'Discriminator_Loss.png', 'Treinamento', 'Validação')
    save_plot(steps, gen_total_loss_list, disc_loss_list, 'Step', 'Loss', 'GenxDisc Loss', 'GenxDisc_Loss.png', 'Generator Loss', 'Discriminator Loss')
    plt.tight_layout()

    # Salva dados em CSV
    data = {
        'Step': steps,
        'Generator Total Loss': gen_total_loss_list,
        'Validation Generator Total Loss': val_gen_total_loss_list,
        'Discriminator Loss': disc_loss_list,
        'Validation Discriminator Loss': val_disc_loss_list,
    }

    df = pd.DataFrame(data)

    # Caminho do arquivo CSV
    csv_file = os.path.join(plot_folder, 'training_metrics.csv')

    # Salva o DataFrame no arquivo CSV
    df.to_csv(csv_file, index=False)

# Deleta os conteúdos de uma pasta


def delete_folder_content(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        contents = os.listdir(folder_path)

        for content in contents:
            content_path = os.path.join(folder_path, content)
            if os.path.isfile(content_path):
                os.remove(content_path)

# Carrega o modelo

model_save_path = r'C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\GANs\cgans\dev\mricganspace_150\Modelo\generator_model.keras'

if mode == 'test':
    from tensorflow.keras.models import load_model
    generator = load_model(model_save_path)

# Uso o conjunto de teste para observar o funcionamento do modelo e salva essas imagens em uma pasta


# if mode == 'train' or mode == 'test':

#     k = 0

#     n_test = len(test_dataset_array)

#     print(n_test)

#     # n_test = 10


#     print("Start model test")
#     image_folder = r"C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\GANs\cgans\dev\mricganspace_150\Generated Images"
#     delete_folder_content(image_folder)
#     file_path = r"C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\GANs\cgans\dev\mricganspace_150\Generated Images\Image"
#     for inp, tar in test_dataset.take(n_test):
#         generate_images_file(generator, inp, tar, file_path + str(k) + '_', k, comp = 0)
#         k += 1
#         if show_predicted_images_after_training:
#             generate_images(generator, inp, tar, file_path)
#     print("Model test finished")




def generate_images_file_loop(model, test_input, tar, file_name, n, input_images, gt_images, pred_images, comp=0):
    prediction = model(test_input, training=True)

    # Lista para armazenar as imagens de cada batch
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    # Salvando as imagens nos arrays separados
    for i in range(3):
        image_array = display_list[i].numpy()

        # Se a imagem for entre 0 e 1 (como float), leve para o intervalo [0, 1]
        if image_array.min() < 0:  # Verifica se a imagem tem valores negativos
            image_array = (image_array + 1) / 2  # Transforma para o intervalo [0, 1]

        # Adiciona as imagens aos arrays correspondentes
        if i == 0:
            input_images.append(image_array)
        elif i == 1:
            gt_images.append(image_array)
        elif i == 2:
            pred_images.append(image_array)

    # # Exibe uma das imagens (opcional)
    # plt.imshow(input_images[0], cmap='gray')
    # plt.title('Exemplo de Input Image')
    # plt.show()

    # Não é necessário salvar as imagens aqui, pois elas estão sendo acumuladas nos arrays
    return input_images, gt_images, pred_images







# Uso o conjunto de teste para observar o funcionamento do modelo e salva essas imagens em uma pasta

def test_model(image_folder, file_path, test_dataset, ite):
    if mode == 'train' or mode == 'test':
        k = 0


        print("Type test_dataset: ", type(test_dataset))
        print("Shape test_dataset: ", np.shape(test_dataset))

        

        n_test = len(test_dataset_array)

        # delete_folder_content(image_folder)

        input_images = []
        gt_images = []
        pred_images = []

        # Itera sobre o dataset de teste
        
        for inp, tar in test_dataset.take(n_test):
            input_images, gt_images, pred_images = generate_images_file_loop(generator, inp, tar, file_path + str(k) + '_', k,
                                                                    input_images, gt_images, pred_images)   

              
                
            k += 1
               

            if show_predicted_images_after_training:
                generate_images(generator, inp, tar, file_path)
    


    print("Teste Finalizado!")
    return input_images, gt_images, pred_images


# def save_both(image1, image2, save_path):
#     combined_image = np.concatenate((image1, image2), axis=1)
#     plt.imshow(combined_image, cmap='gray')
#     plt.axis('off')
#     mpimg.imsave(save_path, combined_image, cmap='gray')


def k_space_merge(gt_images, pred_images, input_images, turns=16):

    k_row, k_col = 512, 512  # Dimensão do espaço k

    mask_matrix = spiral_trajectory_example_4arms(rows=k_row, columns=k_col, turns=turns)
    mask_matrix = np.fft.ifftshift(mask_matrix)

    merged_images = []  # Lista para armazenar as imagens reconstruídas

    for i in range(len(gt_images)):
        imagegt = np.mean(gt_images[i], axis=-1)
        imagepr = np.mean(pred_images[i], axis=-1)
        imagein = np.mean(input_images[i], axis=-1)

        # Transformada direta
        k_spacein = np.fft.ifftshift(direct_transform(imagein))
        k_spacein = k_spacein * mask_matrix  # Aplicação da máscara

        k_spacepr = np.fft.ifftshift(direct_transform(imagepr))

        # Combinação no espaço k
        merged_k = np.where(k_spacein != 0, k_spacein, k_spacepr)

        # Transformada inversa para reconstruir a imagem
        new_img = np.fft.ifftshift(inverse_transform(merged_k))

        combined_image = np.concatenate((imagegt, new_img), axis=1)


        merged_images.append(combined_image)  # Armazena a imagem processada



    return merged_images, gt_img, new_img, input_img
  # Retorna todas as imagens processadas



def SSIM_loop(gt_images, pred_images, input_images): 
    ssim_pr_array, ssim_in_array = [], []

    n = len(gt_images)  # Número de imagens

    for i in range(n):
        # Converte para arrays NumPy se necessário e verifica a consistência da forma
        image1 = np.array(gt_images[i])
        image2 = np.array(pred_images[i])
        image3 = np.array(input_images[i])

        # Verifica se as imagens têm o mesmo formato
        if image1.shape != image2.shape or image1.shape != image3.shape:
            print(f"Erro: As imagens têm formas diferentes no índice {i}.")
            continue  # Salta a iteração caso as imagens não tenham a mesma forma

        # Calcula o SSIM entre as imagens
        ssim_pr_array.append(SSIM(image1, image2))
        ssim_in_array.append(SSIM(image1, image3))

    # Exibe os resultados de SSIM
    print(f"SSIM input: {np.mean(ssim_in_array):.4f} ± {np.std(ssim_in_array):.4f}")
    print(f"SSIM predicted: {np.mean(ssim_pr_array):.4f} ± {np.std(ssim_pr_array):.4f}")


# def load_image(image_path):
#     """Carrega uma imagem PNG preservando seus canais e converte para tons de cinza corretamente."""
#     image = io.imread(image_path, as_gray=False)  # Carrega a imagem mantendo os canais de cor

#     if image.shape[-1] == 4:  # Se a imagem tem 4 canais (RGBA)
#         image = image[:, :, :3]  # Remove o canal alfa

#     # Faz a média dos canais para converter para cinza sem distorções
#     grayscale_image = np.mean(image, axis=-1)

#     return grayscale_image

def SSIM(image1, image2):
    """Calcula o SSIM entre duas imagens convertidas corretamente para escala de cinza."""
    image1_gray = np.mean(image1, axis=-1) if image1.ndim == 3 else image1
    image2_gray = np.mean(image2, axis=-1) if image2.ndim == 3 else image2

    return compare_ssim(image1_gray, image2_gray, data_range=image2_gray.max() - image2_gray.min())



def load_npy(image_array):
    w = image_array.shape[1]  # Largura da imagem
    w = w // 2  # Divide pela metade para separar

    # Separa a imagem real (esquerda) e a imagem input (direita)
    real_image = image_array[:, :w, :]
    input_image = image_array[:, w:, :]

    # Converte para float32 para garantir que está no formato adequado para o modelo
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

def normalize_npy(input_image, real_image):
    def normalize_single(img):
        min_val = tf.reduce_min(img)
        max_val = tf.reduce_max(img)
        
        # Evita divisão por zero caso a imagem tenha valores constantes
        img = tf.cond(
            tf.math.greater(max_val, min_val),
            lambda: (img - min_val) / (max_val - min_val),
            lambda: img  # Se max == min, mantém a imagem como está
        )
        
        # Converte para uint8 para exibição correta
        img = tf.cast(img, tf.uint8)

        return img

    input_image = normalize_single(input_image)
    real_image = normalize_single(real_image)

    return input_image, real_image


def load_image_test_npy(image_file):
    input_image, real_image = load_npy(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    # input_image, real_image = normalize_npy(input_image, real_image)
    return input_image, real_image

i = 5


image_folder = '/content/gdrive/MyDrive/Mestrado/Códigos/Generated Images'
file_path = '/content/gdrive/MyDrive/Mestrado/Códigos/Generated Images/Image'
merge_path = '/content/gdrive/MyDrive/Mestrado/Códigos/Temp'

for n in range(i):
    print("Loop: ", n)

    input_img, gt_img, predicted_img = 0, 0, 0

    # delete_folder_content(image_folder) #deleto o conteúdo da pasta de imagens geradas

    input_img, gt_img, predicted_img = test_model(image_folder, file_path, test_dataset, n) #testo o modelo e gero as imagens

    print("Dataset input size: ", len(input_img))
    print("Dataset gt size: ", len(gt_img))
    print("Dataset pr size: ", len(predicted_img))
    print("Input image size: ", input_img[0].shape)
    print("Gt image size: ", gt_img[0].shape)
    print("Pr image size: ", predicted_img[0].shape)


    SSIM_loop(gt_img, predicted_img, input_img) #checo as métricas das imagens geradas

    # delete_folder_content(dataset_path + 'test') #deleto o conteúdo da pasta de teste



    plt.imshow(input_img[0], cmap='gray')  # Exibe a primeira imagem
    plt.title("Imagem Entrada")  # Define um título
    plt.show()  # Mostra a imagem
    print("input_img Images Min:", input_img[0].min(), "Max:", input_img[0].max())

    plt.imshow(gt_img[0], cmap='gray')  # Exibe a primeira imagem
    plt.title("Imagem GT")  # Define um título
    plt.show()  # Mostra a imagem
    print("gt_img Images Min:", gt_img[0].min(), "Max:", gt_img[0].max())

    plt.imshow(predicted_img[0], cmap='gray')  # Exibe a primeira imagem
    plt.title("Imagem Gerada")  # Define um título
    plt.show()  # Mostra a imagem
    print("predicted_img Images Min:", predicted_img[0].min(), "Max:", predicted_img[0].max())

    merged_images, sgt_img, new_img, sinput_img = k_space_merge(gt_img, predicted_img, input_img, turns=16) #crio o novo conteúdo de testes a partir das imagens geradas


    merged_images = np.expand_dims(merged_images, axis=-1)
    if merged_images.shape[-1] == 1:
        merged_images = np.repeat(merged_images, 3, axis=-1)


    merged_images = np.array(merged_images, dtype=np.float32)


    print("Merged Images Min:", merged_images.min(), "Max:", merged_images.max())
    print("Dataset merged size: ", len(predicted_img))
    print("Merged image size: ", input_img[0].shape)

    plt.imshow(merged_images[0], cmap='gray')  # Exibe a primeira imagem
    plt.title("Imagem Concatenada")  # Define um título
    plt.show()  # Mostra a imagem

    test_dataset = 0

    test_dataset = tf.data.Dataset.from_tensor_slices(merged_images)

    for data in test_dataset.take(1):
        if isinstance(data, tuple):
            image = data[0]
        else:
            image = data

        img_array = image.numpy()
        break

    # Checar valores mínimos e máximos


    test_dataset = test_dataset.map(load_image_test_npy)

    for data in test_dataset.take(1):
        if isinstance(data, tuple):
            image = data[0]
        else:
            image = data

        img_array = image.numpy()
        break

    # Checar valores mínimos e máximos

    test_dataset = test_dataset.batch(BATCH_SIZE)

    for data in test_dataset.take(1):
        if isinstance(data, tuple):
            image = data[0]
        else:
            image = data

        img_array = image.numpy()
        break

    # Checar valores mínimos e máximos

    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    dataset_size = tf.data.experimental.cardinality(test_dataset).numpy()
    print("Tamanho do dataset:", dataset_size)

    for data in test_dataset.take(1):  # Pega apenas 1 batch
        if isinstance(data, tuple):
            input_image, gt_image = data  # As duas imagens do dataset
        else:
            input_image = data  # Caso não seja uma tupla
            gt_image = None

    # Converte para numpy se necessário
        input_image = input_image.numpy()
        gt_image = gt_image.numpy()

    # Exibe a imagem de entrada (input) e a imagem real (ground truth)
        plt.figure(figsize=(10, 5))

    # Exibe a primeira imagem (Input)
        # plt.subplot(1, 2, 1)
        # plt.imshow(input_image[0], cmap='gray')  # Ajuste conforme o formato das suas imagens
        # plt.title("Imagem de Entrada (Input)")
        # plt.axis('off')

    # Exibe a segunda imagem (Ground Truth)
        # plt.subplot(1, 2, 2)
        # plt.imshow(gt_image[0], cmap='gray')  # Ajuste conforme o formato das suas imagens
        # plt.title("Imagem Real (Ground Truth)")
        # plt.axis('off')

        # plt.show()

    # Checar valores mínimos e máximos
    print("Shape:", img_array.shape)



    # Exibir a imagem

    # if mode == 'train' or mode == 'test': #crio o novo dataset de teste
    #     test_dataset = 0
    # try:
    #     test_dataset = tf.data.Dataset.list_files(dataset_path + 'test/*.png')
    # except tf.errors.InvalidArgumentError:
    #     print("ERROR")
    # test_dataset = test_dataset.map(load_image_test)
    # test_dataset = test_dataset.batch(BATCH_SIZE)
    # test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)





    # for arquivo in os.listdir(merge_path):
    #     if arquivo.endswith(".png"):  # Filtra apenas arquivos PNG
    #         shutil.copy2(os.path.join(merge_path, arquivo), os.path.join(image_folder, arquivo))





# Uso o conjunto de teste para observar o funcionamento do modelo e salva essas imagens em uma pasta


if mode == 'train' or mode == 'test':

    k = 0

    n_test = len(test_dataset_array)

    print(n_test)

    # n_test = 10


    print("Start model test")
    image_folder = r"C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\GANs\cgans\dev\mricganspace_150\Generated Images"
    delete_folder_content(image_folder)
    file_path = r"C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\GANs\cgans\dev\mricganspace_150\Generated Images\Image"
    for inp, tar in test_dataset.take(n_test):
        generate_images_file(generator, inp, tar, file_path + str(k) + '_', k, comp = 0)
        k += 1
        if show_predicted_images_after_training:
            generate_images(generator, inp, tar, file_path)
    print("Model test finished")
