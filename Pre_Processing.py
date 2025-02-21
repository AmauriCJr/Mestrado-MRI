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
    plt.imshow(combined_image, cmap='gray')
    plt.axis('off')
    mpimg.imsave(save_path, combined_image, cmap='gray')


# def save_both(image1, image2, save_path):
#     # print(image1.dtype)
#     # print(image2.dtype)
#     # Combine as duas imagens
#     combined_image = np.concatenate((image1, image2), axis=1)
#     # print("Conc: ", np.sum(combined_image))
#     # print(combined_image.dtype)
#     # Verifique o tipo de dado e converta para float64, para garantir alta precisão
#     combined_image = combined_image.astype(np.float64)  # Garantir precisão

#     # Salve a imagem combinada como um arquivo .npy
#     np.save(save_path, combined_image)


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


def low_f_percentage(mask, max_dim):
    n = 20
    num_total_colunas = mask.shape[1]
    
    if n > num_total_colunas:
        raise ValueError("O número de colunas centrais não pode ser maior que o total de colunas da matriz.")
    
    low_f_mask = apply_undersampling_mask_hf(max_dim, n, 1)

    mask = low_f_mask * mask

    soma = np.sum(mask)
    total = num_total_colunas * num_total_colunas

    percentage = (soma * 100) / total if total > 0 else 0  # Evita divisão por zero
    
    print(f"Total poupado nas baixas frequências: {percentage:.2f}%")



def save_all(train_dataset, train_folder, test_folder, val_folder,mask = 'hf', 
             start_interval = 10, interval = 2, train_size = 100, test_size = 20, max_dim = 512, space = 1, 
             spiral_mode_undersampling = 1, spiral_mode = 'center', n_artifacts = 0, mask_percentage = 0.7, turns = 8, number_angles = 20, 
             val_size = 0,
             test_start_interval = 10, test_interval = 2, test_space = 1, test_mask_percentage = 0.7,
             test_turns = 8, test_number_angles = 20, offset = 15, show_mask = 0, limit = 20, show_k_space = 0):


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
    
    low_f_percentage(mask_matrix, max_dim)

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
                    save_both(inversa, inversa_sub, save_path)
                

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
                print("Acabou treino")
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
                print("Acabou validação")
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
                print("Acabou teste")
                break

        


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
    
    save_all(train_dataset = train_dataset, train_folder = train_path, test_folder = test_path, val_folder = val_path, 
             mask ='spiral4', 
             start_interval = 10, interval = 2, train_size = 1,  val_size = 0, test_size = 0, max_dim = 512, space = 1, 
             spiral_mode_undersampling = 0, 
             spiral_mode = 'full', n_artifacts = 0, mask_percentage = 0.3, turns = 16, number_angles = 60,
             test_start_interval = 8, test_interval = 1, test_space = 1, test_mask_percentage = 0.5,
             test_turns = 16, test_number_angles = 32, show_mask = 1, show_k_space = 0)
    
    # print("SSIM treino: ", batch_similarity(train_path))
    # print("SSIM validação: ", batch_similarity(val_path))
    # print("SSIM teste: ", batch_similarity(test_path))
