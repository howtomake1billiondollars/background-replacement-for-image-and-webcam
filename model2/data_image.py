import os
from glob import glob
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

is_sequence = False

# load ảnh vào
# input là gồm 2 thư mục: ảnh và mask
def load_data():
    train_path = 'D:\\Real-time-Background-replacement-in-Video-Conferencing\\people_segmentation'

    images = sorted(glob(os.path.join(train_path, "\\image\\*")))
    masks = sorted(glob(os.path.join(train_path, "\\mask\\*")))

    # chia tập thành tập train : val = 8:2
    train_x, valid_x = train_test_split(images, test_size=0.2, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=0.2, random_state=42)
    return (train_x, train_y), (valid_x, valid_y)

# trộn các dữ liệu đầu vào và mục tiêu một cách ngẫu nhiên
def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

# thay đổi kích thước thành 256x256 pixel
# sau đó chia giá trị pixel cho 255 để nó nằm trong khoảng [0,1]
# chuyển sang kiểu dữ liệu float32
def read_image(images_path):
    x = cv2.imread(images_path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    x = x.astype(np.float32)
    # nếu true == dữ liệu được xử lý dưới dạng chuỗi (sequence)
    # thì tăng thêm 1 chiều nữa
    if is_sequence:
        x = np.expand_dims(x, axis=0)
    return x



def read_mask(masks_path):
    x = cv2.imread(masks_path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1) # thêm 1 chiều nữa để chứa kênh màu
    if is_sequence:
        x = np.expand_dims(x, axis=0)
    return x


def preprocess(image_path, mask_path):
    def f(img_path, msk_path):
        x = read_image(img_path.decode())
        y = read_mask(msk_path.decode())
        return x, y

    image, mask = tf.numpy_function(f, [image_path, mask_path], [tf.float32, tf.float32])
    if is_sequence:
        image.set_shape([1, 256, 256, 3]) # image được mở rộng từ [256, 256, 3] thành [1, 256, 256, 3]
        mask.set_shape([1, 256, 256, 1]) # mask từ [256, 256, 1] thành [1, 256, 256, 1]
    else:
        image.set_shape([256, 256, 3])
        mask.set_shape([256, 256, 1])
    return image, mask
    # trả về các tensor image và mask
    # đã được tiền xử lý và chuẩn bị để sử dụng trong quá trình huấn luyện mô hình.


# tạo tensorflow dataset
# trộn tập dữ liệu
# dùng hàm preprocess như trên 
def tf_dataset(images, masks, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset


def get_data(batch, sequence):
    global is_sequence
    is_sequence = sequence
    (train_x, train_y), (valid_x, valid_y) = load_data()

    train_x, train_y = shuffling(train_x, train_y)
    train_dataset = tf_dataset(train_x, train_y, batch=batch)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

    train_steps = len(train_x) // batch
    valid_steps = len(valid_x) // batch

    if len(train_x) % batch != 0:
        train_steps += 1
    if len(valid_x) % batch != 0:
        valid_steps += 1

    #return (train_dataset, valid_dataset), (train_steps, valid_steps)
    print(train_dataset, valid_dataset)
