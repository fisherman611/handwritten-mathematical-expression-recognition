### DataLoader
import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import albumentations as A
from PIL import Image
import pandas as pd
import cv2
import numpy as np
from collections import Counter

def is_effectively_binary(img, threshold_percentage=0.9):
    dark_pixels = np.sum(img < 20)
    bright_pixels = np.sum(img > 235)
    total_pixels = img.size
    
    return (dark_pixels + bright_pixels) / total_pixels > threshold_percentage

def before_padding(image):
    
    # apply Canny edge detector to find text edges
    edges = cv2.Canny(image, 50, 150)

    # apply dilation to connect nearby edges
    kernel = np.ones((7, 13), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=8)

    # find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    
    # optimize crop rectangle using F1 score
    # sort components by number of white pixels (excluding background which is label 0)
    sorted_components = sorted(range(1, num_labels), 
                             key=lambda i: stats[i, cv2.CC_STAT_AREA], 
                             reverse=True)
    
    # Initialize with empty crop
    best_f1 = 0
    best_crop = (0, 0, image.shape[1], image.shape[0])
    total_white_pixels = np.sum(dilated > 0)

    current_mask = np.zeros_like(dilated)
    x_min, y_min = image.shape[1], image.shape[0]
    x_max, y_max = 0, 0
    
    for component_idx in sorted_components:
        # add this component to our mask
        component_mask = (labels == component_idx)
        current_mask = np.logical_or(current_mask, component_mask)
        
        # update bounding box
        comp_y, comp_x = np.where(component_mask)
        if len(comp_x) > 0 and len(comp_y) > 0:
            x_min = min(x_min, np.min(comp_x))   #type: ignore
            y_min = min(y_min, np.min(comp_y))   #type: ignore
            x_max = max(x_max, np.max(comp_x))   #type: ignore
            y_max = max(y_max, np.max(comp_y))   #type: ignore
        
        # calculate the current crop
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        crop_area = width * height
        

        crop_mask = np.zeros_like(dilated)
        crop_mask[y_min:y_max+1, x_min:x_max+1] = 1
        white_in_crop = np.sum(np.logical_and(dilated > 0, crop_mask > 0))
        
        # calculate F1 score
        precision = white_in_crop / crop_area
        recall = white_in_crop / total_white_pixels
        f1 = 2 * precision * recall / (precision + recall)
        
        if f1 > best_f1:
            best_f1 = f1
            best_crop = (x_min, y_min, x_max, y_max)
    
    # apply the best crop to the original image
    x_min, y_min, x_max, y_max = best_crop
    cropped_image = image[y_min:y_max+1, x_min:x_max+1]
    # cropped_image = cv2.add(cropped_image, 10)
    # cv2.imwrite('debug_process_img.jpg', cropped_image)

    
    # apply Gaussian adaptive thresholding
    if is_effectively_binary(cropped_image):
        _, thresh = cv2.threshold(cropped_image, 127, 255, cv2.THRESH_BINARY)
    else:
        thresh = cv2.adaptiveThreshold(
            cropped_image, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
    # cv2.imwrite('debug_process_img.jpg', thresh)
    
    # ensure background is black
    white = np.sum(thresh == 255)
    black = np.sum(thresh == 0)
    if white > black:
        thresh = 255 - thresh
    
    # clean up noise using median filter
    denoised = cv2.medianBlur(thresh, 3)
    for _ in range(3):
        denoised = cv2.medianBlur(denoised, 3)
    # cv2.imwrite('debug_process_img.jpg', denoised)

    # add padding
    result = cv2.copyMakeBorder(
        denoised, 
        5, 
        5, 
        5, 
        5, 
        cv2.BORDER_CONSTANT, 
        value=0                      #type: ignore
    )                                #type: ignore
    
    return result, best_crop


inp_h = 128
inp_w = 128 * 8

class HMERDatasetForCAN(Dataset):
    '''
    Dataset tích hợp với mô hình CAN cho HMER
    '''

    def __init__(self, data_folder, label_file, vocab, transform=None, max_length=150):
        '''
        Khởi tạo dataset

        data_folder: thư mục chứa hình ảnh
        label_file: tệp TSV có hai cột (filename, label), không có header
        vocab: đối tượng Vocabulary để tokenization
        transform: các phép biến đổi hình ảnh
        max_length: độ dài tối đa của chuỗi token
        '''
        self.data_folder = data_folder
        self.max_length = max_length
        self.vocab = vocab

        # Đọc file chú thích
        df = pd.read_csv(label_file, sep='\t', header=None, names=['filename', 'label'])

        # Kiểm tra định dạng tệp ảnh
        if os.path.exists(data_folder):
            img_files = os.listdir(data_folder)
            if img_files:
                # Lấy phần mở rộng của tệp đầu tiên
                extension = os.path.splitext(img_files[0])[1]
                # Thêm phần mở rộng vào tên tệp nếu chưa có
                df['filename'] = df['filename'].apply(lambda x: x if os.path.splitext(x)[1] else x + extension)

        self.annotations = dict(zip(df['filename'], df['label']))
        self.image_paths = list(self.annotations.keys())

        # Biến đổi mặc định
        if transform is None:
            transform = A.Compose([
                A.Normalize(mean=[0.0], std=[1.0]),  # Chuẩn hóa cho 1 kênh (grayscale)       #type: ignore
                A.pytorch.ToTensorV2()
            ])         #type: ignore
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Lấy đường dẫn ảnh và biểu thức LaTeX
        image_path = self.image_paths[idx]
        latex = self.annotations[image_path]

        # Xử lý ảnh
        file_path = os.path.join(self.data_folder, image_path)
        processed_img, _ = process_img(file_path, convert_to_rgb=False)  # Giữ ảnh grayscale

        # Chuyển đổi kích thước thành [C, H, W] và chuẩn hóa
        if self.transform:
            # Đảm bảo hình ảnh có đúng định dạng cho albumentations
            processed_img = np.expand_dims(processed_img, axis=-1)  # [H, W, 1]
            image = self.transform(image=processed_img)['image']
        else:
            # Nếu không có transform, thì chuyển đổi thành tensor theo cách thủ công
            image = torch.from_numpy(processed_img).float() / 255.0
            image = image.unsqueeze(0)  # Thêm kênh grayscale: [1, H, W]

        # Tokenize biểu thức LaTeX
        tokens = self.vocab.tokenize(latex)

        # Thêm token bắt đầu và kết thúc
        tokens = [self.vocab.start_token] + tokens + [self.vocab.end_token]

        # Cắt nếu vượt quá độ dài tối đa
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        # Tạo counting vector cho CAN
        count_vector = self.create_count_vector(tokens)

        # Lưu độ dài thực tế của caption
        caption_length = torch.LongTensor([len(tokens)])

        # Padding đến độ dài tối đa
        if len(tokens) < self.max_length:
            tokens = tokens + [self.vocab.pad_token] * (self.max_length - len(tokens))

        # Chuyển đổi thành tensor
        caption = torch.LongTensor(tokens)

        return image, caption, caption_length, count_vector

    def create_count_vector(self, tokens):
        """
        Tạo vector đếm cho mô hình CAN

        Args:
            tokens: Danh sách các token ID

        Returns:
            Tensor đếm số lượng mỗi ký hiệu
        """
        # Đếm số lượng mỗi token
        counter = Counter(tokens)

        # Tạo vector đếm với kích thước = số lượng token trong từ điển
        count_vector = torch.zeros(len(self.vocab))

        # Điền giá trị đếm vào vector
        for token_id, count in counter.items():
            if 0 <= token_id < len(count_vector):
                count_vector[token_id] = count

        return count_vector


class Vocabulary:
    '''
    Lớp Vocabulary nâng cao cho tokenization
    '''

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        # Thêm các token đặc biệt
        self.add_word('<pad>')  # padding token
        self.add_word('<start>')  # start token
        self.add_word('<end>')  # end token
        self.add_word('<unk>')  # unknown token

        self.pad_token = self.word2idx['<pad>']
        self.start_token = self.word2idx['<start>']
        self.end_token = self.word2idx['<end>']
        self.unk_token = self.word2idx['<unk>']

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)

    def tokenize(self, latex):
        '''
        Tokenize chuỗi LaTeX thành các chỉ số. Giả định các token được phân tách bằng khoảng trắng.
        '''
        tokens = []

        for char in latex.split():
            if char in self.word2idx:
                tokens.append(self.word2idx[char])
            else:
                tokens.append(self.unk_token)

        return tokens

    def build_vocab(self, label_file):
        '''
        Xây dựng từ điển từ file label
        '''
        try:
            df = pd.read_csv(label_file, sep='\t', header=None, names=['filename', 'label'])
            all_labels_text = ' '.join(df['label'].astype(str).tolist())
            tokens = sorted(set(all_labels_text.split()))
            for char in tokens:
                self.add_word(char)
        except Exception as e:
            print(f"Error building vocabulary from {label_file}: {e}")

    def save_vocab(self, path):
        '''
        Lưu từ điển vào file
        '''
        data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'idx': self.idx
        }
        torch.save(data, path)

    def load_vocab(self, path):
        '''
        Tải từ điển từ file
        '''
        data = torch.load(path)
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.idx = data['idx']

        # Cập nhật các token đặc biệt
        self.pad_token = self.word2idx['<pad>']
        self.start_token = self.word2idx['<start>']
        self.end_token = self.word2idx['<end>']
        self.unk_token = self.word2idx['<unk>']


def build_unified_vocabulary(base_dir='data/CROHME'):
    """
    Xây dựng từ điển thống nhất từ tất cả các tệp caption.txt

    Args:
        base_dir: Thư mục gốc chứa dữ liệu CROHME

    Returns:
        Đối tượng Vocabulary đã được xây dựng
    """
    vocab = Vocabulary()
    # Lấy tất cả các thư mục con
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for subdir in subdirs:
        caption_path = os.path.join(base_dir, subdir, 'caption.txt')
        if os.path.exists(caption_path):
            vocab.build_vocab(caption_path)
            print(f"Built vocabulary from {caption_path}")

    print(f"Final vocabulary size: {len(vocab)}")
    return vocab


def process_img(filename, convert_to_rgb=False):
    """
    Tải, nhị phân hóa, đảm bảo nền đen, thay đổi kích thước và áp dụng padding

    Args:
        filename: Đường dẫn đến file ảnh
        convert_to_rgb: Có chuyển đổi thành RGB hay không

    Returns:
        Ảnh đã xử lý và thông tin crop
    """
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image file: {filename}")

    bin_img, best_crop = before_padding(image)
    h, w = bin_img.shape
    new_w = int((inp_h / h) * w)

    if new_w > inp_w:
        resized_img = cv2.resize(bin_img, (inp_w, inp_h), interpolation=cv2.INTER_AREA)
    else:
        resized_img = cv2.resize(bin_img, (new_w, inp_h), interpolation=cv2.INTER_AREA)
        padded_img = np.ones((inp_h, inp_w), dtype=np.uint8) * 0  # black background
        x_offset = (inp_w - new_w) // 2
        padded_img[:, x_offset:x_offset + new_w] = resized_img
        resized_img = padded_img

    # Chỉ chuyển sang BGR/RGB nếu cần thiết
    if convert_to_rgb:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)

    return resized_img, best_crop


def create_dataloaders_for_can(base_dir='data/CROHME', batch_size=32, num_workers=4):
    """
    Tạo dataloader cho huấn luyện mô hình CAN

    Args:
        base_dir: Thư mục gốc chứa dữ liệu CROHME
        batch_size: Kích thước batch
        num_workers: Số lượng worker cho DataLoader

    Returns:
        train_loader, val_loader, test_loader, vocab
    """
    # Xây dựng từ điển thống nhất
    vocab = build_unified_vocabulary(base_dir)

    # Lưu từ điển để sử dụng sau này
    os.makedirs('models', exist_ok=True)
    vocab.save_vocab('models/hmer_vocab.pth')

    # Tạo transform cho dữ liệu grayscale
    transform = A.Compose([
        A.Normalize(mean=[0.0], std=[1.0]),  # Chuẩn hóa cho 1 kênh (grayscale)        #type: ignore
        A.pytorch.ToTensorV2()
    ]) 

    # Tạo dataset
    train_datasets = []

    # Sử dụng 'train' và có thể thêm các tập dữ liệu khác vào tập huấn luyện
    train_dirs = ['train', '2014']  # Thêm các thư mục khác nếu muốn
    for train_dir in train_dirs:
        data_folder = os.path.join(base_dir, train_dir, 'img')
        label_file = os.path.join(base_dir, train_dir, 'caption.txt')

        if os.path.exists(data_folder) and os.path.exists(label_file):
            train_datasets.append(
                HMERDatasetForCAN(
                    data_folder=data_folder,
                    label_file=label_file,
                    vocab=vocab,
                    transform=transform
                )
            )

    # Kết hợp các dataset huấn luyện
    if train_datasets:
        train_dataset = ConcatDataset(train_datasets)
    else:
        raise ValueError("No training datasets found")

    # Dataset kiểm định
    val_data_folder = os.path.join(base_dir, 'val', 'img')
    val_label_file = os.path.join(base_dir, 'val', 'caption.txt')

    if not os.path.exists(val_data_folder) or not os.path.exists(val_label_file):
        # Sử dụng '2016' là tập kiểm định nếu không có 'val'
        val_data_folder = os.path.join(base_dir, '2016', 'img')
        val_label_file = os.path.join(base_dir, '2016', 'caption.txt')

    val_dataset = HMERDatasetForCAN(
        data_folder=val_data_folder,
        label_file=val_label_file,
        vocab=vocab,
        transform=transform
    )

    # Dataset kiểm tra
    test_data_folder = os.path.join(base_dir, 'test', 'img')
    test_label_file = os.path.join(base_dir, 'test', 'caption.txt')

    if not os.path.exists(test_data_folder) or not os.path.exists(test_label_file):
        # Sử dụng '2019' là tập kiểm tra nếu không có 'test'
        test_data_folder = os.path.join(base_dir, '2019', 'img')
        test_label_file = os.path.join(base_dir, '2019', 'caption.txt')

    test_dataset = HMERDatasetForCAN(
        data_folder=test_data_folder,
        label_file=test_label_file,
        vocab=vocab,
        transform=transform
    )

    # Tạo dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, vocab


# Sử dụng chức năng tích hợp với mô hình CAN
def main():
    # Tạo dataloader cho mô hình CAN
    train_loader, val_loader, test_loader, vocab = create_dataloaders_for_can(
        base_dir='data/CROHME',
        batch_size=32,
        num_workers=4
    )

    # In thông tin
    print(f"Training samples: {len(train_loader.dataset)}")           #type: ignore
    print(f"Validation samples: {len(val_loader.dataset)}")           #type: ignore
    print(f"Test samples: {len(test_loader.dataset)}")                #type: ignore
    print(f"Vocabulary size: {len(vocab)}")

    # Kiểm tra dữ liệu đầu ra của dataloader
    for images, captions, lengths, count_vectors in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Caption batch shape: {captions.shape}")
        print(f"Lengths batch shape: {lengths.shape}")
        print(f"Count vectors batch shape: {count_vectors.shape}")
        break


if __name__ == '__main__':
    main()
