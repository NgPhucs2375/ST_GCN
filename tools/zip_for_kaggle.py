import zipfile
import os

print("Tạo file ZIP tương thích chuẩn với Kaggle...")
zip_filename = 'augmented_data_fixed.zip'
folders_to_zip = ['data/videos_augmented', 'data/raw_ipn_augmented']

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
    for folder in folders_to_zip:
        if os.path.exists(folder):
            for root, _, files in os.walk(folder):
                for file in files:
                    filepath = os.path.join(root, file)
                    # Ghi file vào trong ZIP với cấu trúc đường dẫn giữ nguyên
                    zf.write(filepath, filepath)
                    
print(f"✅ Đã tạo xong file {zip_filename} thành công! File này an toàn để tải lên Kaggle.")
