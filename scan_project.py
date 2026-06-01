import os

# Các thư mục hoặc file muốn quét
TARGET_EXTENSIONS = ['.py', '.md', '.json']
EXCLUDE_DIRS = ['venv', '__pycache__', '.git', 'data', 'outputs_resume']

def main():
    out_path = "context_du_an.txt"
    with open(out_path, "w", encoding="utf-8") as outfile:
        for root, dirs, files in os.walk("."):
            # Loại bỏ các thư mục không cần thiết (theo tên thư mục)
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for file in files:
                if any(file.endswith(ext) for ext in TARGET_EXTENSIONS):
                    file_path = os.path.join(root, file)
                    outfile.write(f"\n\n{'='*20}\nFILE: {file_path}\n{'='*20}\n")
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            outfile.write(f.read())
                    except Exception as e:
                        outfile.write(f"[Không thể đọc file: {e}]")
    print(f"Đã gom xong toàn bộ dự án vào file {out_path}!")

if __name__ == '__main__':
    main()
