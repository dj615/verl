import os
import urllib.request
import zipfile


def main():
    # 1. 下载地址（raw 文件直链）
    url = "https://raw.githubusercontent.com/aadityasingh/HARP/main/HARP.jsonl.zip"

    # 2. 目标目录和文件名
    raw_dir = "/root/workspace/ASR_data/raw"
    os.makedirs(raw_dir, exist_ok=True)

    zip_path = os.path.join(raw_dir, "HARP.jsonl.zip")
    jsonl_path = os.path.join(raw_dir, "HARP.jsonl")

    print(f"[1/3] 保存目录: {raw_dir}")

    # 3. 下载 zip
    print(f"[2/3] 正在从 {url} 下载 HARP.jsonl.zip ...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"下载完成: {zip_path}")

    # 4. 解压得到 HARP.jsonl
    print("[3/3] 正在解压 HARP.jsonl.zip ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # zip 里就一个 HARP.jsonl，保险起见统一按名字提取
        for name in zf.namelist():
            if name.endswith(".jsonl"):
                zf.extract(name, raw_dir)
                # 如果 zip 里带子目录，这里顺手改成你想要的最终路径
                src_path = os.path.join(raw_dir, name)
                if src_path != jsonl_path:
                    os.replace(src_path, jsonl_path)
                break

    print(f"解压完成，JSONL 文件路径: {jsonl_path}")

    os.remove(zip_path)

if __name__ == "__main__":
    main()
