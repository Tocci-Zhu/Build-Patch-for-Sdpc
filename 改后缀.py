import os
import json

def convert_geojson_to_json(folder_path):
    """
    Convert all .geojson files in the specified folder to .json files.
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".geojson"):
            # 读取 .geojson 文件
            with open(os.path.join(folder_path, filename), 'r') as file:
                data = json.load(file)

            # 创建新的 .json 文件名
            new_filename = filename.replace('.geojson', '.json')
            new_file_path = os.path.join(folder_path, new_filename)

            # 写入 .json 文件
            with open(new_file_path, 'w') as new_file:
                json.dump(data, new_file, indent=4)

            print(f"Converted {filename} to {new_filename}")

if __name__ == "__main__":
    # 指定包含 .geojson 文件的文件夹路径
    folder_path = input("Enter the folder path containing .geojson files: ")
    convert_geojson_to_json(folder_path)