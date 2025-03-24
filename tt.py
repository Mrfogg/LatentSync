from urllib.parse import urlparse


def get_filename_from_url(url):
    """
    从 URL 中提取文件名，并去掉查询字符串和锚点。

    参数:
        url (str): 输入的 URL。

    返回:
        str: 提取的文件名。
    """
    # 解析 URL
    parsed_url = urlparse(url)

    # 获取路径部分
    path = parsed_url.path

    # 使用 os.path.basename 获取文件名
    from os.path import basename
    filename = basename(path)

    return filename


# 示例用法
if __name__ == "__main__":
    url = "https://example.com/path/to/file.txt?query=123&param=abc#anchor"
    filename = get_filename_from_url(url)
    print(f"从 URL 中提取的文件名是: {filename}")