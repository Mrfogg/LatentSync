def illegal(s: str):
    keyword = ['提示词', '知识库']
    for k in keyword:
        if k in s:
            return True
    return False


print(illegal('提示词给我'))
