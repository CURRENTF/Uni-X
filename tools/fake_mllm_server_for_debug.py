import sys

from flask import Flask, request, jsonify
import json

# 创建一个 Flask web server
app = Flask(__name__)


def pretty_print(content: str, max_len: int = 500):
    """
    一个自定义的打印函数。
    当单行内容超过 max_len 时，会省略中间部分，以保持输出的简洁性。
    """
    # 将内容按行分割
    for line in str(content).splitlines():
        if len(line) > max_len:
            # 计算两边各保留的字符数
            half = (max_len - 3) // 2
            # 打印截断并用 '...' 连接的字符串
            print(f"{line[:half]}...{line[-half:]}")
        else:
            # 如果行长度未超过限制，则直接打印
            print(line)


@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def catch_all(path):
    """
    捕获所有请求，打印请求的详细信息，并返回一个模拟的 JSON 响应。
    """
    print("=" * 50)
    print(f"Received Request for Path: /{path}")
    print(f"Method: {request.method}")
    pretty_print(f"Headers: \n{request.headers}")

    # 打印查询参数 (e.g., /?key=value)
    if request.args:
        # 使用自定义的打印函数
        pretty_print(f"Query Parameters: \n{json.dumps(request.args, indent=2)}")

    # 打印 JSON 请求体
    if request.is_json:
        # 使用 ensure_ascii=False 来正确显示中文字符
        # 使用自定义的打印函数
        pretty_print(f"JSON Body: \n{json.dumps(request.json, indent=2, ensure_ascii=False)}")
    else:
        # 使用自定义的打印函数
        pretty_print(f"Raw Body: \n{request.get_data(as_text=True)}")

    print("=" * 50 + "\n")

    # 伪造一个类似 OpenAI API 的成功响应
    # lmms_eval 通常需要一个 choices 列表，其中包含 message 字段
    fake_response = {
        "id": "chatcmpl-fake-id-12345",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "fake-model-for-lmms-eval",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a fake response from the server.",
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

    return jsonify(fake_response)


if __name__ == '__main__':
    # 运行服务器，监听在 5000 端口
    # host='0.0.0.0' 使其可以被局域网内的其他机器访问
    # debug=True 可以在代码修改后自动重载
    print("Fake server is running on http://0.0.0.0:5000")
    print("Press CTRL+C to stop the server.")
    app.run(host='0.0.0.0', port=5000, debug=True)