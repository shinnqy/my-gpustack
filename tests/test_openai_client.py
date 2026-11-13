"""第十三章手动检查点；只有直接运行本文件时才访问 Server。"""

import argparse

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(description="检查模型列表和 OpenAI 流式推理")
    parser.add_argument("--base-url", default="http://127.0.0.1:30080/v1")
    parser.add_argument("--model", default="proxy-demo")
    args = parser.parse_args()
    with OpenAI(base_url=args.base_url, api_key="unused") as client:
        print([model.id for model in client.models.list()])
        with client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "你好"}],
            stream=True,
        ) as stream:
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
