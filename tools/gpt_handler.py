import requests
import json
try:
    from .base import BaseHandler
except:
    from base import BaseHandler


key="""Bearer eyJhbGciOiJSUzI1NiIsInNvdXJjZSI6IklBTSIsImtpZCI6IjIwTml0b3h6S1FxMHJPdTlvYVk2UzA3bFBnUTBjSnFQIn0.eyJzdWIiOiI2MDAwMDI5NTIiLCJpc3MiOiJodHRwczpcL1wvZGV2LWEuc3p6aGlqaW5nLmNvbVwvaWFtLW9uZWlkIiwiYmFpQnVVc2VySWQiOiIyMDkxOCIsImZpbmFuY2VVc2VySWQiOiIxMDUyNjI4NTM2MTY1MDA3MzYwIiwiZmVpU2h1SWQiOiJvbl8xOTdkMTRmZjM5MmFhYjZhM2I2ZWQ0NmQ4MWY3NjI4NSIsImF1ZCI6IjQ0MTEyODg1MTkwNTU4MTA1NiIsInNjb3BlIjoib3BlbmlkIiwibmFtZSI6IuacsemUpuelpSIsImZpbmFuY2VTaWduVXNlcklkIjoiMTAyMzI1MzM2MjI4OTI1MDMwNCIsInVzZXJUeXBlIjoiSU5ORVIiLCJleHAiOjE3MDIzNjcxODIsImlhdCI6MTcwMTc2MjM4MiwianRpIjoiY2ZhMmE1NjBhMWU4NGY5NmE0NmMyZmZhMDM4YjUwYTYiLCJlbWFpbCI6InpodWppbnhpYW5nQHpqLnRlY2giLCJhY2NvdW50Ijoiemh1amlueGlhbmcifQ.RWw-8EaD3hMoiNmcXEfR0Y-A1R35aMqCWIKWnse50YM4INBtQNivVHFbsZZRNwup8XRTQe7PWQl0dPfoGH-Kz7eR3xCIZ6--ATW_PP8CbqGTCnbWAbVxT3enXBZjpgx5qE-JN5g-ko5bwpPylah4Wg2B8n6T87wC8Iczc-Aps4L7oevWjPCrne8tha4g3AWmuWGgn00LJjy4cQlIK9ETqLeVJJAZEDm82GLpYePISFERzs4-olnGz5IKcALtVXnX_KAXIloy5q1f3TgeNfPNkfdL1_TeXsrj-TZkJFW9OUVmNtiW7yFOdBN9t9Gv6HfP-Onva9pnOankMSLNXyOLQQ"""
class GPTHandler(BaseHandler):
    def __init__(self, args={}, **kwargs):
        super().__init__(args)
        self.api_url = self.handle_args.get("api_url")
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': key
        }

    def infer(self, input_text, **kwargs):
        data = {
            'messages':[
                {
                    'role': 'system',
                    'content': input_text,
                    # 'history': history
                }
            ]
        }
        result = requests.post(self.api_url, headers=self.headers, data=json.dumps(data))
        text = result.json()['data']['choices'][0]['message']['content']
        return text, None


if __name__ == "__main__":
    gpt_handler = GPTHandler({})
    input_text = '帮我分析这个报错：ImportError: attempted relative import with no known parent package'
    text,_ = gpt_handler.infer(input_text)
    print(text)
        

