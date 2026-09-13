"""Offline installed-tokenizer, Responses conversion and input-policy tests; no GPU."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
from fastapi import FastAPI
from fastapi.testclient import TestClient
from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import construct_input_messages, construct_tool_dicts
from dsv41_guard import OfflineGuard, check_media

p=argparse.ArgumentParser()
p.add_argument('--model',type=Path,required=True)
p.add_argument('--captured-requests',type=Path)
args=p.parse_args()
spec=importlib.util.spec_from_file_location('official_reference',args.model/'encoding/encoding.py')
reference=importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
tokenizer=DeepseekV41Tokenizer.from_pretrained(str(args.model),local_files_only=True)


class RuntimeChecks(unittest.TestCase):
    def test_independent_official_prompts_and_token_ids(self):
        for file in sorted((args.model/'encoding/tests').glob('test_input_*.json')):
            for case in reference.load_cases(str(file)):
                for mode in ['chat','thinking']:
                    for effort in ['low','high','max',42]:
                        messages=copy.deepcopy(case['messages'])
                        expected=reference.encode_messages(copy.deepcopy(messages),thinking_mode=mode,reasoning_effort=effort)
                        actual=tokenizer.apply_chat_template(messages,tokenize=False,thinking=mode=='thinking',reasoning_effort=effort)
                        self.assertEqual(actual,expected)
                        actual_ids=tokenizer.apply_chat_template(messages,tokenize=True,thinking=mode=='thinking',reasoning_effort=effort)
                        self.assertEqual(actual_ids,tokenizer.encode(expected,add_special_tokens=False))

    def test_actual_codex_request_and_namespace_conversion(self):
        if not args.captured_requests:
            self.skipTest('No captured client payload provided')
        count=0
        for row in json.loads(args.captured_requests.read_text()):
            if row['path']!='/v1/responses':continue
            check_media(row['body'])
            req=ResponsesRequest.model_validate(row['body'])
            messages=construct_input_messages(request_input=req.input,request_instructions=req.instructions)
            tools=construct_tool_dicts(req.tools,req.tool_choice)
            prompt=tokenizer.apply_chat_template(messages,tools=tools,tokenize=False,reasoning_effort='high')
            self.assertIn('exec_command',prompt)
            self.assertNotIn('input_text',prompt[:30])
            self.assertGreater(len(tokenizer.encode(prompt)),100)
            count+=1
        self.assertGreater(count,0)

    def test_input_guard_and_unchanged_body(self):
        app=FastAPI()
        @app.post('/v1/chat/completions')
        async def handler(payload:dict):return payload
        app.add_middleware(OfflineGuard)
        with TestClient(app) as client:
            body={'model':'DeepSeek-V4.1-Flash','messages':[{'role':'user','content':'https://example.com is text, not an image fetch'}]}
            self.assertEqual(client.post('/v1/chat/completions',json=body).json(),body)
            body['tools']=[{'type':'function','function':{'name':'schema_probe','parameters':{
                'type':'object','properties':{'type':{'type':'string'},'optional':{'type':['string','null']}}}}}]
            self.assertEqual(client.post('/v1/chat/completions',json=body).json(),body)
            self.assertEqual(client.post('/v1/chat/completions',json={**body,'model':'wrong'}).status_code,400)
            body['messages'][0]['content']=[{'type':'image_url','image_url':{'url':'https://example.invalid/image.png'}}]
            self.assertEqual(client.post('/v1/chat/completions',json=body).status_code,400)
            body['messages'][0]['content'][0]['image_url']['url']='data:image/png;base64,AAAA'
            self.assertEqual(client.post('/v1/chat/completions',json=body).status_code,200)
            self.assertEqual(client.get('/download-model').status_code,404)

    def test_developer_instruction_priority(self):
        messages=[{'role':'developer','content':'Application instruction A.'},{'role':'user','content':'Question'},
                  {'role':'developer','content':'Application instruction B.'}]
        official=[{**m,'role':'system' if m['role']=='developer' else m['role']} for m in messages]
        self.assertEqual(tokenizer.apply_chat_template(messages,tokenize=False,thinking=True,reasoning_effort='high'),
                         reference.encode_messages(official,thinking_mode='thinking',reasoning_effort='high'))


if __name__=='__main__':
    unittest.main(argv=[sys.argv[0]])
