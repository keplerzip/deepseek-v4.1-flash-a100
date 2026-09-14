"""Validate the live vision payload builders against schemas in the pinned image."""
import json
from pydantic import ValidationError
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import construct_input_messages
from vllm.entrypoints.chat_utils import _ResponsesInputImageParser
from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from acceptance import vision_request


def main():
    rows=[]
    for protocol, schema in [('chat',ChatCompletionRequest),('responses',ResponsesRequest),('messages',AnthropicMessagesRequest)]:
        for count in (1,5,8):
            body=vision_request(count,protocol)
            request=schema.model_validate(body)
            key='input' if protocol=='responses' else 'messages'
            assert len(body[key][0]['content'])==count+1
            if protocol=='responses':
                messages=construct_input_messages(request_input=request.input)
                assert len(messages[0]['content'])==count+1
                for item in body['input'][0]['content'][1:]:
                    assert _ResponsesInputImageParser(item)['detail']=='auto'
            rows.append({'protocol':protocol,'images':count,'status':'PASS'})
    invalid=vision_request(5,'responses')
    del invalid['input'][0]['content'][1]['detail']
    try:
        ResponsesRequest.model_validate(invalid)
    except ValidationError as exc:
        assert any(e['loc'][-1]=='detail' and e['type']=='missing' for e in exc.errors())
    else:
        raise AssertionError('Missing image detail should reproduce the pinned schema error')
    print(json.dumps({'status':'PASS','scope':'pinned request schemas and Responses conversion; no GPU inference',
                      'tests':rows,'missing_detail_regression':'PASS','gpu_inference_tested':False}))


if __name__=='__main__':
    main()
