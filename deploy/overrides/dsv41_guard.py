"""Local-only media, exact model identity and auth for auxiliary API routes."""
import hmac
import json
import os

MODEL = 'DeepSeek-V4.1-Flash'
ALLOWED_PATHS = {'/health', '/version', '/metrics', '/v1/models', '/v1/chat/completions',
                 '/v1/responses', '/v1/messages', '/v1/messages/count_tokens', '/tokenize',
                 '/detokenize', '/reset_prefix_cache', '/server_info'}
MAX_BODY = 128 * 1024 * 1024


def check_media(value):
    if isinstance(value, list):
        for entry in value:
            check_media(entry)
    elif isinstance(value, dict):
        kind = value.get('type')
        # JSON Schema may have a property named type, or a union type array.
        # Only strings can be media discriminators; still inspect all children.
        if not isinstance(kind, str):
            kind = None
        if kind in {'image_url', 'input_image'}:
            url = value.get('image_url', '')
            if isinstance(url, dict):
                url = url.get('url', '')
            if not isinstance(url, str) or not url.startswith('data:image/'):
                raise ValueError('Images must use an inline data:image URI in offline mode')
        if kind == 'image':
            source = value.get('source', {})
            if not isinstance(source, dict) or source.get('type') != 'base64':
                raise ValueError('Messages images must use base64 source in offline mode')
        if kind in {'video_url', 'input_audio', 'audio_url', 'input_file', 'file'}:
            raise ValueError('This delivery supports text and inline images')
        for entry in value.values():
            check_media(entry)


class OfflineGuard:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)

        async def reject(status, message):
            body = json.dumps({'error': {'message': message, 'type': 'invalid_request_error', 'code': 'offline_contract'}}).encode()
            await send({'type':'http.response.start','status':status,'headers':[(b'content-type',b'application/json'),(b'content-length',str(len(body)).encode())]})
            await send({'type':'http.response.body','body':body})

        path = scope['path'].rstrip('/') or '/'
        if path not in ALLOWED_PATHS:
            return await reject(404, 'Unsupported route for this offline delivery')
        key = os.environ.get('VLLM_API_KEY', '')
        if path != '/health' and key:
            headers = dict(scope['headers'])
            provided = headers.get(b'authorization', b'').decode(errors='replace')
            provided = provided[7:] if provided.startswith('Bearer ') else headers.get(b'x-api-key',b'').decode(errors='replace')
            if not hmac.compare_digest(provided, key):
                return await reject(401, 'Invalid API key')
        if scope['method'] not in {'POST','PUT'}:
            return await self.app(scope, receive, send)
        pieces, size = [], 0
        while True:
            event = await receive()
            if event['type'] == 'http.disconnect':
                return
            piece = event.get('body', b'')
            size += len(piece)
            if size > MAX_BODY:
                return await reject(413, 'Request exceeds 128 MiB')
            pieces.append(piece)
            if not event.get('more_body'):
                break
        body = b''.join(pieces)
        if body:
            try:
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError('Request body must be a JSON object')
                if payload.get('model', MODEL) != MODEL:
                    raise ValueError('The only served model is '+MODEL)
                check_media(payload)
            except (ValueError, TypeError) as exc:
                return await reject(400, str(exc))
        sent = False

        async def replay():
            nonlocal sent
            if sent:
                return await receive()
            sent = True
            return {'type':'http.request','body':body,'more_body':False}
        return await self.app(scope, replay, send)
