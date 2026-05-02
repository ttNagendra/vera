import urllib.request, json
import base64

_encrypted_key = b'QUl6YVN5RFM3alFvWUs4V2FPSE1QZXlORnQtY2JsUnhkVUhiM3gw'
key = base64.b64decode(_encrypted_key).decode('utf-8')
for model in ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-flash-latest', 'gemini-2.5-flash-lite']:
    url = 'https://generativelanguage.googleapis.com/v1beta/models/' + model + ':generateContent?key=' + key
    body = json.dumps({'contents':[{'parts':[{'text':'Say hi.'}],'role':'user'}],'generationConfig':{'temperature':0,'maxOutputTokens':5}}).encode()
    req = urllib.request.Request(url, data=body, headers={'Content-Type':'application/json'})
    try:
        r = urllib.request.urlopen(req)
        data = json.loads(r.read())
        text = data['candidates'][0]['content']['parts'][0]['text']
        print('OK:', model, '->', text.strip())
    except Exception as e:
        if hasattr(e, 'read'):
            err = json.loads(e.read()).get('error', {})
            status = err.get('status', '?')
            msg = err.get('message', '')[:100]
            print('FAIL', model, ':', status, '-', msg)
        else:
            print('FAIL', model, ':', str(e)[:100])
