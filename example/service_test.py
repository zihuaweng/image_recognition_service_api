import requests
import time

start = time.time()

para = {'imageUrl': 'your url address', 'top': '1'}

r = requests.post("http://0.0.0.0:8888/image/recognition", data=para)
print(r.text)
print(time.time() - start)
