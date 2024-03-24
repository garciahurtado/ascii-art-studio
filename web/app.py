import os
from bottle import route, run, static_file, get

root = 'static'
@get("<filepath:re:.*\.html>")
def callback(filepath='index.html'):
    return static_file(filepath, root)

@get('/')
def callback():
    # return "I/O I/O it's off to hack we go"
    return static_file('index.html', root)


run(host='localhost', port=8080, debug=True)

