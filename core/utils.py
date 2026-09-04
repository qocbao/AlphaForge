import sys
import io

def setup_utf8():
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
