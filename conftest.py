# conftest.py
import sys, os
# prepend the absolute path to your src/ directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
