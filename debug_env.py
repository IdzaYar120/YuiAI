from transformers.utils import is_accelerate_available
import accelerate
from packaging import version
import sys

print(f"Python executable: {sys.executable}")
print(f"Accelerate version: {accelerate.__version__}")
print(f"Is accelerate available via transformers: {is_accelerate_available()}")
try:
    print(f"Version check (>=1.1.0): {version.parse(accelerate.__version__) >= version.parse('1.1.0')}")
except Exception as e:
    print(f"Version check error: {e}")
