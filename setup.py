import sys
from cx_Freeze import setup, Executable



setup(
    name='Test',
    version='1.0',
    description = "Test exe",
    executables = [Executable("UsingHoltWintersExample.py")]
    
)
