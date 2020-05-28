import os
import sys
import re
import commands

a=commands.getoutput("fswebcam --no-banner -r 640x480 image3.jpg")
print(a)
