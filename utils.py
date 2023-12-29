import os
import subprocess

def setup():
    # Set environment variables
    os.environ['XAUTHORITY'] = f"/run/user/{os.getuid()}/gdm/Xauthority"
    os.environ['DISPLAY'] = ':0'

    # Run the xhost command
    subprocess.run(["xhost", "+"])