# SoundsOfYOLO

YOLO from webcam to midi virtual stream to control Ableton

# Install

```
pip install -r requirements.txt
```

## Windows
Make sure Microsoft Visual C++ is installed. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/ 

If you are having issue on windows, here are some things you can try:
- Install opencv from https://opencv.org/releases/
- Install opencv-contrib-python using pip
- Check for Visual C++ Redistributable 2015 : Ensure it is installed on your system.
- Check for Windows Media Feature Pack : Make sure it is installed, as it is essential for opencv.

If that doesn't fix it for you, refer to this issue: https://github.com/opencv/opencv-python/issues/36

Install loopMIDI to create a MIDI virtual port.
https://www.tobias-erichsen.de/software/loopmidi.html

