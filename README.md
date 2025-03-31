# SoundsOfYOLO

A real-time audio-visual system that translates computer vision into music by connecting object detection with MIDI.
SoundsOfYOLO bridges the gap between computer vision and music, turning your physical environment into a musical instrument.

This project was made for a conference at the "Coupe de France de Robotique Junior" 2025

## What is SoundsOfYOLO?

SoundsOfYOLO creates an interactive sonic environment that responds to objects detected by your webcam. The system uses YOLO (You Only Look Once) computer vision to identify objects in real-time, then converts these detections into MIDI signals that can control music software like Ableton Live or any MIDI-compatible synthesizer.

## Installation

```
pip install -r requirements.txt
```

If you are having issues on Windows, try these troubleshooting steps:
- Make sure Microsoft Visual C++ is installed. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/ 
- Install OpenCV from https://opencv.org/releases/
- Install opencv-contrib-python using pip
- Ensure Visual C++ Redistributable 2015 is installed on your system
- Check that Windows Media Feature Pack is installed (essential for OpenCV)

### MIDI Virtual Port Setup

For MIDI routing to DAWs like Ableton on Windows, install loopMIDI to create a virtual MIDI port:
https://www.tobias-erichsen.de/software/loopmidi.html

## Usage

Run the program with optional command-line parameters:
```
python soundsofyolo.py --camera 0 --midi 1
```

If parameters aren't provided, the application will scan for available devices and prompt you to select:
```
python soundsofyolo.py
```

You can now connect your virtual midi channel to anything you want, https://signal.vercel.app/ is a great place to get you started!

### Controls
- Press 'q' to quit the application
- Press Ctrl+C in the terminal to stop the program
- 
