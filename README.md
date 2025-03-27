# SoundsOfYOLO

A real-time audio-visual system that translates computer vision into music by connecting object detection with MIDI.
This project was made for the "Coupe de France de Robotique Junior" 2025

## What is SoundsOfYOLO?

SoundsOfYOLO creates an interactive sonic environment that responds to objects detected by your webcam. The system uses YOLO (You Only Look Once) computer vision to identify objects in real-time, then converts these detections into MIDI signals that can control music software like Ableton Live or any MIDI-compatible synthesizer.

### Key Features:

- **Visual → Sound Translation**: Each detected object class (person, car, cup, etc.) triggers a unique MIDI note
- **Dynamic Expressivity**: The size of detected objects controls MIDI velocity, creating volume dynamics
- **Multi-Device Support**: Select from multiple cameras and MIDI output ports
- **Visual Feedback**: Color-coded detection display shows what objects are being tracked and what sounds they're creating

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

SoundsOfYOLO bridges the gap between computer vision and music, turning your physical environment into a musical instrument.
