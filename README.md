# Light Energy Prototype

## Overview
The **Light Energy Prototype** is a computer vision-based project that uses object detection to control lighting based on the presence of people in a room and the amount of sunlight available. It utilizes OpenCV and YOLO (You Only Look Once) for real-time person detection and dynamically adjusts the lighting conditions.

## Features
- **Real-time person detection** using YOLOv4-tiny.
- **Simulated sunlight level detection** with a random generator.
- **Interactive buttons** to toggle person detection and generate new sunlight levels.
- **Automated lighting control** based on the number of people and sunlight levels.

## How It Works
- The program captures frames from a webcam.
- It detects people in the frame if person detection is enabled.
- The sunlight level is randomly generated and can be refreshed with a button click.
- The lighting adjusts as follows:
  - **ON** if people are present and sunlight is below a threshold.
  - **DIM** if people are present but sunlight is above the threshold.
  - **OFF** if no people are detected.

## Installation

### Prerequisites
Ensure you have Python installed along with the following dependencies:
```bash
pip install opencv-python numpy
```
### Clone Repository
```bash
git clone https://github.com/Raaed-Mirza/Light-Energy-Prototype.git
```
```bash
cd Light-Energy-Prototype
```

### Usage
Run the script with:
```bash
python light_energy_prototype.py
```
### Controls
- Person Detection Button: Toggle detection ON/OFF.
- Generate Sunlight Button: Randomly change sunlight level.
- ESC Key: Exit the program.
