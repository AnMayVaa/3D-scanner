# RealSense D435 3D Scanner & Viewer 📸

A cross-platform Desktop Application built with **Electron**, **Node.js**, and **Python** to interface with the Intel RealSense Depth Camera (D435).

This application provides a real-time live preview via a web interface, aligns depth frames to color frames, and allows users to capture and export 3D Point Cloud data (`.ply`) for further processing.

## ✨ Features

- **Desktop GUI:** Built with Electron for a native application experience.
- **Real-time Preview:** Low-latency video streaming of the camera feed.
- **Auto Alignment:** Automatically aligns Depth frames to RGB frames for pixel-perfect accuracy.
- **3D Capture:** Exports 3D models in **.ply** format (viewable in MeshLab, CloudCompare).
- **Raw Data Saving:** Saves aligned RGB images (`.jpg`) and false-color Depth maps (`.png`).
- **Separated Architecture:** Uses Node.js for the frontend interface and Python for hardware processing (stable & efficient).

## 🛠️ Tech Stack

- **Frontend:** HTML5, CSS3, Socket.io (Client)
- **Middleware:** Node.js (Express, Socket.io, Child Process)
- **Backend/Driver:** Python 3 (pyrealsense2, OpenCV, NumPy)
- **Wrapper:** Electron

## 📋 Prerequisites

Before running the application, ensure you have the following installed:

1. **Hardware**
   - Intel RealSense Depth Camera D435 (Must be connected via **USB**).

2. **Software**
   - [Node.js](https://nodejs.org/) (v14 or higher)
   - [Python](https://www.python.org/) (v3.8 or higher)

## 📥 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AnMayVaa/3D-scanner/tree/main/js
cd realsense-3d-scanner
```

### 2. Install Node.js Dependencies
```bash
npm install
```

### 3. Install Python Dependencies
It is highly recommended to use a virtual environment.

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## ⚙️ Configuration (Important!)

You **must** configure the path to your Python executable in the `server.js` file so Node.js can spawn the camera process correctly.

1. Open `server.js`.
2. Find the line defining `pythonCommand`.
3. Change the path to match your local virtual environment path.

**Example for Windows:**
```javascript
// server.js

// ⚠️ NOTE: Use double backslashes (\\) for Windows paths
const pythonCommand = 'C:\\Users\\Admin\\Desktop\\realsense-3d-scanner\\venv\\Scripts\\python.exe';
```

**Example for Mac/Linux:**
```javascript
const pythonCommand = './venv/bin/python';
```

---

## 🚀 Usage

To start the application, run:

```bash
npm start
```

1. The Electron window will open.
2. Click **"Start Camera"** to initialize the RealSense stream.
3. Aim the camera at the object.
4. Click **"Capture 3D Model"** to save the data.

## 📂 Output Directory

Captured files are automatically saved in the `saved_images/` directory, organized by type:

- **`saved_images/RGB/`**: Aligned Color images (.jpg)
- **`saved_images/Depth/`**: Colorized Depth heatmaps (.png)
- **`saved_images/PLY/`**: 3D Point Cloud models (.ply)

## 🔧 Troubleshooting

| Issue | Possible Solution |
|-------|-------------------|
| **Camera not found** | Ensure the camera is plugged into a USB port. Try unplugging and replugging. |
| **Python Error / ModuleNotFound** | Check your `pythonCommand` path in `server.js`. Ensure you installed libs in the correct `venv`. |
| **Video Lag** | The preview is compressed for performance. The saved raw data will still be high quality. |
| **"Cannot GET /"** | Make sure `server.js` includes `app.get('/', ...)` serving the index.html file. |
