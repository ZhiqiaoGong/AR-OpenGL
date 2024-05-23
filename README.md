# Implementation of Visual Tracking for Planar Markers

## Purpose:
The primary goal of this project was to manually implement the visual tracking of planar markers without using any ready-made SDKs like ARCore or easyAR. The project aimed to replicate the effects similar to a previous experiment using both natural and artificial markers.

## Hardware and Software Environment:
- **Hardware:** Huawei Matebook xPro
- **Software:** Microsoft Visual Studio Community 2019, OpenCV 3.4.15, OpenCV_contrib 3.4.15, OpenGL (glfw, glad), Cmake 3.18.2

## Procedure and Technical Details:

1. **Environment Setup:**
   - Configured OpenCV and related libraries in the system's PATH.
   - Compiled OpenCV using CMake with integration of OpenCV_contrib modules.
   - Configured OpenGL libraries like glfw and established the project in Visual Studio.

2. **Calibration and Marker Generation:**
   - Calibrated the camera using ChAruco images to get precise camera parameters.
   - Generated a specific marker (6x6 size, id=1) using OpenCV's aruco module.

3. **Development of the Visual Tracking Algorithm:**
   - Utilized OpenCV for marker detection and OpenGL for rendering virtual objects in real-time.
   - Implemented the tracking algorithm which involved:
     - Capturing live video feed from a camera.
     - Detecting the marker in each frame.
     - Computing relative pose of the marker with respect to the camera using OpenCV functions.
     - Dynamically updating view matrices based on the detected marker to align the virtual object accurately in 3D space.

4. **OpenGL Graphics Rendering:**
   - Managed 3D transformations using model, view, and projection matrices:
     - **Model Matrix:** Identity matrix as transformations were minimal.
     - **View Matrix:** Updated per frame based on the marker's detected pose.
     - **Projection Matrix:** Computed once using camera parameters, consistent throughout the session.
   - Rendered a simple 3D cube as a virtual object associated with the detected marker.

5. **Performance Measurement:**
   - Calculated the frame rate of the algorithm to evaluate the efficiency and responsiveness of the visual tracking.

## Conclusion and Insights:
The project provided hands-on experience with the integration of image processing and 3D graphics rendering techniques. It deepened understanding of OpenGL's role in visual presentations and the practical use of camera calibration in augmented reality. The project highlighted the challenges and solutions in implementing a robust visual tracking system using foundational libraries like OpenCV and OpenGL without the aid of specialized AR development kits.
