
## Seam Carving Application README

### Overview

This application is a seam carving tool that allows users to remove vertical seams from an image to reduce its width. The app uses various image processing techniques to calculate energy maps and determine which seams to remove, enhancing image retargeting. The seam carving process takes into account the image's saliency, depth gradient and entropy maps.

### Key Features

1. **Energy Calculation**: Computes an energy map based on edge detection, saliency, depth, and entropy maps to guide seam removal.
2. **Seam Finding and Removal**: Identifies and removes the lowest energy seams iteratively.
3. **Seam Visualization**: Highlights seams on the image during the carving process for visualization.
4. **Graphical User Interface (GUI)**: A Tkinter-based GUI for user interaction, allowing easy input of parameters and real-time progress updates.

### Dependencies

- OpenCV
- NumPy
- scikit-image
- Tkinter
- Pillow

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install required Python packages**:
   ```bash
   pip install opencv-python-headless numpy scikit-image pillow
   ```
3. **Sample Dataset**
    In the sample_dataset folder, each subfolder should be named according to the image category (e.g., Diana). Each subfolder should contain the saliency map and depth map for the corresponding images.
### Usage

1. **Run the Application**:
   ```bash
   python seam_carving_app.py
   ```

2. **GUI Interaction**:
   - **Enter Category**: Input the image category (e.g., Diana, Baby, Snowman).
   - **Number of Columns to Delete**: Specify the number of vertical seams (columns) to remove.
   - **Start Seam Carving**: Click the button to begin the process.

3. **Output**:
   - The application will display the image with highlighted seams during the process.
   - Upon completion, the output image will be saved to the specified directory.