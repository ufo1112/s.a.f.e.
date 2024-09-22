
# YOLO Object Detection App

This Streamlit application allows users to upload videos and perform real-time object detection using a YOLO model. It specifically detects safety equipment such as Mask, Safety Vest, and Hardhat, providing live feedback on their presence.

## Features

- **Video Upload**: Supports uploading video files in `.mp4`, `.avi`, and `.mov` formats.
- **Real-time Detection**: Processes video frames in real-time using a YOLO model.
- **Safety Equipment Detection**: Identifies if safety equipment is present.
- **User Interface**: Interactive UI built with Streamlit, including progress bars and warning messages.
- **Customizable Settings**: Adjust the confidence threshold via the sidebar.

## Demo

<!-- Replace with actual path or URL -->

## Installation

### Prerequisites

- Python 3.6 or higher

### Required Libraries

Install the required libraries using pip:

```bash
pip install streamlit opencv-python ultralytics pillow
```

### Clone the Repository

```bash
git clone https://github.com/your_username/your_repository.git
cd your_repository
```

### Model File

Ensure you have the YOLO model file `best.pt` in the project directory. This model should be trained to detect the classes: Mask, Safety Vest, and Hardhat.

If you don't have a trained model, you can:
- Train your own model using the Ultralytics YOLO framework.
- Use a pre-trained model that includes the required classes.

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

### Using the App

1. Open the app in your web browser (usually at `http://localhost:8501`).
2. Upload a video file by clicking on the "Choose a file" button.
3. Adjust the confidence threshold in the sidebar if necessary.
4. The app will process the video and display the frames with detections.

## File Structure

```plaintext
├── app.py          # Main application script
├── style.css       # Custom CSS for styling the app
├── best.pt         # YOLO model file (not included)
├── README.md       # This README file
```

## Customization

### Safety Items

Modify the `SAFETY_ITEMS` dictionary in `app.py` to change the items you want to detect.

```python
SAFETY_ITEMS = {"Mask": False, "Safety Vest": False, "Hardhat": False}
```

### Predefined Colors

Adjust the `PREDEFINED_COLORS` list to change the bounding box colors.

```python
PREDEFINED_COLORS = [
    (0, 0, 255),  # Red
    (0, 255, 0),  # Green
    (255, 0, 0),  # Blue
    # Add more colors if needed
]
```

## Dependencies

- Streamlit
- OpenCV
- Ultralytics YOLO
- Pillow

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or suggestions, please contact:

- **Email**: your_email@example.com
- **GitHub**: [your_username](https://github.com/your_username)

## Acknowledgments

- **Ultralytics YOLO** for the object detection framework.
- **Streamlit** for the easy-to-use web app framework.
