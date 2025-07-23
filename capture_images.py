import cv2
import os
import time
from tkinter import Tk
from tkinter.filedialog import askdirectory

# === CONFIGURATION ===
LABEL = "normal"              # Change to "normal" when needed
CAPTURE_DELAY_MS = 200       # Delay between frames (in milliseconds)
IMAGE_WIDTH = 640            # Width of saved image (optional resize)
IMAGE_HEIGHT = 480           # Height of saved image (optional resize)
CAPTURE_KEY = 32             # Space bar
EXIT_KEY = 27                # ESC
# =======================

# Ask user to select a directory
def choose_directory():
    # Hide the root Tk window
    root = Tk()
    root.withdraw()
    directory = askdirectory(title="Select Directory to Save Images")
    return directory

# Ask user for the directory to save images
save_dir = choose_directory()

if not save_dir:
    print("❌ No directory selected. Exiting...")
    exit()

# Prepare subdirectory based on the label
save_dir = os.path.join(save_dir, LABEL.lower())
os.makedirs(save_dir, exist_ok=True)

# Initialize counter for image filenames
count = len(os.listdir(save_dir))

# Initialize camera
cap = cv2.VideoCapture(0)
print(f"📷 Capturing images for label: '{LABEL}'")
print("➡ Press SPACE to capture an image, ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame.")
        break

    # Resize if needed
    frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # Show capture text
    cv2.putText(frame, f"Capturing: {LABEL}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Image Capture", frame)

    key = cv2.waitKey(CAPTURE_DELAY_MS) & 0xFF

    if key == EXIT_KEY:
        print("🛑 Exiting capture.")
        break
    elif key == CAPTURE_KEY:
        filename = f"{LABEL}_{count}.jpg"
        path = os.path.join(save_dir, filename)
        cv2.imwrite(path, frame)
        print(f"✅ Saved: {path}")
        count += 1

cap.release()
cv2.destroyAllWindows()
