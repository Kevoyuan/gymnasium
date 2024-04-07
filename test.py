from PIL import Image
import numpy as np  # Assuming your frames are numpy arrays

def create_gif_with_pillow(frames, filepath='animation.gif', total_duration_ms=1000):
    if not frames:
        raise ValueError("The frames list is empty.")

    num_frames = len(frames)
    # Calculate the duration each frame is shown to fit the total_duration_ms
    duration_per_frame_ms = total_duration_ms / num_frames  # Duration in milliseconds
    
    # Convert numpy array frames to PIL Images
    pil_images = [Image.fromarray(frame) for frame in frames]

    # Save the frames as a GIF
    pil_images[0].save(
        filepath,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration_per_frame_ms,  # Duration for each frame
        loop=1
    )

# Example usage:
frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(10)]
create_gif_with_pillow(frames, 'animation.gif', 1000)
