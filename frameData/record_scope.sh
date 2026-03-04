#!/bin/bash

# Create a filename with the current date and time in the Documents folder
FILENAME="$HOME/Documents/scope_recording_$(date +%Y%m%d_%H%M%S).mp4"

echo "======================================================"
echo "Starting live preview..."
echo "Recording to: $FILENAME"
echo "======================================================"
echo "PRESS Ctrl+C IN THIS TERMINAL TO STOP RECORDING SAFELY"
echo "======================================================"

# Run gstreamer: 
# - The 'tee name=t' splits the video feed into two branches.
# - 't. ! queue ! videoconvert ! ximagesink' shows the live preview on your screen.
# - 't. ! queue ! videoconvert ! x264enc tune=zerolatency ! mp4mux ! filesink' encodes and saves the video.
gst-launch-1.0 -e v4l2src device=/dev/video0 ! tee name=t \
  t. ! queue ! videoconvert ! ximagesink \
  t. ! queue ! videoconvert ! x264enc tune=zerolatency ! mp4mux ! filesink location="$FILENAME"
