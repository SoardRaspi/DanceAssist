1. The two Agent folders are for music generation which require the 'requirements.txt' file. Run the main.py after running 'pip install requirements.txt'
2. The Eye Tracking folder contains files for the eye tracking demo which require the 'requirements_2.txt' file. Run the main.py after running 'pip install requirements_2.txt'

The main.py in Agent folders generate music of a single frequency at a time and so I suggest to NOT use headphones/earphones while running the main.py for the same.

The main.py in Eye Tracking folder is for tracking where the eye is looking. When running the code, the screen will be displayed with a red circle. Look at the circle and the red squares are the model's predictions. The model has a very basic design and so, works better if you first stabalize the model's predictions by looking at the camera initially and then moving only the eyes and keeping head still as much as possible.
