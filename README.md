# Description
(WIP)
Boids simulation (http://www.red3d.com/cwr/boids/) in python using pyglet, imgui and numba.

It is not ideal to use python for high demanding graphics, and lets not say for $O(n^2)$ simulations. I found a decent setup by combining `pyglet` for the graphics, `imgui` for the user interface and `numba` to speed up the calculations.


https://youtu.be/xzHkp6OICtc

The boids simulation does not seem to match completely the expected behaviour. More work is needed there.

# Reproduce
Create conda environment:
```cmd
conda env create -f environment.yml
conda activate graphics
```
Run main script:
```python
python src/main.py
```
