from time import sleep

import pigpio
import os

os.system("sudo pigpiod")
sleep(1)

pi = pigpio.pi()

pi.set_servo_pulsewidth(17, 0)
pi.set_servo_pulsewidth(18, 0)
