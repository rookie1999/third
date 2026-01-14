from piper_robot import PiperRobot
piper = PiperRobot(can_port="can0")
piper.enable()
piper.go_zero(speed=100)