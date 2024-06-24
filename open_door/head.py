'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-04-17 18:53:37
Version: v1
File: 
Brief: 
'''

import serial

class Head(object):
    def __init__(self,port,baudrate):
        self.port = port
        self.baudrate = baudrate
        self.connect(self.port,self.baudrate)

    def connect(self,port,baudrate):
        print(f'==========\nHead connecting...')
        self.ser = serial.Serial(port, baudrate)
        print(f'Head connected\n==========')

    def to_hex(self,value):
        value_hex = hex(value)
        value_int = int(value_hex, 16)
        return value_int

    def to_low(self,value):
        low_byte = value & 0xFF
        return self.to_hex(low_byte)

    def to_high(self,value):
        high_byte = (value >> 8) & 0xFF
        return self.to_hex(high_byte)

    def get_angle(self,response):
        # response be like: b'UU\x06\x15\x01\x01\xec\x01'
        low_byte = response[-2]  # Get the second-to-last byte (the lower eight bits)
        high_byte = response[-1]  # Get the last byte (the higher eight bits)
        angle = (high_byte << 8) | low_byte  # Combaine the lower and higher eight bits
        # print("Angle (Decimal):", angle)
        return angle

    def get_servo_angle(self,servo_id,if_p=False):
        cmd = bytearray([0x55, 0x55, 0x04, 0x15, 0x01, self.to_hex(servo_id)])
        self.ser.write(cmd)
        response = self.ser.read(8)
        # print(f'response:\n{response}')
        angle = self.get_angle(response)
        if if_p:
            print(f"Get Servo_{servo_id} Angle: {angle}")
        return angle

    def servo_move(self,time, servo_id, angle,if_p=False):
        '''
        @param time: the moving time(speed)
        @param servo_id: id=1: moving vertically; id=2: moving horizonally
        @param: angle: 400-1000(id=1,400 is the lowest); 0-1000(id=2)
        '''
        cmd = bytearray([0x55, 0x55, 0x08, 0x03, 0x01, self.to_low(time), self.to_high(time), self.to_hex(servo_id), self.to_low(angle), self.to_high(angle)])
        self.ser.write(cmd)
        if if_p:
            print(f"Control Servo_{servo_id} Move to Angle_{angle}")

    def disconnect(self):
        self.ser.close()

if __name__ == "__main__":
    port = 'COM3' # Linux:'/dev/ttyUSB0' Win: 'COM3'
    baudrate = 9600
    head = Head(port,baudrate)
    head.get_servo_angle(1,if_p=True)
    head.get_servo_angle(2,if_p=True)
    head.servo_move(1000, 1, 400,if_p=True)
    head.servo_move(1000, 2, 500,if_p=True)

    # disconnect
    head.disconnect()